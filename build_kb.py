"""
build_kb.py
────────────
One-time setup script. Run this ONCE before starting the API.

Builds a multi-source FAISS knowledge base from three open medical datasets:

  1. GBaker/MedQA-USMLE-4-options  — 2 000 clinical vignette Q&A pairs
  2. qiaojin/PubMedQA               — 1 000 PubMed abstract-based Q&A
  3. medmcqa                        — 2 000 Indian medical entrance MCQs

Total: up to 5 000 high-quality medical evidence passages.

Usage:
    python build_kb.py

Output:
    data/kb.index       ← FAISS binary index
    data/kb_meta.json   ← Parallel metadata (question + ground-truth answer)
"""

import json
import os
import sys
from collections import Counter
from typing import List, Tuple

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import config

os.makedirs("data", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────── #
#  Per-source loaders                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def _load_medqa(max_docs: int) -> Tuple[List[str], List[dict]]:
    """GBaker/MedQA-USMLE-4-options — clinical vignettes + correct answers."""
    print(f"  [1/3] MedQA-USMLE  (target: {max_docs} docs)...")
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    except Exception as e:
        print(f"         FAILED: {e}")
        return [], []

    passages, meta = [], []
    for i, row in enumerate(ds):
        if len(passages) >= max_docs:
            break
        q   = row.get("question", "").strip()
        ans = row.get("answer",   "").strip()
        if not q or not ans:
            continue
        passages.append(f"Question: {q} Answer: {ans}")
        meta.append({"question": q, "answer": ans, "source": "MedQA-USMLE"})

    print(f"         Loaded {len(passages)} passages")
    return passages, meta


def _load_pubmedqa(max_docs: int) -> Tuple[List[str], List[dict]]:
    """
    qiaojin/PubMedQA pqa_labeled — clinical trial abstract Q&A.
    Uses long_answer (the detailed answer from the abstract conclusion).
    """
    print(f"  [2/3] PubMedQA     (target: {max_docs} docs)...")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as e:
        print(f"         FAILED: {e}")
        return [], []

    passages, meta = [], []
    for i, row in enumerate(ds):
        if len(passages) >= max_docs:
            break
        q   = row.get("question",    "").strip()
        ans = row.get("long_answer", "").strip()
        decision = row.get("final_decision", "").strip()  # yes / no / maybe
        if not q or not ans:
            continue
        # Include the yes/no decision so the KB can contradict false yes/no claims
        text = f"Question: {q} Answer ({decision}): {ans}"
        passages.append(text)
        meta.append({
            "question": q,
            "answer"  : ans[:500],
            "decision": decision,
            "source"  : "PubMedQA",
        })

    print(f"         Loaded {len(passages)} passages")
    return passages, meta


def _load_medmcqa(max_docs: int) -> Tuple[List[str], List[dict]]:
    """
    medmcqa — 194 k medical entrance MCQs with explanations.
    We only index rows that have an explanation (the most informative ones).
    """
    print(f"  [3/3] MedMCQA      (target: {max_docs} docs)...")
    try:
        ds = load_dataset("medmcqa", split="train")
    except Exception as e:
        print(f"         FAILED: {e}")
        return [], []

    option_keys = ["opa", "opb", "opc", "opd"]

    passages, meta = [], []
    for row in ds:
        if len(passages) >= max_docs:
            break
        q     = (row.get("question") or "").strip()
        exp   = (row.get("exp")      or "").strip()
        cop   = row.get("cop", -1)                     # correct option index (0-3)
        opts  = [(row.get(k) or "") for k in option_keys]

        # Require an explanation; skip rows with missing data
        if not q or not exp or cop not in range(4):
            continue

        correct_opt = opts[cop].strip()
        if not correct_opt:
            continue

        text = f"Question: {q} Correct answer: {correct_opt}. Explanation: {exp}"
        passages.append(text)
        meta.append({
            "question": q,
            "answer"  : f"{correct_opt}. {exp}"[:500],
            "subject" : (row.get("subject_name") or ""),
            "source"  : "MedMCQA",
        })

    print(f"         Loaded {len(passages)} passages")
    return passages, meta


# ─────────────────────────────────────────────────────────────────────────── #
#  Main builder                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def build():
    print("\nBuilding multi-source medical knowledge base...")
    print(f"Targets: {config.KB_SOURCES}\n")

    # ── 1. Load all sources ───────────────────────────────────────────────
    all_passages: List[str] = []
    all_meta:     List[dict] = []

    for loader, key in [
        (_load_medqa,    "medqa_usmle"),
        (_load_pubmedqa, "pubmedqa"),
        (_load_medmcqa,  "medmcqa"),
    ]:
        max_n = config.KB_SOURCES.get(key, 0)
        if max_n <= 0:
            continue
        p, m = loader(max_n)
        all_passages.extend(p)
        all_meta.extend(m)

    if not all_passages:
        print("All downloads failed — falling back to seed KB...")
        _build_seed_kb()
        return

    # Respect hard cap
    all_passages = all_passages[: config.KB_MAX_DOCS]
    all_meta     = all_meta    [: config.KB_MAX_DOCS]

    print(f"\nTotal passages for indexing: {len(all_passages)}")

    # ── 2. Embed ──────────────────────────────────────────────────────────
    print(f"\nLoading embedding model: {config.EMBEDDING_MODEL}")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    print("Embedding passages (this takes ~2-4 min on CPU)...")
    embeddings = embedder.encode(
        all_passages,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # ── 3. Build FAISS index ──────────────────────────────────────────────
    print("\nBuilding FAISS IndexFlatIP index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, config.KB_INDEX_PATH)
    with open(config.KB_META_PATH, "w") as f:
        json.dump(all_meta, f, indent=2)

    # Source breakdown
    counts = Counter(m["source"] for m in all_meta)
    print(f"\nKnowledge base built successfully!")
    print(f"   Index  : {config.KB_INDEX_PATH}  ({index.ntotal} vectors, dim={dim})")
    print(f"   Meta   : {config.KB_META_PATH}")
    print(f"   Sources: {dict(counts)}")




def _build_seed_kb():
    """
    Fallback: hardcoded seed knowledge base of 30 reliable medical facts.
    Used only if all downloads fail (e.g. no internet).
    Still gives a working demo with limited coverage.
    """
    seed_facts = [
        ("What is the first-line treatment for Type 2 diabetes?",
         "Metformin is the first-line pharmacological treatment for Type 2 diabetes, combined with lifestyle modification."),
        ("What are symptoms of appendicitis?",
         "Classic symptoms of appendicitis include periumbilical pain migrating to the right lower quadrant (McBurney's point), nausea, vomiting, fever, and rebound tenderness."),
        ("What is the mechanism of aspirin as an antiplatelet?",
         "Aspirin irreversibly inhibits cyclooxygenase (COX-1 and COX-2), preventing thromboxane A2 synthesis, which reduces platelet aggregation."),
        ("What is the maximum safe daily dose of acetaminophen in adults?",
         "The maximum safe daily dose of acetaminophen (paracetamol) in healthy adults is 4 grams (4000 mg) per day. In patients with liver disease or alcohol use, 2 grams per day is recommended."),
        ("What is the mechanism of action of metformin?",
         "Metformin primarily reduces hepatic glucose production by activating AMPK, which inhibits gluconeogenesis. It also improves peripheral insulin sensitivity."),
        ("What are the signs of myocardial infarction?",
         "Signs of MI include crushing chest pain radiating to the left arm or jaw, diaphoresis, shortness of breath, nausea, and ST elevation on ECG."),
        ("What is the first-line treatment for hypertension?",
         "First-line antihypertensives include thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers. Choice depends on patient comorbidities."),
        ("What are contraindications for metformin?",
         "Metformin is contraindicated in severe renal impairment (eGFR < 30), active liver disease, and conditions predisposing to lactic acidosis such as heart failure or sepsis."),
        ("What is the mechanism of warfarin?",
         "Warfarin inhibits Vitamin K epoxide reductase, reducing synthesis of clotting factors II, VII, IX, and X, and proteins C and S."),
        ("What is serotonin syndrome?",
         "Serotonin syndrome is a drug reaction from excess serotonin, presenting with the triad of neuromuscular abnormalities (clonus, hyperreflexia), autonomic dysfunction (tachycardia, hyperthermia), and altered mental status."),
        ("What drugs cause serotonin syndrome?",
         "Serotonin syndrome is caused by combinations of serotonergic drugs including SSRIs, SNRIs, MAOIs, tramadol, linezolid, triptans, and St. John's Wort."),
        ("What are the types of diabetes mellitus?",
         "Type 1 diabetes is autoimmune destruction of pancreatic beta cells causing absolute insulin deficiency. Type 2 diabetes is insulin resistance with progressive beta cell failure."),
        ("What is the treatment for anaphylaxis?",
         "The first-line treatment for anaphylaxis is intramuscular epinephrine (adrenaline) 0.3-0.5 mg in the lateral thigh. Secondary treatments include antihistamines and corticosteroids."),
        ("What is atrial fibrillation?",
         "Atrial fibrillation is an irregular, often rapid heart rate caused by chaotic electrical signals in the atria, presenting with palpitations, dyspnea, and irregular pulse."),
        ("What is the difference between systolic and diastolic heart failure?",
         "Systolic (HFrEF) involves reduced ejection fraction (< 40%) from impaired contraction. Diastolic (HFpEF) involves preserved ejection fraction with impaired relaxation."),
        ("What is the mechanism of ACE inhibitors?",
         "ACE inhibitors block angiotensin-converting enzyme, preventing conversion of angiotensin I to angiotensin II, reducing vasoconstriction, aldosterone release, and blood pressure."),
        ("What causes COPD?",
         "COPD is caused primarily by cigarette smoking (80-90% of cases) causing chronic airway inflammation, leading to emphysema and chronic bronchitis."),
        ("What is pneumonia?",
         "Pneumonia is an infection of the lung parenchyma presenting with fever, cough, sputum production, pleuritic chest pain, and consolidation on chest X-ray."),
        ("What is the role of insulin in diabetes?",
         "Insulin is required for Type 1 diabetes as there is absolute deficiency. In Type 2 diabetes, insulin is used when oral agents fail to achieve glycaemic control."),
        ("What is the mechanism of beta-blockers?",
         "Beta-blockers competitively antagonise catecholamines at beta-adrenergic receptors, reducing heart rate, contractility, and blood pressure."),
        ("What is DVT?",
         "Deep vein thrombosis (DVT) is a blood clot in a deep vein, usually in the leg, presenting with unilateral calf pain, swelling, and warmth."),
        ("How is pulmonary embolism diagnosed?",
         "PE is diagnosed using CT pulmonary angiography (CTPA) as gold standard, supported by D-dimer, V/Q scan, and clinical probability scores like Wells criteria."),
        ("What is sepsis?",
         "Sepsis is life-threatening organ dysfunction caused by dysregulated host response to infection, defined by a SOFA score increase of ≥2."),
        ("What is the treatment for sepsis?",
         "Sepsis management follows the Surviving Sepsis Campaign bundle: early broad-spectrum antibiotics within 1 hour, IV fluid resuscitation, blood cultures, and vasopressors if hypotensive."),
        ("What are opioid side effects?",
         "Common opioid side effects include constipation, nausea, sedation, respiratory depression, pruritus, and urinary retention. Respiratory depression is the most dangerous."),
        ("What is naloxone used for?",
         "Naloxone is an opioid antagonist used to reverse opioid overdose, rapidly restoring respiratory drive. It has a shorter half-life than most opioids, so redosing may be needed."),
        ("What is the mechanism of statins?",
         "Statins inhibit HMG-CoA reductase, the rate-limiting enzyme in cholesterol synthesis, reducing LDL cholesterol and stabilising atherosclerotic plaques."),
        ("What are signs of stroke?",
         "Stroke presents with sudden onset facial droop, arm weakness, speech difficulty (FAST acronym), vision changes, severe headache, or ataxia."),
        ("What is the treatment for ischemic stroke?",
         "Ischemic stroke treatment includes IV thrombolysis with alteplase within 4.5 hours of onset, mechanical thrombectomy for large vessel occlusion within 24 hours, and antiplatelet therapy."),
        ("What is hypothyroidism?",
         "Hypothyroidism is deficiency of thyroid hormone presenting with fatigue, weight gain, cold intolerance, constipation, bradycardia, and elevated TSH with low T4."),
    ]

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    passages = [f"Question: {q} Answer: {a}" for q, a in seed_facts]
    meta     = [{"question": q, "answer": a, "source": "Seed-KB"} for q, a in seed_facts]

    embeddings = embedder.encode(passages, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, config.KB_INDEX_PATH)
    with open(config.KB_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅  Seed KB built ({len(seed_facts)} facts)")


if __name__ == "__main__":
    if os.path.exists(config.KB_INDEX_PATH):
        print(f"⚡  KB already exists at {config.KB_INDEX_PATH}")
        resp = input("   Rebuild? [y/N]: ").strip().lower()
        if resp != "y":
            print("   Skipping.")
            sys.exit(0)

    build()