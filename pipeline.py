"""
pipeline.py
────────────
The complete MedCAFAS detection engine — 3 layers, fully CPU-based, no training needed.

Layer 1 — Self-Consistency   : Sample Ollama 3× → measure semantic drift
Layer 2 — Retrieval          : FAISS top-k → cosine similarity vs. answer
Layer 3 — NLI Critic         : cross-encoder/nli-deberta-v3-base  → entailment

All components are lazy-loaded once and cached as module-level singletons.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pickle
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
#  Singletons (loaded once at first use)                                     #
# ─────────────────────────────────────────────────────────────────────────── #

_embedder: Optional[SentenceTransformer] = None
_nli_model: Optional[CrossEncoder]       = None
_faiss_index: Optional[faiss.Index]      = None
_kb_meta: Optional[List[Dict]]           = None
_bm25_data:   Optional[Dict]             = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        logger.info(f"Loading NLI critic: {config.NLI_MODEL}")
        _nli_model = CrossEncoder(config.NLI_MODEL, max_length=512)
    return _nli_model


def _get_kb() -> Tuple[faiss.Index, List[Dict]]:
    global _faiss_index, _kb_meta
    if _faiss_index is None:
        try:
            _faiss_index = faiss.read_index(config.KB_INDEX_PATH)
            with open(config.KB_META_PATH) as f:
                _kb_meta = json.load(f)
            logger.info(f"KB loaded: {_faiss_index.ntotal} docs")
        except FileNotFoundError:
            raise RuntimeError(
                "Knowledge base not found. Please run:  python build_kb.py"
            )
    return _faiss_index, _kb_meta


def _get_bm25() -> Dict:
    """Load the BM25 index built by build_kb.py.  Returns empty dict if missing."""
    global _bm25_data
    if _bm25_data is None:
        try:
            with open(config.BM25_INDEX_PATH, "rb") as f:
                _bm25_data = pickle.load(f)
            logger.info(f"BM25 index loaded: {len(_bm25_data.get('passages', []))} docs")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"BM25 index not available, falling back to cosine-only ({e})")
            _bm25_data = {}
    return _bm25_data


# ─────────────────────────────────────────────────────────────────────────── #
#  Data classes                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class ClaimResult:
    """
    Per-claim verification result from the FACTSCORE-style decomposition layer.
    Each atomic sentence in the LLM answer is checked independently against the KB.
    """
    claim:         str          # The atomic claim text
    best_evidence: str          # Best supporting/contradicting KB snippet
    entailment:    float        # [0, 1]  how well KB supports this claim
    contradiction: float        # [0, 1]  how strongly KB contradicts this claim
    retrieval_sim: float        # [0, 1]  FAISS cosine sim for this claim
    verdict:       str          # "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED"


@dataclass
class LayerResult:
    consistency_score: float    # [0,1]  higher = more consistent = less risky
    consistency_risk:  float    # [0,1]  1 - consistency (used for aggregation)
    retrieval_score:   float    # [0,1]  hybrid BM25+cosine sim vs. best KB doc
    retrieval_risk:    float    # [0,1]  1 - retrieval_score
    critic_entailment: float    # [0,1]  NLI entailment prob (higher = supported)
    critic_risk:       float    # [0,1]  1 - critic_entailment
    temporal_risk:     float    # [0,1]  1.0 if future-year claim detected
    entity_risk:       float    = 0.0   # [0,1]  fraction of key terms missing from evidence
    claim_breakdown:   List[ClaimResult] = field(default_factory=list)
    samples:           List[str] = field(default_factory=list)
    citations:         List[Dict] = field(default_factory=list)


@dataclass
class PredictionResult:
    question:        str
    answer:          str
    risk_score:      float         # Final [0, 1]
    risk_flag:       str           # "LOW" | "CAUTION" | "HIGH"
    confidence:      float         # 1 - risk_score
    explanation:     str
    breakdown:       Dict
    citations:       List[Dict]
    claim_breakdown: List[Dict]    # Per-claim FACTSCORE-style results
    temporal_flags:  List[Dict]    # Future-year claims detected
    latency_ms:      float


# ─────────────────────────────────────────────────────────────────────────── #
#  Layer 1 — Self-Consistency                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def _ask_ollama(question: str, temperature: float = 0.0) -> str:
    """
    Single synchronous call to local Ollama API.
    Returns the model's answer text.
    """
    system_msg = (
        "You are a knowledgeable medical assistant. "
        "Provide accurate, concise answers based on established clinical guidelines. "
        "If you are not sure, say so explicitly."
    )
    payload = {
        "model"  : config.OLLAMA_MODEL,
        "prompt" : f"{system_msg}\n\nQuestion: {question}\n\nAnswer:",
        "stream" : False,
        "options": {"temperature": temperature, "num_predict": 300},
    }
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. "
            "Is it running?  Run: ollama serve"
        )


def layer1_consistency(question: str) -> Tuple[str, List[str], float]:
    """
    Sample the LLM NUM_SAMPLES times and measure semantic consistency.

    Returns:
        primary_answer  — The deterministic (temp=0) answer used downstream
        samples         — All sampled answers
        consistency     — [0, 1] semantic similarity across samples
    """
    # Primary answer at temperature 0 (deterministic)
    primary = _ask_ollama(question, temperature=0.0)

    # Diverse samples to test consistency
    samples = [primary]
    for _ in range(config.NUM_SAMPLES - 1):
        samples.append(_ask_ollama(question, temperature=config.SAMPLE_TEMP))

    # Embed all samples
    embedder = _get_embedder()
    embeddings = embedder.encode(samples, normalize_embeddings=True)

    # Mean pairwise cosine similarity
    sim_matrix = cosine_similarity(embeddings)
    n = len(samples)
    # Upper triangle only (no self-similarity)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sims  = [float(sim_matrix[i][j]) for i, j in pairs]
    consistency = float(np.mean(sims)) if sims else 1.0

    logger.info(f"Layer 1 — Consistency: {consistency:.3f}")
    return primary, samples, consistency


# ─────────────────────────────────────────────────────────────────────────── #
#  Layer 2 — Retrieval Verification                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _hybrid_scores(
    query: str, distances: np.ndarray, indices: np.ndarray, meta: List[Dict]
) -> List[Tuple[float, int]]:
    """
    Combine FAISS cosine similarities with BM25 lexical scores.
    Returns list of (hybrid_score, meta_index) sorted descending.
    """
    bm25_data = _get_bm25()
    bm25_boost: Dict[int, float] = {}
    if bm25_data and "bm25" in bm25_data:
        tokens = query.lower().split()
        raw_scores = bm25_data["bm25"].get_scores(tokens)   # all KB docs
        bm25_max = float(raw_scores.max())
        if bm25_max > 0:
            for idx in indices[0]:
                if 0 <= int(idx) < len(raw_scores):
                    bm25_boost[int(idx)] = float(raw_scores[int(idx)]) / bm25_max

    w = config.BM25_WEIGHT if bm25_boost else 0.0
    candidates: List[Tuple[float, int]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        cosine = float(dist)
        bm25_s = bm25_boost.get(int(idx), 0.0)
        hybrid = (1.0 - w) * cosine + w * bm25_s
        candidates.append((hybrid, int(idx)))

    candidates.sort(reverse=True)
    return candidates


def _mmr_rerank(
    query_emb: np.ndarray,
    candidates: List[Tuple[float, int]],
    meta: List[Dict],
    top_k: int,
    lam: float,
) -> List[Tuple[float, int]]:
    """
    Max-Marginal Relevance (MMR) re-ranking to diversify retrieved evidence.

    Reduces redundancy in top-k results by penalising candidates that are
    too similar to already-selected documents.  This ensures the NLI critic
    sees diverse evidence rather than three variations of the same paragraph,
    directly addressing the FN-Ret error pattern (88.9% of errors).

    MMR(d) = λ · Relevance(d, q) - (1-λ) · max_selected Sim(d, s)

    Args:
        query_emb  – normalised embedding of the query (1, dim)
        candidates – (hybrid_score, meta_idx) sorted by relevance
        meta       – KB metadata for passage text
        top_k      – how many to select
        lam        – trade-off: 1.0 = pure relevance, 0.0 = pure diversity

    Returns:
        Re-ranked list of (hybrid_score, meta_idx) of length top_k.
    """
    if len(candidates) <= top_k:
        return candidates

    embedder = _get_embedder()

    # Embed all candidate passages once
    passages = []
    for _, idx in candidates:
        doc = meta[idx]
        passages.append(doc.get("answer", "") or doc.get("question", ""))
    cand_embs = embedder.encode(passages, normalize_embeddings=True,
                                convert_to_numpy=True).astype(np.float32)

    # Relevance scores normalised to [0, 1]
    rel_scores = np.array([sc for sc, _ in candidates], dtype=np.float32)
    max_rel = rel_scores.max() if rel_scores.max() > 0 else 1.0
    rel_norm = rel_scores / max_rel

    selected_indices: List[int] = []   # positions within `candidates`
    selected_embs: List[np.ndarray] = []

    for _ in range(top_k):
        best_idx = -1
        best_mmr = -float("inf")

        for i in range(len(candidates)):
            if i in selected_indices:
                continue

            relevance = rel_norm[i]

            # Max similarity to already-selected docs
            if selected_embs:
                sims = cosine_similarity(
                    cand_embs[i:i+1], np.array(selected_embs)
                )[0]
                max_sim = float(sims.max())
            else:
                max_sim = 0.0

            mmr = lam * relevance - (1.0 - lam) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        selected_embs.append(cand_embs[best_idx])

    return [candidates[i] for i in selected_indices]


def layer2_retrieval(answer: str) -> Tuple[float, List[Dict]]:
    """
    Embed the LLM answer and retrieve top-k documents from the KB using
    hybrid BM25 + cosine similarity scoring.

    Strategy:
      1. FAISS fetches BM25_CANDIDATES (20) nearest neighbours by cosine sim.
      2. BM25 lexical scores are computed for those candidates and blended in.
      3. Top-k by hybrid score are returned as citations.

    Returns:
        retrieval_score  — [0, 1] hybrid score of best document
        citations        — Top-k retrieved docs
    """
    index, meta = _get_kb()
    embedder    = _get_embedder()

    answer_emb = embedder.encode(
        [answer], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    n_candidates = max(config.TOP_K, getattr(config, 'MMR_CANDIDATES', config.BM25_CANDIDATES))
    distances, indices = index.search(answer_emb, n_candidates)

    candidates = _hybrid_scores(answer, distances, indices, meta)

    # MMR re-ranking: diversify evidence so NLI sees varied perspectives
    lam = getattr(config, 'MMR_LAMBDA', 1.0)
    if lam < 1.0:
        candidates = _mmr_rerank(answer_emb, candidates, meta, config.TOP_K, lam)
    else:
        candidates = candidates[:config.TOP_K]

    citations: List[Dict] = []
    sims: List[float] = []
    for hybrid, idx in candidates:
        sims.append(hybrid)
        doc = meta[idx]
        citations.append({
            "source"    : doc.get("source", "KB"),
            "question"  : doc.get("question", ""),
            "answer"    : doc.get("answer", "")[:200],
            "similarity": round(hybrid, 4),
        })

    retrieval_score = float(max(sims)) if sims else 0.0
    logger.info(f"Layer 2 — Hybrid top score: {retrieval_score:.3f}")
    return retrieval_score, citations


def layer2_retrieval_single(query: str) -> Tuple[float, List[Dict]]:
    """
    Retrieve top-k KB documents for a single query string using hybrid scoring
    with MMR diversity re-ranking.
    Used for per-claim retrieval in the claim decomposition loop.
    """
    index, meta = _get_kb()
    embedder    = _get_embedder()

    query_emb = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    n_candidates = max(config.TOP_K, getattr(config, 'MMR_CANDIDATES', config.BM25_CANDIDATES))
    distances, indices = index.search(query_emb, n_candidates)

    candidates = _hybrid_scores(query, distances, indices, meta)

    # MMR re-ranking for per-claim retrieval too
    lam = getattr(config, 'MMR_LAMBDA', 1.0)
    if lam < 1.0:
        candidates = _mmr_rerank(query_emb, candidates, meta, config.TOP_K, lam)
    else:
        candidates = candidates[:config.TOP_K]

    citations: List[Dict] = []
    sims: List[float] = []
    for hybrid, idx in candidates:
        sims.append(hybrid)
        doc = meta[idx]
        citations.append({
            "source"    : doc.get("source", "KB"),
            "question"  : doc.get("question", ""),
            "answer"    : doc.get("answer", "")[:200],
            "similarity": round(hybrid, 4),
        })

    retrieval_score = float(max(sims)) if sims else 0.0
    return retrieval_score, citations


# ─────────────────────────────────────────────────────────────────────────── #
#  Layer 2b — Medical Entity Overlap Check                                   #
# ─────────────────────────────────────────────────────────────────────────── #

# Common pharmaceutical name suffixes — identify drug mentions reliably
_DRUG_SUFFIX_RE = re.compile(
    r"\b\w{4,}(?:olol|opril|sartan|statin|azide|mycin|cillin|oxacin"
    r"|zepam|azine|tidine|lukast|triptan|dronate|mab|\bnib|zumab|prazole"
    r"|cycline|vir\b|navir|mivir|tidine)\b",
    re.IGNORECASE,
)



# Common drug names that do NOT match the suffix regex but are frequently
# swapped in hallucination perturbations (warfarin↔heparin, etc.).
_DRUG_NAMES: frozenset = frozenset({
    "warfarin", "heparin", "aspirin", "clopidogrel", "metformin", "glipizide",
    "metoprolol", "atenolol", "amoxicillin", "vancomycin", "lisinopril", "losartan",
    "furosemide", "hydrochlorothiazide", "morphine", "fentanyl", "epinephrine",
    "atropine", "insulin", "prednisone", "prednisolone", "dexamethasone",
    "ibuprofen", "naproxen", "acetaminophen", "paracetamol", "digoxin", "amiodarone",
    "spironolactone", "amlodipine", "diltiazem", "verapamil", "nitroglycerin",
    "adenosine", "dopamine", "dobutamine", "norepinephrine", "phenylephrine",
    "haloperidol", "quetiapine", "olanzapine", "risperidone", "clozapine",
    "lithium", "valproate", "carbamazepine", "phenytoin", "levetiracetam",
    "gabapentin", "tramadol", "codeine", "oxycodone", "naloxone", "methadone",
    "buprenorphine", "clonidine", "hydralazine", "nifedipine", "captopril",
    "enalapril", "ramipril", "candesartan", "valsartan", "rosuvastatin",
    "simvastatin", "pravastatin", "ezetimibe", "colchicine", "allopurinol",
    "methotrexate", "hydroxychloroquine", "sulfasalazine", "azathioprine",
    "cyclosporine", "tacrolimus", "tamoxifen", "letrozole", "anastrozole",
    "cisplatin", "carboplatin", "paclitaxel", "docetaxel", "doxorubicin",
    "cyclophosphamide", "fluorouracil", "gemcitabine", "imatinib",
})

# Curated whitelist of medically-relevant acronyms only.
# Generic statistical / prose acronyms (BMI, CI, RR, HR, OR, SD, etc.) are
# deliberately excluded because they appear in both hallucinated AND correct
# answers, adding noise without discrimination power.
_MEDICAL_ACRONYMS: frozenset = frozenset({
    # Drug classes
    "ssri", "snri", "maoi", "nsaid", "ace", "arb", "ccb", "ppi", "tca",
    "statin", "ocp", "hrt",
    # Diseases / syndromes
    "copd", "chf", "ckd", "afib", "dvt", "pe", "mi", "acs", "cva", "tia",
    "hiv", "hbv", "hcv", "tb", "uti", "sti", "mrsa", "ards", "sle", "dm",
    "t2dm", "t1dm", "gerd", "ibs", "ibd",
    # Procedures / investigations
    "ecg", "ekg", "mri", "ct", "cbc", "bmp", "cmp", "bnp", "inr", "aptt",
    "ptt", "ast", "alt", "gfr", "egfr", "bun", "hba1c", "a1c", "ldl", "hdl",
    "echo", "tee", "tte",
    # Drug name fragments (often appear as upper-case in medical text)
    "pcr", "icu", "nicu", "er", "or",
})


def extract_medical_terms(text: str) -> List[str]:
    """
    Precision-focused extraction of medical entities.

    Includes ONLY terms that have strong discriminative value:
      - Drug names matched by pharmaceutical-suffix regex  (metformin, atorvastatin …)
      - Explicit dosage patterns                           (500 mg, 10 ml, 2.5 mcg …)
      - Whitelisted clinical acronyms                      (SSRI, NSAID, COPD …)

    Deliberately EXCLUDED (high noise, appear equally in correct and hallucinated text):
      - All-caps generic acronyms (BMI, CI, RR, HR, OR, SD …)
      - Capitalised words at sentence starts
      - Statistical terminology
    """
    terms: List[str] = []

    # 1. Drug suffix pattern (pharmaceutical-name endings)
    terms.extend(m.group().lower() for m in _DRUG_SUFFIX_RE.finditer(text))

    # 2. Dosage patterns: numbers + units (highly specific to clinical text)
    terms.extend(
        m.group().lower()
        for m in re.finditer(
            r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|ml|mmol|g\b|units?|iu)\b",
            text, re.IGNORECASE,
        )
    )

    # 3. Only add ALL-CAPS acronyms that are in the curated medical whitelist
    for word in re.findall(r'\b[A-Z]{2,6}\b', text):
        if word.lower() in _MEDICAL_ACRONYMS:
            terms.append(word.lower())

    # 4. Named drug lookup — common drugs not caught by the suffix regex
    text_lower = text.lower()
    for drug in _DRUG_NAMES:
        if re.search(r'\b' + re.escape(drug) + r'\b', text_lower):
            terms.append(drug)

    # 5. Mid-sentence title-case words (≥5 chars) — disease / anatomy / procedure terms.
    #    Only capture words that follow a lowercase char + space (i.e. NOT sentence-
    #    initial capitalisation) to avoid noise from the first word of a sentence.
    for m in re.finditer(r'(?<=[a-z,;:]\s)[A-Z][a-z]{4,}\b', text):
        terms.append(m.group().lower())

    # Deduplicate, drop tokens shorter than 3 chars
    seen: set = set()
    result: List[str] = []
    for t in terms:
        t = t.strip()
        if len(t) >= 3 and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def layer2b_entity_check(answer: str, citations: List[Dict]) -> float:
    """
    Entity overlap check (Layer 2b).

    Hallucinated answers often introduce plausible-sounding entities (wrong
    drug names, fabricated diagnoses) that simply do not appear in the
    retrieved KB evidence — because the evidence discusses the *correct*
    entities instead.

    For every key medical term extracted from the answer, we check whether it
    appears anywhere in the pooled citation text.  The fraction of missing
    terms becomes the entity_risk signal.

    Returns:
        entity_risk  — [0, 1], 0 = all terms found, 1 = no terms found.
    """
    if not config.ENTITY_CHECK or not citations:
        return 0.0

    answer_terms = extract_medical_terms(answer)
    if len(answer_terms) < config.ENTITY_MIN_TERMS:
        return 0.0   # not enough identifiable terms — skip

    # Pool all citation text (lower-cased for matching)
    evidence_text = " ".join(
        (cit.get("answer", "") + " " + cit.get("question", "")).lower()
        for cit in citations
    )

    missing = [t for t in answer_terms if t not in evidence_text]
    entity_risk = len(missing) / len(answer_terms)

    logger.info(
        f"Layer 2b — {len(answer_terms)} terms, {len(missing)} missing from evidence "
        f"→ entity_risk={entity_risk:.3f}"
    )
    return round(entity_risk, 4)


# ─────────────────────────────────────────────────────────────────────────── #
#  Claim Decomposition (FACTSCORE-style per-claim verification)              #
# ─────────────────────────────────────────────────────────────────────────── #

def decompose_claims(text: str) -> List[str]:
    """
    Split an LLM answer into individual atomic medical claims.

    Strategy:
      1. Split on sentence-ending punctuation (primary).
      2. If the whole answer is a single long sentence, also split on
         semicolons and conjunctive commas for finer granularity.
      3. Filter fragments shorter than CLAIM_MIN_WORDS.
      4. Cap at MAX_CLAIMS to prevent quadratic NLI cost.
    """
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    claims = [s.strip() for s in raw if len(s.split()) >= config.CLAIM_MIN_WORDS]

    # If single long sentence — break on clause boundaries too
    if len(claims) <= 1 and len(text.split()) > 30:
        raw2   = re.split(
            r'[;]|\s*,\s*(?:and|but|however|while|whereas|although)\s+',
            text, flags=re.IGNORECASE
        )
        claims = [s.strip() for s in raw2 if len(s.split()) >= config.CLAIM_MIN_WORDS]

    if not claims:
        claims = [text.strip()]

    return claims[: config.MAX_CLAIMS]


# ─────────────────────────────────────────────────────────────────────────── #
#  Temporal Claim Detection                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def detect_temporal_claims(claims: List[str]) -> Tuple[float, List[Dict]]:
    """
    Detect claims that reference calendar years beyond the current date.
    Future-dated claims (e.g. "HORIZON-9 trial published in 2027") are
    near-certain hallucinations or fabricated events.

    Returns:
        temporal_risk  — 1.0 if any future year found, 0.0 otherwise
        flags          — list of dicts describing flagged claims
    """
    if not config.TEMPORAL_DETECTION:
        return 0.0, []

    current_year = datetime.now().year
    flags: List[Dict] = []

    for claim in claims:
        years = re.findall(r'\b((?:19|20)\d{2})\b', claim)
        for y_str in years:
            y = int(y_str)
            if y > current_year:
                flags.append({
                    "claim" : claim,
                    "year"  : y,
                    "reason": (
                        f"Year {y} is in the future "
                        f"(current year: {current_year}) — likely fabricated event"
                    ),
                })

    temporal_risk = 1.0 if flags else 0.0
    if flags:
        logger.info(f"Temporal — {len(flags)} future-year claim(s) detected: {[f['year'] for f in flags]}")

    return temporal_risk, flags


# ─────────────────────────────────────────────────────────────────────────── #
#  Layer 3 — NLI Critic                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def layer3_critic(
    answer: str,
    citations: List[Dict],
    question: str = "",
) -> Tuple[float, List[ClaimResult]]:
    """
    FACTSCORE-style per-claim NLI analysis with per-claim KB retrieval
    and softmax temperature scaling for paraphrase handling.

    For each atomic claim decomposed from the answer:
      1. Retrieve the best-matching KB documents for *that specific claim*.
      2. Run NLI (cross-encoder) to get raw logits.
      3. **Temperature Scaling**: If retrieval similarity is very high
         (>= SEM_SIM_SUPPORT_THRESH) but NLI reports neutral, apply a lower
         softmax temperature to the NLI logits.  This "squashes" the neutral
         and contradiction probabilities so entailment naturally rises.
         Unlike the previous flat boost, this preserves the relative ordering
         of the NLI logits and only amplifies the model's existing signal.
      4. Assign a verdict: SUPPORTED / UNSUPPORTED / CONTRADICTED.

    Also checks whether the original question/claim is contradicted by KB evidence —
    a strong signal that the user's input is itself a false medical claim.

    NLI labels for cross-encoder/nli-deberta-v3-base:
        index 0 = contradiction, index 1 = entailment, index 2 = neutral

    Returns:
        critic_score    — [0, 1] higher = better supported / less risky
        claim_results   — per-claim ClaimResult list (empty if no claims)
    """
    if not citations and not answer.strip():
        logger.warning("Layer 3 — Empty answer and citations; returning neutral 0.5")
        return 0.5, []

    nli    = _get_nli_model()
    claims = decompose_claims(answer)

    if not claims:
        return 0.5, []

    claim_results: List[ClaimResult] = []
    all_scores:    List[float]       = []

    for claim in claims:
        # Per-claim KB retrieval (finds the most relevant evidence for this claim)
        claim_sim, claim_cits = layer2_retrieval_single(claim)

        # Fall back to answer-level citations if per-claim retrieval comes up empty
        evidence_cits = claim_cits if claim_cits else citations

        if not evidence_cits:
            claim_results.append(ClaimResult(
                claim         = claim,
                best_evidence = "",
                entailment    = 0.5,
                contradiction = 0.0,
                retrieval_sim = 0.0,
                verdict       = "UNSUPPORTED",
            ))
            all_scores.append(0.5)
            continue

        # NLI: (KB evidence, claim) pairs
        pairs = [
            (cit["answer"], claim)
            for cit in evidence_cits
            if cit.get("answer")
        ]
        if not pairs:
            all_scores.append(0.5)
            continue

        # ── Short-claim fallback ──────────────────────────────────────────
        if len(claim.split()) < config.NLI_MIN_CLAIM_WORDS:
            proxy = min(0.50, max(0.35, claim_sim))
            claim_results.append(ClaimResult(
                claim         = claim,
                best_evidence = evidence_cits[0]["answer"][:200] if evidence_cits else "",
                entailment    = round(proxy, 4),
                contradiction = 0.0,
                retrieval_sim = round(claim_sim, 4),
                verdict       = "UNSUPPORTED",
            ))
            all_scores.append(proxy)
            logger.debug(f"Layer 3 — short claim ({len(claim.split())} words), proxy={proxy:.3f}: {claim[:60]}")
            continue

        scores = nli.predict(pairs, apply_softmax=False)   # raw logits
        scores = np.array(scores, dtype=np.float64)

        # ── Softmax temperature scaling ──────────────────────────────────
        # Default: standard temperature (T=1.0) → normal softmax.
        # When semantic similarity is high but NLI is ambiguous, lower T
        # to sharpen the distribution — amplifying the model's existing
        # entailment signal without inventing support that isn't there.
        temperature = 1.0
        # Only scale temperature when NLI's top class is entailment (amplify existing signal)
        top_class_is_entailment = False
        if len(scores) > 0:
            mean_logits = scores.mean(axis=0)
            top_class_is_entailment = int(mean_logits.argmax()) == 1

        if (claim_sim >= config.SEM_SIM_SUPPORT_THRESH
                and top_class_is_entailment):
            # Interpolate temperature: sim 0.90→T_LOW (gentle), sim 0.98→T_HIGH (aggressive)
            t_range  = getattr(config, 'NLI_TEMP_SCALE_LOW', 0.85) - getattr(config, 'NLI_TEMP_SCALE_HIGH', 0.60)
            sim_frac = min(1.0, (claim_sim - config.SEM_SIM_SUPPORT_THRESH) / 0.08)
            temperature = getattr(config, 'NLI_TEMP_SCALE_LOW', 0.85) - t_range * sim_frac
            logger.debug(
                f"Layer 3 — temp-scaling T={temperature:.2f} "
                f"(claim_sim={claim_sim:.3f}): {claim[:60]}"
            )

        # Apply temperature-scaled softmax to each (evidence, claim) pair
        scaled_probs = []
        for logit_row in scores:
            logit_row = logit_row / temperature
            logit_row = logit_row - logit_row.max()          # numerical stability
            exp_row   = np.exp(logit_row)
            prob_row  = exp_row / exp_row.sum()
            scaled_probs.append(prob_row)

        # index 0=contradiction, 1=entailment, 2=neutral
        max_entailment    = max(float(s[1]) for s in scaled_probs)
        max_contradiction = max(float(s[0]) for s in scaled_probs)

        # Best evidence = the doc that most strongly entails this claim
        best_idx      = max(range(len(scaled_probs)), key=lambda i: float(scaled_probs[i][1]))
        best_evidence = evidence_cits[best_idx]["answer"] if best_idx < len(evidence_cits) else ""

        # Verdict thresholds
        if max_contradiction > 0.65:
            verdict = "CONTRADICTED"
        elif max_entailment > 0.45:
            verdict = "SUPPORTED"
        else:
            verdict = "UNSUPPORTED"

        # Net score penalised by contradiction (same logic as before, now per-claim)
        net = max_entailment * (1.0 - max_contradiction)
        all_scores.append(net)

        claim_results.append(ClaimResult(
            claim         = claim,
            best_evidence = best_evidence[:200],
            entailment    = round(max_entailment, 4),
            contradiction = round(max_contradiction, 4),
            retrieval_sim = round(claim_sim, 4),
            verdict       = verdict,
        ))

    mean_entailment = float(np.mean(all_scores)) if all_scores else 0.5
    logger.info(
        f"Layer 3 — mean_entailment={mean_entailment:.3f}  "
        f"({len(claims)} claims, {len(claim_results)} with evidence)"
    )

    # ── Whole-answer contradiction check (original question vs KB) ────────
    # If the user's *question* is itself a false medical claim (e.g. "ibuprofen
    # and warfarin have no interaction"), KB evidence will contradict it directly.
    claim_contradiction = 0.0
    if question and question.strip():
        q_pairs = [
            (cit.get("answer", ""), question)
            for cit in citations
            if cit.get("answer", "")
        ]
        if q_pairs:
            q_scores            = nli.predict(q_pairs, apply_softmax=True)
            contradiction_probs = [float(s[0]) for s in q_scores]
            claim_contradiction = max(contradiction_probs)
            logger.info(f"Layer 3 — Max question contradiction: {claim_contradiction:.3f}")

    # Final score: mean claim entailment modulated by question-level contradiction.
    # We apply only 30% of the contradiction signal (not 100%) because q_contradict
    # measures KB-vs-question tension, which is noisy for yes/no research questions:
    # the KB always "contradicts" the question framing even for correct answers.
    # A 0.3 weight preserves the signal for genuinely contradicted factual claims
    # while preventing correct answers with moderate NLI evidence from being
    # collapsed to near-zero.
    critic_score = mean_entailment * (1.0 - 0.3 * claim_contradiction)
    logger.info(
        f"Layer 3 — critic_score={critic_score:.3f}  "
        f"(mean_ent={mean_entailment:.3f}  q_contradict={claim_contradiction:.3f})"
    )
    return float(critic_score), claim_results


# ─────────────────────────────────────────────────────────────────────────── #
#  Score Aggregator                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def aggregate(
    consistency: float,
    retrieval: float,
    critic: float,
    temporal_risk: float = 0.0,
    entity_risk: float = 0.0,
) -> Tuple[float, str]:
    """
    Combine the three detection signals into a final risk score.

    All inputs are "goodness" scores [0,1] (higher = more trustworthy).
    We invert them to get risk contributions, then apply weights.

    **Confidence Gate** (new): When Layer 1 self-consistency is very high
    (>= CONFIDENCE_GATE_THRESH), the LLM is confident in its answer.
    In this case, clamp the NLI critic score to at least CONFIDENCE_GATE_NLI_FLOOR
    so that a noisy NLI "neutral" can't over-rule a confident, retrieval-backed LLM.

    Two hard overrides push the flag directly to HIGH regardless of
    the weighted score:
      1. Both retrieval AND NLI critic fail simultaneously (fabricated content)
      2. Any claim references a future calendar year (impossible fact)

    Returns:
        risk_score — [0, 1], higher = more risky
        risk_flag  — "LOW" | "CAUTION" | "HIGH"
    """
    w = config.WEIGHTS

    # ── Confidence Gate ──────────────────────────────────────────────────
    # If the LLM is highly self-consistent AND retrieval is decent,
    # floor the NLI score to prevent false positives from noisy NLI.
    gate_thresh = getattr(config, 'CONFIDENCE_GATE_THRESH', 999.0)
    gate_floor  = getattr(config, 'CONFIDENCE_GATE_NLI_FLOOR', 0.60)
    # Only fire when consistency was actually measured (not dummy 1.0 from no-LLM eval).
    # Genuine consistency is always < 1.0 due to sampling noise.
    consistency_is_real = consistency < 0.999
    if consistency_is_real and consistency >= gate_thresh and retrieval >= 0.80:
        if critic < gate_floor:
            logger.info(
                f"Aggregate — Confidence gate fired: consistency={consistency:.3f} "
                f"→ NLI floor {critic:.3f} → {gate_floor:.3f}"
            )
            critic = gate_floor

    # ── Safety Buffer (High-Conflict Detection) ──────────────────────────
    # When the LLM is confident but Retrieval+NLI strongly disagree,
    # flag as HIGH-RISK/INCONCLUSIVE.  This catches fluent but wrong answers.
    safety_buffer_triggered = False
    if (getattr(config, 'SAFETY_BUFFER_ENABLED', False)
            and consistency_is_real):
        evidence_support = (retrieval + critic) / 2.0
        conflict_gap = consistency - evidence_support
        conflict_thresh = getattr(config, 'SAFETY_BUFFER_CONFLICT', 0.40)
        min_gap = getattr(config, 'SAFETY_BUFFER_MIN_GAP', 0.30)

        if (conflict_gap >= conflict_thresh
                and consistency >= 0.80
                and evidence_support < (1.0 - min_gap)):
            safety_buffer_triggered = True
            logger.info(
                f"Aggregate — Safety buffer triggered: "
                f"consistency={consistency:.3f} vs evidence_support={evidence_support:.3f} "
                f"(gap={conflict_gap:.3f} >= {conflict_thresh})"
            )

    consistency_risk = 1.0 - consistency
    retrieval_risk   = 1.0 - retrieval
    critic_risk      = 1.0 - critic

    risk_score = (
        w["consistency"]       * consistency_risk
        + w["retrieval"]       * retrieval_risk
        + w["critic"]         * critic_risk
        + w.get("entity", 0.0) * entity_risk
    )
    risk_score = float(np.clip(risk_score, 0.0, 1.0))

    # Hard override 1: retrieval + NLI critic both failing → fabricated content
    hard_override = (
        retrieval_risk > config.RISK_HARD_RETRIEVAL_THRESHOLD
        and critic_risk > config.RISK_HARD_CRITIC_THRESHOLD
    )

    # Hard override 2: future-year claim detected (near-certain hallucination)
    temporal_override = temporal_risk >= 1.0

    if hard_override or temporal_override:
        flag       = "HIGH"
        risk_score = max(risk_score, config.RISK_HIGH + 0.01)  # ensure numeric score reflects HIGH
        if temporal_override:
            logger.info("Aggregate — Temporal override triggered (future-year claim detected)")
        if hard_override:
            logger.info("Aggregate — Hard override triggered (retrieval+critic both failed)")
    elif safety_buffer_triggered:
        flag       = "HIGH"
        risk_score = max(risk_score, config.RISK_HIGH + 0.01)
        logger.info("Aggregate — Safety buffer override: LLM-evidence conflict → HIGH")
    elif risk_score < config.RISK_LOW:
        flag = "LOW"
    elif risk_score < config.RISK_HIGH:
        flag = "CAUTION"
    else:
        flag = "HIGH"

    return risk_score, flag, safety_buffer_triggered


def _build_explanation(
    flag: str,
    risk: float,
    layers: LayerResult,
    temporal_flags: List[Dict],
    safety_buffer: bool = False,
) -> str:
    base = {
        "LOW"    : "✅ Answer appears well-supported. Low hallucination risk.",
        "CAUTION": "⚠️  Some uncertainty detected. Review before clinical use.",
        "HIGH"   : "🚨 High hallucination risk. Do NOT use without expert verification.",
    }[flag]

    if safety_buffer:
        base = "🚨 HIGH CONFLICT: LLM is confident but evidence does not support the answer. Treat as inconclusive."

    concerns = {
        "self-consistency"  : layers.consistency_risk,
        "evidence retrieval": layers.retrieval_risk,
        "NLI entailment"    : layers.critic_risk,
        "entity overlap"    : layers.entity_risk,
    }
    dominant = max(concerns, key=concerns.get)
    explanation = f"{base}  |  Primary signal: {dominant} (risk={concerns[dominant]:.2f})"

    if temporal_flags:
        years = ", ".join(str(f["year"]) for f in temporal_flags)
        explanation += f"  |  Temporal flag: future year(s) detected [{years}]"

    return explanation


# ─────────────────────────────────────────────────────────────────────────── #
#  Public API                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def predict(question: str) -> PredictionResult:
    """
    Full end-to-end pipeline.
    Call this from the FastAPI endpoint or directly in scripts.
    """
    t0 = time.perf_counter()

    # Layer 1 — Self-Consistency
    answer, samples, consistency = layer1_consistency(question)

    # Layer 2 — Whole-answer retrieval (for global citations + display)
    retrieval_score, citations = layer2_retrieval(answer)

    # Layer 2b — Entity overlap check (catches wrong-entity hallucinations)
    entity_risk = layer2b_entity_check(answer, citations)

    # Layer 3 — Per-claim NLI with per-claim KB retrieval (FACTSCORE-style)
    critic_entailment, claim_results = layer3_critic(
        answer, citations, question=question
    )

    # Temporal detection — scan all decomposed claims for future-year references
    claims               = decompose_claims(answer)
    temporal_risk, temporal_flags = detect_temporal_claims(claims)

    # Final aggregation (includes temporal + entity hard-override + safety buffer)
    risk_score, risk_flag, safety_buffer_fired = aggregate(
        consistency, retrieval_score, critic_entailment, temporal_risk, entity_risk
    )

    layers = LayerResult(
        consistency_score = consistency,
        consistency_risk  = 1.0 - consistency,
        retrieval_score   = retrieval_score,
        retrieval_risk    = 1.0 - retrieval_score,
        critic_entailment = critic_entailment,
        critic_risk       = 1.0 - critic_entailment,
        temporal_risk     = temporal_risk,
        entity_risk       = entity_risk,
        claim_breakdown   = claim_results,
        samples           = samples,
        citations         = citations,
    )

    latency = (time.perf_counter() - t0) * 1000

    # Verdicts summary for breakdown dict
    verdict_counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0}
    for cr in claim_results:
        verdict_counts[cr.verdict] = verdict_counts.get(cr.verdict, 0) + 1

    return PredictionResult(
        question    = question,
        answer      = answer,
        risk_score  = round(risk_score, 4),
        risk_flag   = risk_flag,
        confidence  = round(1.0 - risk_score, 4),
        explanation = _build_explanation(risk_flag, risk_score, layers, temporal_flags, safety_buffer=safety_buffer_fired),
        breakdown   = {
            "consistency_score" : round(consistency, 4),
            "retrieval_score"   : round(retrieval_score, 4),
            "nli_entailment"    : round(critic_entailment, 4),
            "entity_risk"       : round(entity_risk, 4),
            "consistency_risk"  : round(layers.consistency_risk, 4),
            "retrieval_risk"    : round(layers.retrieval_risk, 4),
            "critic_risk"       : round(layers.critic_risk, 4),
            "temporal_risk"     : round(temporal_risk, 4),
            "n_claims"          : len(claim_results),
            "verdict_counts"    : verdict_counts,
        },
        citations       = citations,
        claim_breakdown = [asdict(cr) for cr in claim_results],
        temporal_flags  = temporal_flags,
        latency_ms      = round(latency, 1),
    )


# ─────────────────────────────────────────────────────────────────────────── #
#  Quick CLI test                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the first-line treatment for Type 2 diabetes?"
    print(f"\n  Question: {q}\n")

    result = predict(q)

    print(f"  Answer     : {result.answer[:200]}...")
    print(f"  Risk Flag  : {result.risk_flag}  (score={result.risk_score:.3f})")
    print(f"  Explanation: {result.explanation}")
    print(f"  Latency    : {result.latency_ms:.0f}ms")
    print(f"\n  Breakdown  : {json.dumps(result.breakdown, indent=2)}")

    if result.temporal_flags:
        print(f"\n  Temporal Flags ({len(result.temporal_flags)}):")
        for tf in result.temporal_flags:
            print(f"    Year {tf['year']}: {tf['reason']}")

    if result.claim_breakdown:
        print(f"\n  Per-Claim Breakdown ({len(result.claim_breakdown)} claims):")
        for i, cr in enumerate(result.claim_breakdown, 1):
            print(f"  [{i}] [{cr['verdict']:12}] ent={cr['entailment']:.2f}  contr={cr['contradiction']:.2f}  sim={cr['retrieval_sim']:.2f}")
            print(f"       Claim    : {cr['claim'][:100]}")
            print(f"       Evidence : {cr['best_evidence'][:100]}")

    if result.citations:
        print(f"\n  Top citation: {result.citations[0]['answer'][:150]}...")