# MedCAFAS
**Medical Confidence-Aware Factuality Assessment System**

> *A training-free, fully CPU-deployable hallucination detection engine for medical LLMs, combining self-consistency sampling, multi-source retrieval-augmented verification, and FACTSCORE-style per-claim NLI analysis.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15.2-black)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

Large language models (LLMs) deployed in medical question-answering systems routinely produce confident but factually incorrect answers — a phenomenon known as hallucination.
MedCAFAS addresses this gap through a **three-layer detection pipeline** that requires no fine-tuning, no GPU, and no proprietary APIs. The system operates entirely on commodity CPU hardware and provides:

- **Claim-level explanations** showing exactly which sentences in an LLM answer are unsupported or contradicted by medical evidence
- **Temporal hallucination detection** that instantly identifies claims referencing future events (e.g. fabricated clinical trials)
- **Calibrated risk scores** with bootstrap 95% confidence intervals suitable for use in academic publications

No existing CPU-only open-source system performs end-to-end per-claim verification against a multi-source medical knowledge base. MedCAFAS fills this gap.

---

## Architecture

```
Input: Medical question / claim
       |
  Layer 1 — Self-Consistency  (Ollama / phi3.5)
  Sample LLM x 3 at temps {0.0, 0.8, 0.8}
  -> Mean pairwise semantic similarity -> consistency_score
       |
  Layer 2 — Retrieval Verification  (FAISS + all-MiniLM-L6-v2)
  Embed answer -> search 5 000-doc multi-source KB
  -> Top-k cosine similarity -> retrieval_score, citations
       |
  Layer 3 — NLI Critic  (cross-encoder/nli-deberta-v3-small)
  Decompose answer into atomic claims
  For EACH claim: per-claim KB retrieval + NLI entailment/contradiction
  -> verdict per claim: SUPPORTED / UNSUPPORTED / CONTRADICTED
  + whole-answer question contradiction check
       |
  Temporal Detection
  Regex scan claims for future calendar years -> hard override
       |
  Aggregation
  Weighted combination + 2 hard overrides
  -> risk_score in [0,1]  risk_flag: LOW / CAUTION / HIGH

Output: risk_flag, risk_score, explanation, per-claim breakdown,
        temporal_flags, citations
```

### Component Summary

| Component | Model / Tool | Size | Purpose |
|---|---|---|---|
| LLM sampler | phi3.5 (Ollama) | ~2.2 GB | Self-consistency sampling |
| Embedder | all-MiniLM-L6-v2 | 80 MB | Sentence embeddings |
| Vector DB | FAISS IndexFlatIP | — | O(n) exact nearest-neighbour |
| NLI critic | cross-encoder/nli-deberta-v3-small | 85 MB | Per-claim entailment / contradiction |
| KB sources | MedQA-USMLE + PubMedQA + MedMCQA | — | 5 000 medical evidence passages |
| Backend | FastAPI + uvicorn | — | REST API, port 8000 |
| Frontend | Next.js 15 + TypeScript + Tailwind | — | Interactive UI, port 3000 |

---

## Research Contributions

### 1. FACTSCORE-style Per-Claim Verification for Medical Text
Unlike prior systems that score an entire LLM answer holistically, MedCAFAS **decomposes each answer into atomic claims** and independently retrieves and scores evidence for each one. This is directly inspired by [FActScoring (Min et al., 2023)](https://arxiv.org/abs/2305.14251) but adapted for open-domain medical QA with a CPU-only stack.

Each claim receives:
- A dedicated per-claim KB retrieval query (not shared with other claims)
- An NLI entailment score from the best-matching KB passage
- An NLI contradiction score
- A verdict: **SUPPORTED** / **UNSUPPORTED** / **CONTRADICTED**

### 2. Temporal Claim Detection
Claims referencing future calendar years (e.g. *"HORIZON-9 trial published in 2027"*) are logically impossible and therefore near-certain hallucinations. A lightweight regex pass over decomposed claims provides a **zero-cost hard override** to HIGH risk with near-100% recall on this class.

### 3. Multi-Source Medical Knowledge Base
The KB aggregates three complementary open datasets:

| Dataset | Docs | Domain |
|---|---|---|
| GBaker/MedQA-USMLE-4-options | 2 000 | USMLE clinical vignettes |
| qiaojin/PubMedQA (pqa_labeled) | 1 000 | Clinical trial abstract Q&A |
| medmcqa | 2 000 | Medical entrance MCQ + explanations |
| **Total** | **5 000** | Multi-domain clinical coverage |

### 4. Calibrated Risk Scoring
MedCAFAS outputs **Expected Calibration Error (ECE)** alongside accuracy metrics. A well-calibrated detector (ECE < 0.05) is a prerequisite for deployment in clinical decision support. Bootstrap 95% confidence intervals are provided for all scalar metrics.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai/) with `phi3.5:latest` pulled

```bash
ollama pull phi3.5
```

### Step 1 — Python environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

### Step 2 — Build the knowledge base (run once)

```bash
python build_kb.py
# Downloads MedQA-USMLE + PubMedQA + MedMCQA (~4-6 min on first run)
# Outputs: data/kb.index  data/kb_meta.json
```

### Step 3 — Start the API

```bash
uvicorn api:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs
```

### Step 4 — Start the frontend

```bash
cd frontend
npm install
npm run dev
# UI: http://localhost:3000
```

---

## API Reference

### `POST /predict`

Request:
```json
{ "question": "What is the first-line treatment for Type 2 diabetes?" }
```

Response (abbreviated):
```json
{
  "risk_flag": "LOW",
  "risk_score": 0.24,
  "breakdown": {
    "verdict_counts": { "SUPPORTED": 2, "UNSUPPORTED": 1, "CONTRADICTED": 0 },
    "temporal_risk": 0.0,
    "n_claims": 3
  },
  "claim_breakdown": [
    {
      "claim": "Metformin is the first-line pharmacological treatment...",
      "entailment": 0.84, "contradiction": 0.03,
      "verdict": "SUPPORTED"
    }
  ],
  "temporal_flags": []
}
```

---

## Evaluation

### Running the eval suite

```bash
# Fast (no LLM, ~2 min for 100 samples)
python eval.py --no-llm --tune --ci --samples 100 --save results.json

# Full pipeline (~2 hours for 40 samples)
python eval.py --samples 40 --ci --save results_full.json
```

### Flag reference

| Flag | Description |
|---|---|
| `--no-llm` | Skip Layer 1 (Ollama). Tests retrieval + NLI only. |
| `--tune` | Grid-search optimal RISK_LOW / RISK_HIGH thresholds |
| `--ci` | Bootstrap 95% confidence intervals (1 000 resamples) |
| `--samples N` | Number of eval samples (0 = full test set ~1 273) |
| `--save FILE` | Write per-sample results to JSON |

### Metrics

- **Accuracy** — binary classification correct rate
- **Precision / Recall / F1** — hallucinated class
- **ROC-AUC** — threshold-free ranking quality
- **ECE** — Expected Calibration Error (< 0.05 = well-calibrated)
- **Bootstrap 95% CIs** — for accuracy, F1, and ROC-AUC

### Evaluation dataset
Test samples from `GBaker/MedQA-USMLE-4-options` test split.
Hallucinated answers use rule-based perturbation of correct answers:
- Negation flips (`first-line` → `last-resort`)
- Drug substitutions (warfarin ↔ heparin, metformin ↔ glipizide, …)
- Direction inversions (`increases` → `decreases`)

---

## Configuration

All tuneable parameters in [`config.py`](config.py):

```python
KB_MAX_DOCS      = 5000
KB_SOURCES = {"medqa_usmle": 2000, "pubmedqa": 1000, "medmcqa": 2000}
CLAIM_MIN_WORDS  = 4          # min words to treat a fragment as a claim
MAX_CLAIMS       = 10         # cap per answer (limits NLI cost)
TEMPORAL_DETECTION = True
RISK_LOW         = 0.30       # below -> LOW
RISK_HIGH        = 0.40       # above -> HIGH
WEIGHTS = {"consistency": 0.30, "retrieval": 0.35, "critic": 0.35}
```

---

## Limitations

| Limitation | Impact | Status |
|---|---|---|
| Short answers (< 4 words) | NLI cannot score single-word answers | Use with full-sentence answers |
| KB coverage gaps | Rare subspecialty topics may be absent | Roadmap: expand to PubMedQA full |
| Subtle numerical errors | "500 mg" vs "50 mg" | Roadmap: numerical claim extraction |
| Layer 1 requires Ollama | Self-consistency needs local LLM | `--no-llm` flag available |
| CPU latency | ~60–120 s per full query | Quantised models; faster embedder |

---

## Roadmap

- [ ] Numerical claim extraction and verification
- [ ] BioMedBERT NLI ensemble (clinical domain adaptation)
- [ ] PubMedQA full expansion (~211k artificial QA docs)
- [ ] Streaming API (stream per-claim results as they complete)
- [ ] Span-level UI highlighting (mark exact supported/contradicted phrases)
- [ ] Deployment guide (FastAPI → Render, Next.js → Vercel)

---

## Citation

```bibtex
@software{medcafas2026,
  author  = {Singh, Suraj Pratap},
  title   = {{MedCAFAS}: Medical Confidence-Aware Factuality Assessment System},
  year    = {2026},
  url     = {https://github.com/surajpsingh020/medcafas},
  note    = {Training-free CPU-only per-claim hallucination detection for medical LLMs}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
