# 🏥 MedCAFAS

**Medical Confidence-Aware Factuality Assessment System**

> *A training-free, fully CPU-deployable hallucination detection engine for medical LLMs, combining self-consistency sampling, multi-source retrieval-augmented verification, and FACTSCORE-style per-claim NLI analysis.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15.2-black)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 The Problem & Solution

Large language models (LLMs) deployed in medical question-answering systems routinely produce confident but factually incorrect answers — a phenomenon known as **hallucination**. In clinical settings, a single fabricated drug dosage or invented contraindication can directly harm patients.

MedCAFAS addresses this critical safety gap through a **three-layer detection pipeline** that requires no fine-tuning, no GPU, and no proprietary APIs. It acts as an automated, strict fact-checker — breaking down LLM responses into individual clinical claims and mathematically scoring each one against a verified database of **50,000 medical documents** before showing them to a user.

Operating entirely on commodity CPU hardware, MedCAFAS provides:

- **Claim-level explanations** showing exactly which sentences are unsupported or contradicted
- **Temporal hallucination detection** that instantly catches fabricated future events (e.g. invented clinical trials)
- **Calibrated risk scores** with bootstrap 95% confidence intervals suitable for clinical deployment logic

No existing CPU-only open-source system performs end-to-end per-claim verification against a multi-source medical knowledge base. MedCAFAS fills this gap.

---

## 🧠 Architecture

```
Input: Medical question / claim
       │
 Layer 1 — Self-Consistency  (Ollama / phi3.5)
 Sample LLM × 3 at temps {0.0, 0.8, 0.8}
 → Mean pairwise semantic similarity → consistency_score
       │
 Layer 2 — Hybrid Retrieval Verification  (FAISS + BM25 + BioLinkBERT-base)
 Embed answer → BM25 lexical pre-filter (k=20) + FAISS cosine re-rank
 Hybrid score: 40% BM25 + 60% cosine similarity
 → Top-k hybrid-scored docs → retrieval_score, citations
       │
 Layer 2b — Entity Overlap Check  (pharma regex + curated acronym whitelist)
 Extract drug names / dosages / clinical acronyms from answer
 Check each term appears in pooled citation text
 → entity_risk: fraction of answer entities absent from evidence
       │
 Layer 3 — NLI Critic  (cross-encoder/nli-deberta-v3-base)
 Decompose answer into atomic claims (FACTSCORE-style)
 For EACH claim (≥ 8 words): per-claim KB retrieval + NLI entailment
 Short claims (< 8 words): Q-E similarity + negation penalty
 → verdict per claim: SUPPORTED / UNSUPPORTED / CONTRADICTED
 + whole-answer question contradiction check
       │
 Temporal Detection
 Regex scan claims for future calendar years → hard override
       │
 Aggregation  (consistency 25% + retrieval 30% + critic 30% + entity 15%)
 Weighted combination + 2 hard overrides
 → risk_score ∈ [0, 1]   risk_flag: LOW / CAUTION / HIGH

Output: risk_flag, risk_score, explanation, per-claim breakdown,
        temporal_flags, entity_risk, citations
```

### Component Summary

| Component | Model / Tool | Size | Purpose |
|---|---|---|---|
| LLM sampler | phi3.5 (Ollama) | ~2.2 GB | Self-consistency sampling |
| Embedder | michiyasunaga/BioLinkBERT-base | 440 MB | Biomedical sentence embeddings |
| Lexical retriever | BM25Okapi (rank-bm25) | — | 40% of hybrid retrieval score |
| Vector DB | FAISS IndexFlatIP (768-dim) | — | O(n) exact nearest neighbour |
| NLI critic | cross-encoder/nli-deberta-v3-base | 184 MB | Per-claim entailment / contradiction |
| KB sources | MedQA-USMLE + PubMedQA + MedMCQA | — | 50,000 medical evidence passages |
| Backend | FastAPI + uvicorn | — | Async REST API (600 s timeouts) |
| Frontend | Next.js 15 + TypeScript + Tailwind | — | Interactive UI |

---

## 🔬 Research Contributions & Engineering Highlights

### 1. FACTSCORE-style Per-Claim Verification

Unlike prior systems that score an entire LLM answer holistically, MedCAFAS **decomposes each answer into atomic claims** and independently retrieves and scores evidence for each one. Adapted for open-domain medical QA on a CPU-only stack, each claim receives a dedicated retrieval query and an NLI verdict (**SUPPORTED** / **UNSUPPORTED** / **CONTRADICTED**).

### 2. Negation-Aware Scoring Guardrails

Cosine similarity is traditionally negation-blind — *"Drug X"* and *"NOT Drug X"* retrieve identical evidence. MedCAFAS applies lexical negation detection using compiled word-boundary regex. When explicit negation markers are detected, the similarity score is mathematically inverted, preventing dangerous false-positive evaluations. **This single engineering fix raised ROC-AUC from 0.694 to 0.952.**

### 3. Multi-Source Medical Knowledge Base

The vector database aggregates three complementary open datasets for robust clinical coverage:

| Dataset | Docs | Domain |
|---|---|---|
| MedQA-USMLE-4-options | 10,000 | USMLE clinical vignettes |
| PubMedQA (pqa_labeled) | 19,000 | Clinical trial abstract Q&A |
| MedMCQA | 21,000 | Medical entrance MCQ + explanations |
| **Total** | **50,000** | **Multi-domain clinical coverage** |

### 4. Robust Fault Tolerance

Engineered to handle heavy CPU-bound tensor operations without crashing. Implements robust asynchronous circuit breakers (600-second timeouts) to prevent server deadlocks during deep NLI cross-encoder evaluations.

---

## 📊 Evaluation & Metrics

MedCAFAS is rigorously benchmarked using a rule-based perturbation strategy (negation / drug swaps) to test detection accuracy.

### Final Results (MedQA-USMLE, n=60, phi3.5, 50k-doc KB)

| Mode | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| No-LLM (Layer 2+3 only) | 63.3% | 0.450 | 0.898 |
| **Full pipeline (all layers)** | **88.3%** | **0.881** | **0.952** |

**NLI discrimination gap:** correct answers score 0.615 mean entailment vs 0.280 for hallucinated (0.335 gap).

### Confusion Matrix (Full Pipeline)

|  | Pred: NOT-HALL | Pred: HALL |
|---|---|---|
| True: NOT-HALL | 27 | 3 |
| True: HALL | 4 | **26** |

### Evaluation Datasets

| `--dataset` | Source | Label Strategy |
|---|---|---|
| `medqa` | GBaker/MedQA-USMLE test split | Rule-based perturbation (negation / drug swap) |
| `pubmedqa` | qiaojin/PubMedQA long_answer | Rule-based perturbation of clinical abstracts |
| `llm` | qiaojin/PubMedQA (final_decision) | Real phi3.5 outputs; labeled by yes/no decision-matching |

### Running the Eval Suite

```bash
# Fast: no-LLM mode (~2 min for 100 samples)
python eval.py --no-llm --dataset medqa --samples 60 --ci --save results.json

# Full pipeline with all 3 layers
python eval.py --dataset medqa --samples 60 --ci --save results_full.json

# Gold standard: real phi3.5 outputs with decision-matching labels
python eval.py --dataset llm --samples 40 --ci --save results_llm.json
```

| Flag | Description |
|---|---|
| `--no-llm` | Skip Layer 1 (Ollama). Tests retrieval + NLI only. |
| `--dataset` | `medqa` (default) / `pubmedqa` / `llm` |
| `--tune` | Grid-search optimal RISK_LOW / RISK_HIGH thresholds |
| `--ci` | Bootstrap 95% confidence intervals (1,000 resamples) |
| `--samples N` | Number of eval samples (0 = full test set) |
| `--save FILE` | Write per-sample results to JSON |

---

## ⚙️ Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai/) with `phi3.5:latest` pulled

```bash
ollama pull phi3.5
```

### Step 1 — Python Environment

```bash
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

### Step 2 — Build the Knowledge Base (Run Once)

```bash
python build_kb.py
# Downloads MedQA-USMLE + PubMedQA + MedMCQA (~4-6 min)
# Outputs: data/kb.index, data/kb_meta.json
```

### Step 3 — Start the API & Frontend

```bash
# Terminal 1: Backend
uvicorn api:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
# Access UI at: http://localhost:3000
```

---

## 🔌 API Reference

### `POST /predict`

**Request:**
```json
{ "question": "What is the first-line treatment for Type 2 diabetes?" }
```

**Response** (abbreviated):
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
      "entailment": 0.84,
      "verdict": "SUPPORTED"
    }
  ],
  "temporal_flags": []
}
```

---

## 🛠️ Configuration

All tuneable parameters live in [`config.py`](config.py):

```python
# Retrieval
KB_MAX_DOCS      = 50000
BM25_WEIGHT      = 0.40       # 40% BM25 lexical + 60% cosine in hybrid score

# NLI / claims
CLAIM_MIN_WORDS      = 4
MAX_CLAIMS           = 10
NLI_MIN_CLAIM_WORDS  = 8      # shorter claims fall back to retrieval-similarity

# Risk thresholds
RISK_LOW   = 0.20             # below → LOW
RISK_HIGH  = 0.30             # above → HIGH

# Layer weights
WEIGHTS = {"consistency": 0.25, "retrieval": 0.30, "critic": 0.30, "entity": 0.15}

# Timeouts
OLLAMA_TIMEOUT = 600           # 10 min per Ollama call (CPU-safe)
```

---

## 📁 Project Structure

```
medcafas/
├── pipeline.py          # Core 3-layer detection engine (1,500+ lines)
├── config.py            # All tuneable parameters
├── api.py               # FastAPI backend with async circuit breakers
├── build_kb.py          # One-time knowledge base builder
├── eval.py              # Evaluation suite (accuracy, F1, ROC-AUC, CI)
├── requirements.txt     # Pinned Python dependencies
├── data/
│   ├── kb.index         # FAISS vector index (50k docs)
│   └── kb_meta.json     # Document metadata
└── frontend/
    ├── app/             # Next.js 15 app router
    ├── components/      # React UI components
    └── lib/types.ts     # TypeScript interfaces
```

---

## ⚠️ Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Short answers (< 4 words) | NLI cannot score single-word answers | Use with full-sentence answers |
| KB coverage gaps | Rare subspecialty topics may be absent | Expand to PubMedQA full (~211k docs) |
| Subtle numerical errors | "500 mg" vs "50 mg" not caught | Roadmap: numerical claim extraction |
| Layer 1 requires Ollama | Self-consistency needs local LLM | `--no-llm` flag available |
| CPU latency | ~3–5 min per full pipeline query | Quantised models; faster embedder |

---

## 🗺️ Roadmap

- [ ] Numerical claim extraction and verification
- [ ] BioMedBERT NLI ensemble for specialized clinical domain adaptation
- [ ] Streaming API (stream per-claim results to the UI as they complete)
- [ ] Span-level UI highlighting (mark exact supported/contradicted phrases in the text)

---

## 📖 Citation

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

## 📄 License

MIT — see [LICENSE](LICENSE).

---

Developed by **Suraj Pratap Singh**.
