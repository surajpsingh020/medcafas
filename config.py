"""
config.py — Central configuration for MedCAFAS MVP
All tuneable parameters in one place.
"""

# ── LLM (Ollama) ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "phi3.5:latest"          # ~2.2GB, fast on CPU. Alternatives: llama3.2:1b, mistral

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "michiyasunaga/BioLinkBERT-base"  # 768-dim, biomedical domain pre-trained

# ── Knowledge Base / FAISS ────────────────────────────────────────────────────
KB_INDEX_PATH    = "data/kb.index"
KB_META_PATH     = "data/kb_meta.json"
KB_MAX_DOCS      = 8000              # Multi-source expanded KB (4 sources)

# ── BM25 Hybrid Retrieval ─────────────────────────────────────────────────────
BM25_INDEX_PATH  = "data/kb_bm25.pkl"  # Serialised BM25Okapi + raw passages
BM25_WEIGHT      = 0.40  # 40% BM25 lexical + 60% cosine semantic in hybrid score
BM25_CANDIDATES  = 20    # FAISS fetches this many candidates; BM25 reranks to TOP_K

# ── Entity Overlap Check (Layer 2b) ──────────────────────────────────────────
ENTITY_CHECK     = True  # Verify answer entities appear in retrieved evidence
ENTITY_MIN_TERMS = 2     # Skip check if fewer than this many key terms found
KB_SOURCES = {
    "medqa_usmle"        : 2000,   # GBaker/MedQA-USMLE-4-options  (clinical vignettes)
    "pubmedqa"           : 1000,   # qiaojin/PubMedQA pqa_labeled  (clinical trial abstracts)
    "medmcqa"           : 2000,   # medmcqa                        (medical entrance MCQs)
    "pubmedqa_artificial": 3000,   # qiaojin/PubMedQA pqa_artificial (211k synthetic trials)
}

# ── Self-Consistency ──────────────────────────────────────────────────────────
NUM_SAMPLES      = 3                 # Re-sample LLM 3 times (balance speed vs. signal)
SAMPLE_TEMP      = 0.8               # Sampling temperature for diversity
CONSISTENCY_RISK_THRESHOLD = 0.65   # Below this → inconsistent → risky

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K            = 3                 # Retrieve top-3 docs
MIN_SIM          = 0.40              # Below this cosine similarity → retrieval failed
MMR_LAMBDA       = 0.85             # Max-Marginal Relevance: 1.0 = pure relevance, 0.0 = pure diversity
MMR_CANDIDATES   = 20               # How many FAISS candidates to consider before MMR re-ranking

# ── NLI Critic ────────────────────────────────────────────────────────────────
NLI_MODEL        = "cross-encoder/nli-deberta-v3-base"    # ~180MB, MNLI-acc=90.04% (was: small ~84%)
NLI_BATCH_SIZE   = 8

# ── Semantic Similarity Safety Net (Temperature-Scaled) ──────────────────────
# When retrieval cosine similarity >= this threshold but NLI gives "neutral",
# apply softmax temperature scaling on NLI logits to let entailment rise.
# Handles paraphrased LLM answers that are factually correct but phrased
# differently from KB evidence.
SEM_SIM_SUPPORT_THRESH = 0.90   # cosine sim above which temp-scaling activates
NLI_TEMP_SCALE_HIGH    = 0.60   # temperature when sim >= 0.98 (aggressive squash)
NLI_TEMP_SCALE_LOW     = 0.85   # temperature when sim ~= 0.90 (gentle squash)

# Minimum claim length for NLI scoring.  Cross-encoder NLI models produce
# near-zero entailment for short noun phrases (e.g. "Cholesterol embolization")
# which are valid medical answers but not parseable as hypotheses.  Claims
# shorter than this threshold fall back to retrieval-similarity scoring.
NLI_MIN_CLAIM_WORDS = 8# ── Claim Decomposition (FACTSCORE-style per-claim verification) ──────────
CLAIM_MIN_WORDS  = 4        # Minimum words for a fragment to be treated as a claim
MAX_CLAIMS       = 10       # Cap claims per answer (prevents excessive NLI calls)

# ── Temporal Detection ────────────────────────────────────────────────────
TEMPORAL_DETECTION = True   # Flag claims that reference future calendar years
# ── Score Aggregation ─────────────────────────────────────────────────────────
WEIGHTS = {
    "consistency" : 0.25,   # How much answers vary across samples
    "retrieval"   : 0.30,   # How well claims are backed by evidence (BM25 + cosine)
    "critic"      : 0.30,   # NLI entailment score (claim vs. evidence)
    "entity"      : 0.15,   # Fraction of answer entities absent from retrieved evidence
}

# ── Confidence Gate ───────────────────────────────────────────────────────────
# If Layer 1 consistency is very high (LLM is confident), bypass NLI critic
# to prevent the noisy NLI from over-ruling a confident LLM.
CONFIDENCE_GATE_THRESH = 0.92   # consistency above this → trust LLM, use retrieval-only
CONFIDENCE_GATE_NLI_FLOOR = 0.60  # when gate fires, set NLI score to at least this

# ── Safety Buffer (High-Conflict Detection) ──────────────────────────────────
# When Layer 1 (LLM) and Layer 2+3 (Retrieval/NLI) strongly disagree, the
# answer is flagged as HIGH-RISK/INCONCLUSIVE regardless of the weighted score.
# This catches scenarios where a confident LLM produces fluent but wrong answers
# that the retrieval/NLI layers detect as unsupported.
SAFETY_BUFFER_ENABLED   = True
SAFETY_BUFFER_CONFLICT  = 0.40   # |consistency - avg(retrieval, NLI)| above this → conflict
SAFETY_BUFFER_MIN_GAP   = 0.30   # minimum gap between LLM confidence and evidence support

RISK_LOW     = 0.20          # Below → 🟢 LOW  (tuned on PubMedQA 100-sample raw scores)
RISK_HIGH    = 0.30          # Above → 🔴 HIGH  (between → 🟡 CAUTION)

# Hard-override: if BOTH retrieval and NLI critic fail this badly, force HIGH
# regardless of weighted score (catches fabricated / future-dateed facts)
RISK_HARD_RETRIEVAL_THRESHOLD = 0.50   # retrieval_risk > this
RISK_HARD_CRITIC_THRESHOLD    = 0.85   # critic_risk    > this
