"""
config.py — Central configuration for MedCAFAS MVP
All tuneable parameters in one place.
"""

# ── LLM (Ollama) ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "phi3.5:latest"          # ~2.2GB, fast on CPU. Alternatives: llama3.2:1b, mistral
OLLAMA_TIMEOUT   = 600                      # Seconds to wait for a single Ollama call
OLLAMA_MAX_QUESTION_CHARS = 3000            # Truncate input to ~750 tokens before sending to phi3.5

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "michiyasunaga/BioLinkBERT-base"  # 768-dim, biomedical domain pre-trained

# ── Knowledge Base / FAISS ────────────────────────────────────────────────────
KB_INDEX_PATH    = "data/kb.index"
KB_META_PATH     = "data/kb_meta.json"
KB_MAX_DOCS      = 50000             # Expanded KB for USMLE-level verification

# ── BM25 Hybrid Retrieval ─────────────────────────────────────────────────────
BM25_INDEX_PATH  = "data/kb_bm25.pkl"  # Serialised BM25Okapi + raw passages
BM25_WEIGHT      = 0.40  # 40% BM25 lexical + 60% cosine semantic in hybrid score
BM25_CANDIDATES  = 20    # FAISS fetches this many candidates; BM25 reranks to TOP_K

# ── Entity Overlap Check (Layer 2b) ──────────────────────────────────────────
ENTITY_CHECK     = True  # Verify answer entities appear in retrieved evidence
ENTITY_MIN_TERMS = 2     # Skip check if fewer than this many key terms found
KB_SOURCES = {
    "medqa_usmle"        : 10000,   # GBaker/MedQA-USMLE-4-options  (full train split)
    "pubmedqa"           : 1000,    # qiaojin/PubMedQA pqa_labeled  (clinical trial abstracts)
    "medmcqa"            : 20000,   # medmcqa                       (medical entrance MCQs)
    "pubmedqa_artificial": 20000,   # qiaojin/PubMedQA pqa_artificial (synthetic trials)
}

# ── Text Chunking (KB build-time) ────────────────────────────────────────
KB_CHUNK_MAX_WORDS  = 300   # Maximum words per chunk
KB_CHUNK_OVERLAP    = 50    # Overlap words between consecutive chunks

# ── Self-Consistency ──────────────────────────────────────────────────────────
NUM_SAMPLES      = 3                 # Re-sample LLM 3 times (balance speed vs. signal)
SAMPLE_TEMP      = 0.8               # Sampling temperature for diversity
CONSISTENCY_RISK_THRESHOLD = 0.65   # Below this → inconsistent → risky

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K            = 5                 # Retrieve top-5 docs (raised from 3 to find more substantive evidence)
MIN_SIM          = 0.40              # Below this cosine similarity → retrieval failed
MMR_LAMBDA       = 0.85             # Max-Marginal Relevance: 1.0 = pure relevance, 0.0 = pure diversity
MMR_CANDIDATES   = 20               # How many FAISS candidates to consider before MMR re-ranking

# ── NLI Critic ────────────────────────────────────────────────────────────────
NLI_MODEL        = "cross-encoder/nli-deberta-v3-base"    # ~180MB, MNLI-acc=90.04% (was: small ~84%)
NLI_BATCH_SIZE   = 8
NLI_MIN_EVIDENCE_WORDS = 10         # Skip evidence < this for NLI (MedQA answers are often 1-5 words → useless as premises)

# ── Semantic Similarity Safety Net (Temperature-Scaled) ──────────────────────
# When retrieval cosine similarity >= this threshold but NLI gives "neutral",
# apply softmax temperature scaling on NLI logits to let entailment rise.
# Handles paraphrased LLM answers that are factually correct but phrased
# differently from KB evidence.
SEM_SIM_SUPPORT_THRESH = 0.92   # cosine sim above which temp-scaling activates (raised for BioLinkBERT)
NLI_TEMP_SCALE_HIGH    = 0.60   # temperature when sim >= 0.98 (aggressive squash)
NLI_TEMP_SCALE_LOW     = 0.85   # temperature when sim ~= 0.90 (gentle squash)

# Minimum claim length for NLI scoring.  Cross-encoder NLI models produce
# near-zero entailment for short noun phrases (e.g. "Cholesterol embolization")
# which are valid medical answers but not parseable as hypotheses.  Claims
# shorter than this threshold fall back to retrieval-similarity scoring.
NLI_MIN_CLAIM_WORDS = 8

# ── NLI Verdict Thresholds ────────────────────────────────────────────────────
# Per-claim verdict classification in layer3_critic.
# Applied to the best (evidence, claim) pair's softmax probabilities.
NLI_VERDICT_CONTRADICTION_THRESH = 0.65   # p(contradiction) above this → CONTRADICTED
NLI_VERDICT_ENTAILMENT_THRESH    = 0.45   # p(entailment) above this → SUPPORTED

# ── Question-Contradiction Modulation ────────────────────────────────────────
# When KB evidence contradicts the *question framing* (common for yes/no
# research questions), we dampen the signal to avoid penalising correct
# answers.  The final critic score is:
#   critic = mean_entailment × (1 - Q_CONTRADICTION_WEIGHT × max_q_contradiction)
Q_CONTRADICTION_WEIGHT = 0.30

# ── Short-Claim Fallback Proxy Range ─────────────────────────────────────────
# When a claim is too short for NLI and no question context is available,
# the retrieval similarity is clamped to this range as a proxy score.
SHORT_CLAIM_PROXY_MIN = 0.35
SHORT_CLAIM_PROXY_MAX = 0.50

# ── Q-E Score Remapping (Short Claims with Question Context) ─────────────────
# Maps cosine-sim from [QE_REMAP_SIM_LOW, QE_REMAP_SIM_HIGH] →
#                       [QE_REMAP_SCORE_LOW, QE_REMAP_SCORE_HIGH]
QE_REMAP_SIM_LOW    = 0.50
QE_REMAP_SIM_HIGH   = 0.90
QE_REMAP_SCORE_LOW  = 0.30
QE_REMAP_SCORE_HIGH = 0.80

# ── Safety Buffer Override Delta ─────────────────────────────────────────────
# When safety buffer or hard override fires, risk_score is forced to at least
# RISK_HIGH + this delta to ensure the numeric score reflects the HIGH flag.
RISK_HIGH_OVERRIDE_DELTA = 0.01

# ── Claim Decomposition (FACTSCORE-style per-claim verification) ──────────
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

# ── Confidence Gate (Dual-Key) ────────────────────────────────────────────────
# The gate now requires BOTH conditions ("dual-key") before it will trust the
# LLM and floor the NLI score.  This prevents the gate from silencing the NLI
# critic whenever phi3.5 is merely verbose-and-consistent (the "Yes-Man" bias).
#   Key 1: LLM self-consistency >= CONFIDENCE_GATE_THRESH
#   Key 2: Top-1 retrieval similarity >= CONFIDENCE_GATE_RETRIEVAL_THRESH
# If EITHER key fails, the NLI critic score is used as-is.
CONFIDENCE_GATE_THRESH              = 0.96   # raised from 0.92 – stricter LLM bar
CONFIDENCE_GATE_RETRIEVAL_THRESH    = 0.95   # new: retrieval must also strongly agree
CONFIDENCE_GATE_NLI_FLOOR           = 0.60   # when gate fires, set NLI score to at least this

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
