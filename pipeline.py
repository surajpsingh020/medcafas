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
import os
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


class OllamaError(RuntimeError):
    """Raised when the local Ollama LLM is unreachable or returns empty output."""
    pass


# ─────────────────────────────────────────────────────────────────────────── #
#  Text-length safety for cross-encoder (DeBERTa 512-token limit)            #
# ─────────────────────────────────────────────────────────────────────────── #

# DeBERTa tokenises roughly 1 token per 4 characters.  With a 512-token
# budget shared across premise + hypothesis + 3 special tokens, we cap
# each side conservatively.  This is a character-level heuristic that avoids
# importing the DeBERTa tokeniser for a fast path.
_NLI_MAX_PREMISE_CHARS  = 900   # ~225 tokens for evidence (premise)
_NLI_MAX_HYPOTHESIS_CHARS = 500 # ~125 tokens for claim (hypothesis)
# Total ≈ 350 tokens → comfortably under 512 even with long sub-words.


def _truncate_for_nli(premise: str, hypothesis: str) -> Tuple[str, str]:
    """
    Truncate an NLI (premise, hypothesis) pair to fit within the
    cross-encoder's 512-token context window.

    Truncation is on character boundaries at a word break to avoid
    splitting mid-word.  The hypothesis (claim) gets priority since
    it's the assertion being evaluated.
    """
    if len(hypothesis) > _NLI_MAX_HYPOTHESIS_CHARS:
        cut = hypothesis[:_NLI_MAX_HYPOTHESIS_CHARS].rfind(" ")
        if cut > 0:
            hypothesis = hypothesis[:cut]
        else:
            hypothesis = hypothesis[:_NLI_MAX_HYPOTHESIS_CHARS]

    if len(premise) > _NLI_MAX_PREMISE_CHARS:
        cut = premise[:_NLI_MAX_PREMISE_CHARS].rfind(" ")
        if cut > 0:
            premise = premise[:cut]
        else:
            premise = premise[:_NLI_MAX_PREMISE_CHARS]

    return premise, hypothesis


# ─────────────────────────────────────────────────────────────────────────── #
#  Singletons (loaded once at first use)                                     #
# ─────────────────────────────────────────────────────────────────────────── #

_embedder: Optional[SentenceTransformer] = None
_nli_model: Optional[CrossEncoder]       = None
_faiss_index: Optional[faiss.Index]      = None
_kb_meta: Optional[List[Dict]]           = None
_bm25_data:   Optional[Dict]             = None
_claim_cache: Dict[Tuple[str, str], str] = {}

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
    """
    Load the FAISS index and metadata JSON.

    Performs an integrity check on first load:
      - Both files must exist
      - index.ntotal must equal len(meta)
      - Corrupted FAISS files raise RuntimeError (not a raw C++ crash)

    Raises:
        RuntimeError — if KB is missing, corrupted, or index/meta counts diverge.
    """
    global _faiss_index, _kb_meta
    if _faiss_index is None:
        # ── Load FAISS index ──────────────────────────────────────────────
        if not os.path.exists(config.KB_INDEX_PATH):
            raise RuntimeError(
                f"Knowledge base not found at {config.KB_INDEX_PATH}. "
                "Please run:  python build_kb.py"
            )
        try:
            _faiss_index = faiss.read_index(config.KB_INDEX_PATH)
        except Exception as e:
            raise RuntimeError(
                f"FAISS index is corrupted or unreadable ({config.KB_INDEX_PATH}): {e}. "
                "Delete data/kb.index and re-run:  python build_kb.py"
            ) from e

        # ── Load metadata ─────────────────────────────────────────────────
        if not os.path.exists(config.KB_META_PATH):
            _faiss_index = None  # reset so next call retries
            raise RuntimeError(
                f"KB metadata not found at {config.KB_META_PATH}. "
                "Please run:  python build_kb.py"
            )
        try:
            with open(config.KB_META_PATH) as f:
                _kb_meta = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            _faiss_index = None
            raise RuntimeError(
                f"KB metadata is corrupted ({config.KB_META_PATH}): {e}. "
                "Delete data/kb_meta.json and re-run:  python build_kb.py"
            ) from e

        # ── Integrity check: vector count must match metadata count ───────
        if _faiss_index.ntotal != len(_kb_meta):
            n_index = _faiss_index.ntotal
            n_meta  = len(_kb_meta)
            # Reset singletons so the corrupted state isn't cached
            _faiss_index = None
            _kb_meta = None
            raise RuntimeError(
                f"KB integrity check failed: FAISS index has {n_index} vectors "
                f"but kb_meta.json has {n_meta} entries. These must match. "
                "Delete data/kb.index and data/kb_meta.json, then re-run:  python build_kb.py"
            )

        logger.info(
            f"KB loaded: {_faiss_index.ntotal} docs "
            f"(integrity check passed: index == meta == {_faiss_index.ntotal})"
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
    retrieval_claim: str        # Contextualized claim used for retrieval
    best_evidence: str          # Best supporting/contradicting KB snippet
    entailment:    float        # [0, 1]  how well KB supports this claim
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
#  Negation markers (shared by Layer 1 consistency & Layer 3 Q-E scoring)    #
# ─────────────────────────────────────────────────────────────────────────── #

# Compiled regex with word boundaries to prevent false matches like
# "except" → "exceptional" or "unlikely" → "unlikelyhood".
# Multi-word phrases use explicit spacing; single words use \b anchors.
_NEGATION_RE = re.compile(
    r"""(?ix)                       # case-insensitive, verbose
    \b(?:
        not\s+correct              # "not correct"
      | not\s+the\b               # "not the ..."
      | incorrect                  # "incorrect"
      | is\s+not\b                # "is not"
      | are\s+not\b               # "are not"
      | was\s+not\b               # "was not"
      | were\s+not\b              # "were not"
      | cannot\b                   # "cannot"
      | should\s+not\b            # "should not"
      | would\s+not\b             # "would not"
      | does\s+not\b              # "does not"
      | do\s+not\b                # "do not"
      | it\s+is\s+not\b           # "it is not"
      | not\s+a\b                 # "not a ..."
      | not\s+an\b                # "not an ..."
      | no\s+evidence\b           # "no evidence"
      | except\b                   # "except" but NOT "exceptional"
      | contraindicated\b          # "contraindicated"
      | avoid\b                    # "avoid" but NOT "avoidance"
      | unlikely\b                 # "unlikely" but NOT "unlikelyhood"
      | ruled\s+out\b             # "ruled out"
    )
    """
)


def _has_negation(text: str) -> bool:
    """Check whether text contains any medical negation marker (word-boundary safe)."""
    return bool(_NEGATION_RE.search(text))


# ─────────────────────────────────────────────────────────────────────────── #
#  Layer 1 — Self-Consistency                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def _ask_ollama(question: str, temperature: float = 0.0) -> str:
    """
    Single synchronous call to local Ollama API.
    Returns the model's answer text.
    Raises OllamaError on timeout, connection failure, or empty response.
    """
    system_msg = (
        "You are a knowledgeable medical assistant. "
        "Provide accurate, concise answers based on established clinical guidelines. "
        "If you are not sure, say so explicitly."
    )

    # ── Input truncation: cap question to ~3000 chars (~750 tokens) to stay
    #    well within phi3.5's 4096-token context after system prompt + framing.
    max_question_chars = getattr(config, 'OLLAMA_MAX_QUESTION_CHARS', 3000)
    if len(question) > max_question_chars:
        logger.warning(
            f"Ollama — question truncated from {len(question)} to {max_question_chars} chars"
        )
        question = question[:max_question_chars]

    payload = {
        "model"  : config.OLLAMA_MODEL,
        "prompt" : f"{system_msg}\n\nQuestion: {question}\n\nAnswer:",
        "stream" : False,
        "options": {"temperature": temperature, "num_predict": 300},
    }
    timeout = getattr(config, 'OLLAMA_TIMEOUT', 300)
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        if not answer:
            raise OllamaError(
                "Ollama returned an empty response. Model may be loading or OOM."
            )
        return answer
    except requests.exceptions.ReadTimeout:
        raise OllamaError(
            f"Ollama read timeout ({timeout}s). Is the model loaded? "
            f"Try: ollama run {config.OLLAMA_MODEL}"
        )
    except requests.exceptions.ConnectionError:
        raise OllamaError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. "
            "Is it running?  Run: ollama serve"
        )
    except requests.exceptions.HTTPError as e:
        raise OllamaError(
            f"Ollama HTTP error: {e}. "
            f"Model '{config.OLLAMA_MODEL}' may not be loaded. "
            f"Try: ollama pull {config.OLLAMA_MODEL}"
        )


def contextualize_claim(question: str, claim: str) -> str:
    """
    Rewrite a claim into a standalone sentence using question context.

    This is used immediately before per-claim embedding/FAISS retrieval so
    pronouns and omitted subjects are resolved (e.g., "It" -> "Appendicitis").
    Returns the original claim on any Ollama failure.
    """
    q = (question or "").strip()
    c = (claim or "").strip()
    if not q or not c:
        return c

    system_msg = (
        "You rewrite medical claims for retrieval. "
        "Output exactly one rewritten sentence only. "
        "No preface, no explanation, no quotes, no markdown. "
        "Preserve medical meaning and uncertainty wording."
    )

    prompt = (
        "Rewrite the CLAIM so it is fully self-contained using QUESTION context.\n"
        "Rules:\n"
        "1) Resolve pronouns/coreference to explicit medical entities from QUESTION.\n"
        "2) Keep the claim's meaning unchanged.\n"
        "3) Keep it as one sentence.\n"
        "4) Output only the rewritten sentence.\n\n"
        "Example 1\n"
        "QUESTION: What are the symptoms of appendicitis?\n"
        "CLAIM: It often causes nausea and vomiting.\n"
        "REWRITE: Appendicitis often causes nausea and vomiting.\n\n"
        "Example 2\n"
        "QUESTION: How is bacterial meningitis diagnosed?\n"
        "CLAIM: It is confirmed with CSF analysis.\n"
        "REWRITE: Bacterial meningitis is confirmed with CSF analysis.\n\n"
        "Now rewrite this input:\n"
        f"QUESTION: {q}\n"
        f"CLAIM: {c}\n"
        "REWRITE:"
    )

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": f"{system_msg}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 80,
        },
    }

    timeout = min(getattr(config, "OLLAMA_TIMEOUT", 300), 30)

    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        rewritten = (resp.json().get("response", "") or "").strip()
        if not rewritten:
            return c

        # Keep only the first sentence if model returns multiple lines/sentences.
        one_line = " ".join(rewritten.split())
        first_sentence = re.split(r"(?<=[.!?])\s+", one_line)[0].strip()
        return first_sentence or c
    except Exception as e:
        logger.warning(f"Claim contextualization failed, using original claim: {e}")
        return c


def layer1_consistency(question: str) -> Tuple[str, List[str], float]:
    """
    Sample the LLM NUM_SAMPLES times and measure semantic consistency.

    Returns:
        primary_answer  — The deterministic (temp=0) answer used downstream
        samples         — All sampled answers
        consistency     — [0, 1] semantic similarity across samples

    Raises:
        OllamaError — if the primary Ollama call fails (timeout, connection, empty)
    """
    # Primary answer at temperature 0 (deterministic)
    # This MUST succeed — let OllamaError propagate to caller
    primary = _ask_ollama(question, temperature=0.0)

    # Diverse samples to test consistency
    # Sample failures are non-fatal: fall back to primary as duplicate
    samples = [primary]
    for _ in range(config.NUM_SAMPLES - 1):
        try:
            samples.append(_ask_ollama(question, temperature=config.SAMPLE_TEMP))
        except OllamaError as e:
            logger.warning(f"Layer 1 — sample call failed ({e}), using primary as fallback")
            samples.append(primary)

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


def _bm25_primary_retrieval(query: str, top_k: int = 5) -> Tuple[float, List[Dict]]:
    """
    BM25-primary retrieval for short queries (1-5 words) where embedding
    search fails because a single term like 'Cisplatin' maps to a
    generic region of the embedding space.

    Strategy:
      1. BM25 scores ALL 50k docs by lexical match → top N candidates.
      2. Embed query + candidate passages → cosine similarity re-rank.
      3. Return top-k by hybrid score.

    This inverts the default FAISS-primary approach and is critical for
    achieving answer-aware NLI: 'Cisplatin' finds cisplatin docs,
    'Methotrexate' finds methotrexate docs.
    """
    index, meta = _get_kb()
    bm25_data = _get_bm25()
    embedder  = _get_embedder()

    if not bm25_data or "bm25" not in bm25_data:
        # Fall back to FAISS if BM25 unavailable
        query_emb = embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        distances, indices_arr = index.search(query_emb, top_k)
        citations = []
        sims = []
        for dist, idx in zip(distances[0], indices_arr[0]):
            if idx < 0 or idx >= len(meta):
                continue
            doc = meta[idx]
            sims.append(float(dist))
            citations.append({
                "source": doc.get("source", "KB"),
                "question": doc.get("question", ""),
                "answer": doc.get("answer", ""),
                "similarity": round(float(dist), 4),
            })
        return (float(max(sims)) if sims else 0.0, citations)

    # Step 1: BM25 over all docs
    tokens = query.lower().split()
    raw_scores = bm25_data["bm25"].get_scores(tokens)
    # Get top 3× candidates for re-ranking
    n_cand = min(top_k * 4, 20)
    top_indices = np.argsort(raw_scores)[::-1][:n_cand]

    # Skip docs with zero BM25 score (no lexical match)
    top_indices = [int(i) for i in top_indices if raw_scores[i] > 0]
    if not top_indices:
        # No lexical matches — fall back to FAISS
        query_emb = embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        distances, indices_arr = index.search(query_emb, top_k)
        citations = []
        sims = []
        for dist, idx in zip(distances[0], indices_arr[0]):
            if idx < 0 or idx >= len(meta):
                continue
            doc = meta[idx]
            sims.append(float(dist))
            citations.append({
                "source": doc.get("source", "KB"),
                "question": doc.get("question", ""),
                "answer": doc.get("answer", ""),
                "similarity": round(float(dist), 4),
            })
        return (float(max(sims)) if sims else 0.0, citations)

    # Step 2: Compute cosine similarity for BM25 candidates
    query_emb = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    # Get embedding vectors for BM25 candidates from FAISS index
    cand_embs = np.array([index.reconstruct(i) for i in top_indices], dtype=np.float32)
    cos_sims = cosine_similarity(query_emb, cand_embs)[0]

    # Step 3: Hybrid BM25 + cosine
    bm25_max = float(raw_scores[top_indices].max()) if len(top_indices) > 0 else 1.0
    if bm25_max == 0:
        bm25_max = 1.0

    w = config.BM25_WEIGHT
    candidates = []
    for i, idx in enumerate(top_indices):
        bm25_norm = float(raw_scores[idx]) / bm25_max
        cosine = float(cos_sims[i])
        hybrid = (1.0 - w) * cosine + w * bm25_norm
        candidates.append((hybrid, idx))

    candidates.sort(reverse=True)
    candidates = candidates[:top_k]

    citations = []
    sims = []
    for hybrid, idx in candidates:
        doc = meta[idx]
        sims.append(hybrid)
        citations.append({
            "source": doc.get("source", "KB"),
            "question": doc.get("question", ""),
            "answer": doc.get("answer", ""),
            "similarity": round(hybrid, 4),
        })

    retrieval_score = float(max(sims)) if sims else 0.0
    return retrieval_score, citations


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


def layer2_retrieval(answer: str, question: str = "") -> Tuple[float, List[Dict]]:
    """
    Embed the LLM answer and retrieve top-k documents from the KB using
    hybrid BM25 + cosine similarity scoring.

    **Answer-Aware Retrieval**: When a question is provided, the retrieval
    query is `question + answer` rather than just `answer`.  This ensures
    that correct and hallucinated answers to the *same* question retrieve
    *different* KB evidence — the correct answer pulls confirming docs while
    the wrong answer pulls unrelated or contradicting docs.

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

    # Answer-aware query: combine question context with answer for retrieval
    retrieval_query = f"{question} {answer}".strip() if question else answer

    answer_emb = embedder.encode(
        [retrieval_query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    n_candidates = max(config.TOP_K, getattr(config, 'MMR_CANDIDATES', config.BM25_CANDIDATES))
    distances, indices = index.search(answer_emb, n_candidates)

    candidates = _hybrid_scores(retrieval_query, distances, indices, meta)

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
            "answer"    : doc.get("answer", ""),
            "similarity": round(hybrid, 4),
        })

    retrieval_score = float(max(sims)) if sims else 0.0
    logger.info(f"Layer 2 — Hybrid top score: {retrieval_score:.3f}")
    return retrieval_score, citations


def layer2_retrieval_single(query: str, context: str = "") -> Tuple[float, List[Dict]]:
    """
    Retrieve top-k KB documents for a single query string using hybrid scoring
    with MMR diversity re-ranking.
    Used for per-claim retrieval in the claim decomposition loop.

    For SHORT queries (< 5 words) without context: uses BM25-primary retrieval
    to find docs by lexical match (e.g., "Cisplatin" → cisplatin docs).
    For longer queries or when context is provided: uses FAISS-primary with
    hybrid scoring as before.
    """
    # Short queries: BM25-primary (lexical match for drug/term names)
    if len(query.split()) < 5 and not context:
        return _bm25_primary_retrieval(query, top_k=config.TOP_K)

    index, meta = _get_kb()
    embedder    = _get_embedder()

    # Answer-aware: combine question context with claim for better evidence
    retrieval_query = f"{context} {query}".strip() if context else query

    query_emb = embedder.encode(
        [retrieval_query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    n_candidates = max(config.TOP_K, getattr(config, 'MMR_CANDIDATES', config.BM25_CANDIDATES))
    distances, indices = index.search(query_emb, n_candidates)

    candidates = _hybrid_scores(retrieval_query, distances, indices, meta)

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
            "answer"    : doc.get("answer", ""),
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

    global _claim_cache
    claim_results: List[ClaimResult] = []
    all_scores:    List[float]       = []

    for claim in claims:
        # ── 1. Contextualize the claim using Llama 3.1 ──
        retrieval_claim = claim
        if question.strip():
            cache_key = (question, claim)
            if cache_key not in _claim_cache:
                _claim_cache[cache_key] = contextualize_claim(question, claim)
            retrieval_claim = _claim_cache[cache_key]

        # ── 2. The Amnesia Fix: Give the rewritten claim to DeBERTa ──
        nli_claim = retrieval_claim
        is_short_claim = len(nli_claim.split()) < config.NLI_MIN_CLAIM_WORDS
        if is_short_claim and question.strip():
            q_truncated = " ".join(question.split()[:80])
            nli_claim = f"{q_truncated} Answer: {nli_claim}"

        # ── 3. The FAISS Magnet Fix: Retrieve using ONLY the clean claim ──
        claim_sim, claim_cits = layer2_retrieval_single(retrieval_claim)

        # Fall back to answer-level citations if per-claim retrieval comes up empty
        evidence_cits = claim_cits if claim_cits else citations

        # ── 4. The "MCQ Blindspot" Fix ──
        # Merge the clinical vignette (question) with the factual option (answer)
        # so DeBERTa gets the full medical context, not just a single word like "T10".
        for cit in evidence_cits:
            cit["answer"] = f"Context: {cit.get('question', '')} Fact: {cit.get('answer', '')}".strip()

        if not evidence_cits:
            claim_results.append(ClaimResult(
                claim         = claim,
                retrieval_claim = retrieval_claim,
                best_evidence = "",
                entailment    = 0.5,
                retrieval_sim = 0.0,
                verdict       = "UNSUPPORTED",
            ))
            all_scores.append(0.5)
            continue

        # NLI: (KB evidence, claim) pairs — use nli_claim for evaluation
        # Filter out very short evidence (e.g. MedQA 1-5 word answers) that
        # give NLI no factual content to evaluate against.  Only evidence
        # with >= NLI_MIN_EVIDENCE_WORDS is useful as an NLI premise.
        min_ev_words = getattr(config, 'NLI_MIN_EVIDENCE_WORDS', 10)
        nli_evidence = [
            cit for cit in evidence_cits
            if cit.get("answer") and len(cit["answer"].split()) >= min_ev_words
        ]
        # Fall back to ALL evidence if no substantial docs available
        if not nli_evidence:
            nli_evidence = [cit for cit in evidence_cits if cit.get("answer")]

        pairs = [
            _truncate_for_nli(cit["answer"], nli_claim)
            for cit in nli_evidence
        ]
        if not pairs:
            all_scores.append(0.5)
            continue

        # ── Short-claim: Question-Evidence alignment scoring ──────────────
        # For short claims (noun phrases like "Cisplatin"), NLI produces
        # neutral scores regardless of correctness.  Instead, score by how
        # well the answer-specific KB evidence aligns with the question:
        #   - Correct answer → evidence about that topic → HIGH sim with Q
        #   - Wrong answer → evidence about different topic → LOW sim with Q
        # This provides discrimination even when NLI can't evaluate.
        if is_short_claim and question.strip():
            embedder = _get_embedder()
            q_emb = embedder.encode(
                [question], normalize_embeddings=True, convert_to_numpy=True
            )
            ev_texts = [cit["answer"] for cit in nli_evidence]
            ev_embs = embedder.encode(
                ev_texts, normalize_embeddings=True, convert_to_numpy=True
            )
            qe_sims = cosine_similarity(q_emb, ev_embs)[0]
            best_qe_idx = int(np.argmax(qe_sims))
            qe_score = float(qe_sims[best_qe_idx])
            best_ev = nli_evidence[best_qe_idx]["answer"]

            # Map Q-E similarity to 0-1 score:
            # sim ~0.85+ → strong alignment (0.7-1.0)
            # sim ~0.70  → moderate (0.5)
            # sim ~0.50  → weak (0.3)
            # Linear remap from [QE_REMAP_SIM_LOW, QE_REMAP_SIM_HIGH] →
            #                   [QE_REMAP_SCORE_LOW, QE_REMAP_SCORE_HIGH]
            sim_range   = config.QE_REMAP_SIM_HIGH - config.QE_REMAP_SIM_LOW
            score_range = config.QE_REMAP_SCORE_HIGH - config.QE_REMAP_SCORE_LOW
            net = max(0.1, min(0.9,
                config.QE_REMAP_SCORE_LOW + (qe_score - config.QE_REMAP_SIM_LOW) * (score_range / sim_range)
            ))

            # ── Negation penalty ──────────────────────────────────────────
            # Cosine similarity is negation-blind: "Drug X" and "NOT Drug X"
            # retrieve the same evidence and produce the same Q-E score.
            # When the answer/claim contains explicit negation markers AND
            # the Q-E alignment is moderate-to-high, the evidence actually
            # SUPPORTS the topic — but the answer NEGATES it → hallucination.
            #
            # Fix: invert the net score so that high Q-E alignment under
            # negation maps to LOW support (= high risk of hallucination).
            claim_lower = claim.lower()
            answer_lower = answer.lower()
            has_negation = _has_negation(claim_lower) or _has_negation(answer_lower)
            if has_negation:
                # Invert: high alignment + negation → contradiction signal
                net = max(0.1, min(0.9, 1.0 - net))
                logger.debug(
                    f"Layer 3 — negation detected in claim, "
                    f"inverted Q-E net: {1.0 - net:.3f} → {net:.3f}: "
                    f"{claim[:60]}"
                )

            verdict = "CONTRADICTED" if has_negation else (
                "SUPPORTED" if net > 0.55 else "UNSUPPORTED"
            )

            claim_results.append(ClaimResult(
                claim         = claim,
                retrieval_claim = retrieval_claim,
                best_evidence = best_ev,
                entailment    = round(qe_score, 4),
                retrieval_sim = round(claim_sim, 4),
                verdict       = verdict,
            ))
            all_scores.append(net)
            logger.debug(
                f"Layer 3 — short claim Q-E scoring: qe_sim={qe_score:.3f} "
                f"net={net:.3f} negated={has_negation}: {claim[:60]}"
            )
            continue

        # ── Short-claim fallback (only when no question context available) ─
        # If claim is short AND we couldn't extend it with a question,
        # fall back to retrieval-based proxy.
        if is_short_claim and not question.strip():
            proxy = min(config.SHORT_CLAIM_PROXY_MAX, max(config.SHORT_CLAIM_PROXY_MIN, claim_sim))
            claim_results.append(ClaimResult(
                claim         = claim,
                retrieval_claim = retrieval_claim,
                best_evidence = nli_evidence[0]["answer"] if nli_evidence else "",
                entailment    = round(proxy, 4),
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
            # Interpolate temperature: sim 0.92→T_LOW (gentle), sim 1.0→T_HIGH (aggressive)
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

        # ── Three-class per-pair scoring ────────────────────────────────
        # index 0=contradiction, 1=entailment, 2=neutral
        #
        # Old formula: net = max_ent * (1 - max_con) collapsed neutral to ~0
        # because max_ent is near-zero when evidence is merely irrelevant.
        # Fix: score each (evidence, claim) pair independently using a linear
        #   pair_net = 0.5 + 0.5*p_ent - 0.5*p_con
        # This gives: entailed→~1.0, neutral→~0.5, contradicted→~0.0
        # Then take the BEST pair (most favorable evidence wins).
        pair_nets = []
        for s in scaled_probs:
            p_ent = float(s[1])
            p_con = float(s[0])
            pair_nets.append(0.5 + 0.5 * p_ent - 0.5 * p_con)

        best_idx      = max(range(len(pair_nets)), key=lambda i: pair_nets[i])
        net           = pair_nets[best_idx]
        best_evidence = nli_evidence[best_idx]["answer"] if best_idx < len(nli_evidence) else ""

        max_entailment    = max(float(s[1]) for s in scaled_probs)
        max_contradiction = max(float(s[0]) for s in scaled_probs)

        # Verdict thresholds (based on best-pair probabilities)
        best_ent = float(scaled_probs[best_idx][1])
        best_con = float(scaled_probs[best_idx][0])
        if best_con > config.NLI_VERDICT_CONTRADICTION_THRESH:
            verdict = "CONTRADICTED"
        elif best_ent > config.NLI_VERDICT_ENTAILMENT_THRESH:
            verdict = "SUPPORTED"
        else:
            verdict = "UNSUPPORTED"

        all_scores.append(net)

        claim_results.append(ClaimResult(
            claim         = claim,
            retrieval_claim = retrieval_claim,
            best_evidence = best_evidence,
            entailment    = round(max_entailment, 4),
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
            _truncate_for_nli(cit.get("answer", ""), question)
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
    critic_score = mean_entailment * (1.0 - config.Q_CONTRADICTION_WEIGHT * claim_contradiction)
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
) -> Tuple[float, str, bool]:
    """
    Combine the three detection signals into a final risk score.

    All inputs are "goodness" scores [0,1] (higher = more trustworthy).
    We invert them to get risk contributions, then apply weights.

    **Conflict-First Aggregation**: Safety Buffer is evaluated BEFORE the
    Confidence Gate.  If the LLM and evidence layers disagree strongly,
    the answer is forced HIGH regardless of gate eligibility.

    **Confidence Gate (Dual-Key)**: Only fires when BOTH keys are met:
      Key 1: consistency >= CONFIDENCE_GATE_THRESH (0.96)
      Key 2: retrieval   >= CONFIDENCE_GATE_RETRIEVAL_THRESH (0.95)
    If either key fails, the NLI critic score is used as-is.
    If the Safety Buffer has already triggered, the gate is blocked.

    Two additional hard overrides push the flag directly to HIGH:
      1. Both retrieval AND NLI critic fail simultaneously (fabricated content)
      2. Any claim references a future calendar year (impossible fact)

    Returns:
        risk_score             — [0, 1], higher = more risky
        risk_flag              — "LOW" | "CAUTION" | "HIGH"
        safety_buffer_triggered — True if conflict-first override fired
    """
    w = config.WEIGHTS

    # ── Safety Buffer (High-Conflict Detection) — evaluated FIRST ────────
    # "Conflict-First" rule: if the LLM is confident but Retrieval+NLI
    # strongly disagree, the Safety Buffer overrides the Confidence Gate
    # and forces HIGH-RISK.  This catches fluent but wrong answers that
    # a "Yes-Man" LLM would otherwise whitewash via the gate.
    safety_buffer_triggered = False
    # Only fire when consistency was actually measured (not dummy 1.0 from no-LLM eval).
    # Genuine consistency is always < 1.0 due to sampling noise.
    consistency_is_real = consistency < 0.999
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
                f"Aggregate — Safety buffer triggered (conflict-first): "
                f"consistency={consistency:.3f} vs evidence_support={evidence_support:.3f} "
                f"(gap={conflict_gap:.3f} >= {conflict_thresh})"
            )

    # ── Confidence Gate (Dual-Key) ───────────────────────────────────────
    # Gate only fires when BOTH keys are satisfied:
    #   Key 1: LLM consistency >= CONFIDENCE_GATE_THRESH  (0.96)
    #   Key 2: Top-1 retrieval >= CONFIDENCE_GATE_RETRIEVAL_THRESH  (0.95)
    # If the Safety Buffer has already flagged a conflict, the gate is
    # NEVER allowed to override it ("conflict-first" principle).
    gate_thresh     = getattr(config, 'CONFIDENCE_GATE_THRESH', 999.0)
    gate_ret_thresh = getattr(config, 'CONFIDENCE_GATE_RETRIEVAL_THRESH', 0.95)
    gate_floor      = getattr(config, 'CONFIDENCE_GATE_NLI_FLOOR', 0.60)
    gate_fires = (
        consistency_is_real
        and not safety_buffer_triggered          # conflict-first: buffer wins
        and consistency >= gate_thresh            # Key 1: LLM must be highly consistent
        and retrieval   >= gate_ret_thresh        # Key 2: retrieval must strongly agree
    )
    if gate_fires and critic < gate_floor:
        logger.info(
            f"Aggregate — Confidence gate fired (dual-key): "
            f"consistency={consistency:.3f}, retrieval={retrieval:.3f} "
            f"→ NLI floor {critic:.3f} → {gate_floor:.3f}"
        )
        critic = gate_floor

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
        risk_score = max(risk_score, config.RISK_HIGH + config.RISK_HIGH_OVERRIDE_DELTA)
        if temporal_override:
            logger.info("Aggregate — Temporal override triggered (future-year claim detected)")
        if hard_override:
            logger.info("Aggregate — Hard override triggered (retrieval+critic both failed)")
    elif safety_buffer_triggered:
        flag       = "HIGH"
        risk_score = max(risk_score, config.RISK_HIGH + config.RISK_HIGH_OVERRIDE_DELTA)
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

def predict(question: str, answer: Optional[str] = None) -> PredictionResult:
    """
    Full end-to-end pipeline.
    Call this from the FastAPI endpoint or directly in scripts.

    If *answer* is provided (eval / detection mode), Ollama still generates
    its own answers for Layer 1, but consistency is measured between the
    **provided** answer and Ollama's answers.  Layers 2-3 evaluate the
    provided answer, not the LLM-generated one.
    """
    t0 = time.perf_counter()

    if answer is None:
        # Normal mode — generate + self-check
        answer, samples, consistency = layer1_consistency(question)
    else:
        # Detection mode — check a specific answer against Ollama
        _, ollama_samples, _ = layer1_consistency(question)
        embedder = _get_embedder()
        all_texts = [answer] + ollama_samples
        embs = embedder.encode(all_texts, normalize_embeddings=True)
        sims = cosine_similarity(embs[0:1], embs[1:])[0]
        consistency = float(np.mean(sims))
        samples = ollama_samples
        logger.info(f"Layer 1 — Consistency (provided vs Ollama): {consistency:.3f}")

        # Negation correction: cosine similarity is negation-blind —
        # "This is NOT correct: X" and Ollama's "X" will have high similarity
        # despite being semantic opposites.  Cap consistency to signal doubt.
        answer_lower = answer.lower()
        if _has_negation(answer_lower):
            consistency = min(consistency, 0.50)
            logger.info(f"Layer 1 — Negation penalty → consistency capped at {consistency:.3f}")

    # Layer 2 — Answer-aware retrieval (question+answer for targeted evidence)
    retrieval_score, citations = layer2_retrieval(answer, question=question)

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
            print(f"  [{i}] [{cr['verdict']:12}] ent={cr['entailment']:.2f}  sim={cr['retrieval_sim']:.2f}")
            print(f"       Claim    : {cr['claim'][:100]}")
            print(f"       Evidence : {cr['best_evidence'][:100]}")

    if result.citations:
        print(f"\n  Top citation: {result.citations[0]['answer'][:150]}...")