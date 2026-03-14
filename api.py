"""
api.py — MedCAFAS FastAPI Server

Run:
    uvicorn api:app --reload --port 8000

Docs:
    http://localhost:8000/docs   (Swagger UI)
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pipeline import predict, PredictionResult, OllamaError

logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "MedCAFAS API",
    description = "Medical hallucination detection — 3-layer CPU-based engine",
    version     = "0.1.0-mvp",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global timeout for the pipeline (seconds) ────────────────────────────────
PREDICT_TIMEOUT_SECONDS = 600  # 10 minutes — CPU inference can be slow


# ── Global exception handler — catches ANY unhandled error ────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Return a clean JSON 500 instead of leaking the Python traceback.
    The full traceback is logged server-side for debugging.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {type(exc).__name__}",
            "message": str(exc)[:200],  # truncate to avoid info leak
        },
    )


# ── Request / Response ────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        example="What is the first-line treatment for Type 2 diabetes?",
    )


class PredictResponse(BaseModel):
    question       : str
    answer         : str
    risk_flag      : str
    risk_score     : float
    confidence     : float
    explanation    : str
    breakdown      : dict
    citations      : list
    claim_breakdown: list   # Per-claim FACTSCORE-style results
    temporal_flags : list   # Future-year claims flagged
    latency_ms     : float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Quick health check.  Verifies that critical models can be loaded
    (FAISS index + embedder) without actually running inference.
    """
    errors = []
    try:
        from pipeline import _get_kb
        index, meta = _get_kb()
        if index.ntotal != len(meta):
            errors.append(
                f"KB index/meta mismatch: {index.ntotal} vectors vs {len(meta)} metadata entries"
            )
    except Exception as e:
        errors.append(f"KB load failed: {e}")

    try:
        from pipeline import _get_embedder
        _get_embedder()
    except Exception as e:
        errors.append(f"Embedder load failed: {e}")

    try:
        from pipeline import _get_nli_model
        _get_nli_model()
    except Exception as e:
        errors.append(f"NLI model load failed: {e}")

    if errors:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "errors": errors, "model": "MedCAFAS-MVP"},
        )
    return {"status": "ok", "model": "MedCAFAS-MVP"}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: QuestionRequest):
    """
    Analyse a medical question for hallucination risk.

    - **question**: The medical question to evaluate

    Returns the LLM answer, a risk score, risk flag (LOW/CAUTION/HIGH),
    and a detailed breakdown of all three detection layers.

    Applies a server-side timeout to prevent hung Ollama calls from
    blocking the API indefinitely.
    """
    try:
        # Run the synchronous pipeline in a thread pool with a hard timeout
        loop = asyncio.get_event_loop()
        result: PredictionResult = await asyncio.wait_for(
            loop.run_in_executor(None, predict, req.question),
            timeout=PREDICT_TIMEOUT_SECONDS,
        )

        return PredictResponse(
            question       = result.question,
            answer         = result.answer,
            risk_flag      = result.risk_flag,
            risk_score     = result.risk_score,
            confidence     = result.confidence,
            explanation    = result.explanation,
            breakdown      = result.breakdown,
            citations      = result.citations,
            claim_breakdown= result.claim_breakdown,
            temporal_flags = result.temporal_flags,
            latency_ms     = result.latency_ms,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Pipeline timed out after {PREDICT_TIMEOUT_SECONDS}s. "
                   f"Ollama may be unresponsive. Check: ollama list",
        )
    except OllamaError as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM unavailable: {e}",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))