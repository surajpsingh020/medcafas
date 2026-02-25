"""
api.py — MedCAFAS FastAPI Server

Run:
    uvicorn api:app --reload --port 8000

Docs:
    http://localhost:8000/docs   (Swagger UI)
"""

from __future__ import annotations
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import predict, PredictionResult

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


# ── Request / Response ────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        example="What is the first-line treatment for Type 2 diabetes?",
    )


class PredictResponse(BaseModel):
    question    : str
    answer      : str
    risk_flag   : str
    risk_score  : float
    confidence  : float
    explanation : str
    breakdown   : dict
    citations   : list
    claim_breakdown: list   # Per-claim FACTSCORE-style results
    temporal_flags : list   # Future-year claims flagged
    latency_ms  : float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick health check."""
    return {"status": "ok", "model": "MedCAFAS-MVP"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: QuestionRequest):
    """
    Analyse a medical question for hallucination risk.

    - **question**: The medical question to evaluate
    
    Returns the LLM answer, a risk score, risk flag (LOW/CAUTION/HIGH),
    and a detailed breakdown of all three detection layers.
    """
    try:
        result: PredictionResult = predict(req.question)
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
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")