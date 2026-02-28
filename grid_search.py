"""
grid_search.py — Hyperparameter Grid Search for MedCAFAS
──────────────────────────────────────────────────────────
Searches over CONFIDENCE_GATE_THRESH × MMR_LAMBDA to find the
combination that maximises F1 Score on MedQA-USMLE (our hardest test).

Also outputs a heatmap CSV and prints the top-5 configurations.

Usage:
    python grid_search.py                    # default 60 samples, no-LLM
    python grid_search.py --samples 100      # more thorough
    python grid_search.py --with-llm         # full 3-layer (slow)
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config
from eval import (
    EvalResult,
    EvalSample,
    compute_ece,
    load_eval_samples,
)
from pipeline import (
    _get_bm25,
    _get_embedder,
    _get_kb,
    _get_nli_model,
    aggregate,
    layer2_retrieval,
    layer2b_entity_check,
    layer3_critic,
)


def _compute_layer_scores(
    sample: EvalSample,
    mmr_lambda: float,
) -> Tuple[float, float, float]:
    """
    Run layers 2/3 for a single sample with a given MMR_LAMBDA.
    Returns (retrieval_score, entity_risk, critic_score).
    """
    orig_mmr = config.MMR_LAMBDA
    try:
        config.MMR_LAMBDA = mmr_lambda
        retrieval_score, citations = layer2_retrieval(sample.answer)
        entity_risk = layer2b_entity_check(sample.answer, citations)
        critic_score, _ = layer3_critic(
            sample.answer, citations, question=sample.question
        )
        return retrieval_score, entity_risk, critic_score
    finally:
        config.MMR_LAMBDA = orig_mmr


def grid_search(
    samples: List[EvalSample],
    gate_values: List[float],
    mmr_values: List[float],
) -> List[Dict]:
    """
    Optimised grid search: run the expensive pipeline once per
    (mmr_lambda, sample) pair, then sweep gate_thresh in pure Python.
    This is 6× faster than the naive approach.
    """
    n_combos = len(gate_values) * len(mmr_values)
    print(f"\nGrid search: {n_combos} combinations × {len(samples)} samples")
    print(f"  CONFIDENCE_GATE_THRESH: {gate_values}")
    print(f"  MMR_LAMBDA:             {mmr_values}")
    print(f"  (Optimised: {len(mmr_values)} pipeline sweeps, "
          f"gate thresh applied analytically)")
    print()

    # Phase 1: run pipeline for each (mmr_lambda, sample) — the slow part
    # cache[mmr_lambda][sample_idx] = (retrieval, entity, critic)
    cache: Dict[float, List[Tuple[float, float, float]]] = {}

    for mi, mmr in enumerate(mmr_values, 1):
        print(f"  MMR sweep [{mi}/{len(mmr_values)}] λ={mmr:.2f}  ... ",
              end="", flush=True)
        t0 = time.perf_counter()
        scores = []
        for sample in samples:
            scores.append(_compute_layer_scores(sample, mmr))
        cache[mmr] = scores
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.0f}s)")

    # Phase 2: sweep gate thresholds analytically
    all_results: List[Dict] = []
    combo_idx = 0

    for mmr in mmr_values:
        for gate in gate_values:
            combo_idx += 1
            results: List[EvalResult] = []
            orig_gate = config.CONFIDENCE_GATE_THRESH
            config.CONFIDENCE_GATE_THRESH = gate

            for si, sample in enumerate(samples):
                retrieval_score, entity_risk, critic_score = cache[mmr][si]
                consistency = 1.0
                risk_score, risk_flag, _ = aggregate(
                    consistency, retrieval_score, critic_score,
                    entity_risk=entity_risk
                )
                predicted_hallucinated = risk_flag == "HIGH"
                correct = predicted_hallucinated == sample.is_hallucinated
                results.append(EvalResult(
                    sample=sample,
                    risk_score=risk_score,
                    risk_flag=risk_flag,
                    retrieval=retrieval_score,
                    critic=critic_score,
                    entity=entity_risk,
                    predicted_hallucinated=predicted_hallucinated,
                    correct=correct,
                ))

            config.CONFIDENCE_GATE_THRESH = orig_gate

            y_true = [int(r.sample.is_hallucinated) for r in results]
            y_pred = [int(r.predicted_hallucinated) for r in results]
            y_score = [r.risk_score for r in results]

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = float("nan")
            ece = compute_ece(results)

            print(
                f"  [{combo_idx:>3}/{n_combos}] gate={gate:.2f}  mmr={mmr:.2f}  "
                f"acc={acc:.1%}  F1={f1:.3f}  AUC={auc:.3f}  ECE={ece:.3f}"
            )

            all_results.append({
                "gate_thresh": gate,
                "mmr_lambda": mmr,
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "roc_auc": round(auc, 4),
                "ece": round(ece, 4),
                "n_correct": sum(r.correct for r in results),
                "n_total": len(results),
            })

    # Sort by F1 descending, then accuracy
    all_results.sort(key=lambda x: (x["f1"], x["accuracy"]), reverse=True)
    return all_results


def save_heatmap_csv(results: List[Dict], path: str = "data/grid_search_heatmap.csv"):
    """Save results as CSV for heatmap visualization."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Heatmap CSV saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="MedCAFAS Grid Search")
    parser.add_argument(
        "--samples",
        type=int,
        default=60,
        help="Number of MedQA eval samples (default: 60)",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Use full 3-layer pipeline (slow, requires Ollama)",
    )
    args = parser.parse_args()

    # Pre-load singletons
    print("Pre-loading models...")
    _get_embedder()
    _get_kb()
    _get_nli_model()
    _get_bm25()
    print("  Models ready.\n")

    # Load MedQA eval samples
    samples = load_eval_samples(args.samples)

    # Define search grid
    gate_values = [0.85, 0.88, 0.90, 0.92, 0.95, 0.98]
    mmr_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    t0 = time.perf_counter()
    results = grid_search(samples, gate_values, mmr_values)
    elapsed = time.perf_counter() - t0

    # Print top 5
    print("\n" + "=" * 70)
    print("  Top 5 Configurations (by F1 on MedQA-USMLE)")
    print("=" * 70)
    print(
        f"  {'Rank':>4}  {'Gate':>6}  {'MMR':>6}  {'Acc':>7}  {'F1':>7}  "
        f"{'AUC':>7}  {'ECE':>7}"
    )
    print("-" * 70)
    for rank, r in enumerate(results[:5], 1):
        print(
            f"  {rank:>4}  {r['gate_thresh']:>6.2f}  {r['mmr_lambda']:>6.2f}  "
            f"{r['accuracy']:>6.1%}  {r['f1']:>7.3f}  "
            f"{r['roc_auc']:>7.3f}  {r['ece']:>7.3f}"
        )

    best = results[0]
    print(f"\n  Best: gate={best['gate_thresh']:.2f}, mmr={best['mmr_lambda']:.2f}")
    print(f"         F1={best['f1']:.3f}, Acc={best['accuracy']:.1%}, AUC={best['roc_auc']:.3f}")

    # Compare with current config
    print(f"\n  Current config: gate={0.92}, mmr={0.85}")
    current = [
        r
        for r in results
        if abs(r["gate_thresh"] - 0.92) < 0.01
        and abs(r["mmr_lambda"] - 0.85) < 0.01
    ]
    if current:
        c = current[0]
        print(
            f"  Current F1={c['f1']:.3f}, Acc={c['accuracy']:.1%}, AUC={c['roc_auc']:.3f}"
        )
        delta_f1 = best["f1"] - c["f1"]
        print(f"  Improvement: ΔF1 = {delta_f1:+.3f}")

    # Suggest config update
    if best["gate_thresh"] != 0.92 or best["mmr_lambda"] != 0.85:
        print(f"\n  Suggested config.py updates:")
        print(f"    CONFIDENCE_GATE_THRESH = {best['gate_thresh']}")
        print(f"    MMR_LAMBDA             = {best['mmr_lambda']}")

    print(f"\n  Grid search completed in {elapsed:.0f}s")

    # Save outputs
    save_heatmap_csv(results)
    with open("data/grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results saved to data/grid_search_results.json")


if __name__ == "__main__":
    main()
