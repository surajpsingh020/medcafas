"""
ablation_study.py -- Comprehensive ablation study for MedCAFAS
==============================================================

Measures the contribution of each pipeline component by disabling them
one-at-a-time and re-evaluating.  Produces a LaTeX-ready table and a
grouped bar chart.

Configurations tested:
  1. Full pipeline          (all layers)
  2. Remove BM25            (FAISS cosine only)
  3. Remove entity check    (Layer 2b off)
  4. Remove NLI critic      (retrieval score only)
  5. NLI only               (no retrieval, no entity)
  6. Remove sem-sim boost   (original NLI without paraphrase handling)

Usage:
    python ablation_study.py                    # PubMedQA, 100 samples
    python ablation_study.py --dataset pubmedqa --samples 100
    python ablation_study.py --dataset all --samples 60
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from eval import (
    EvalResult,
    EvalSample,
    load_eval_samples,
    load_eval_samples_pubmedqa,
    eval_no_llm,
    compute_ece,
)
from pipeline import (
    layer2_retrieval,
    layer2b_entity_check,
    layer3_critic,
    aggregate,
    _get_embedder,
    _get_kb,
    _get_nli_model,
    _get_bm25,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# -------------------------------------------------------------------------- #
#  Ablation eval functions                                                    #
# -------------------------------------------------------------------------- #

def _eval_full(sample: EvalSample) -> EvalResult:
    """Full pipeline (all layers active)."""
    return eval_no_llm(sample)


def _eval_no_bm25(sample: EvalSample) -> EvalResult:
    """Disable BM25 — use only FAISS cosine similarity for retrieval."""
    original_weight = config.BM25_WEIGHT
    config.BM25_WEIGHT = 0.0
    try:
        result = eval_no_llm(sample)
    finally:
        config.BM25_WEIGHT = original_weight
    return result


def _eval_no_entity(sample: EvalSample) -> EvalResult:
    """Disable Layer 2b entity overlap check."""
    original_flag = config.ENTITY_CHECK
    config.ENTITY_CHECK = False
    try:
        result = eval_no_llm(sample)
    finally:
        config.ENTITY_CHECK = original_flag
    return result


def _eval_no_nli(sample: EvalSample) -> EvalResult:
    """Disable NLI critic — use retrieval + entity only."""
    retrieval_score, citations = layer2_retrieval(sample.answer)
    entity_risk = layer2b_entity_check(sample.answer, citations)
    # Replace critic with neutral 0.5
    critic_score = 0.5
    consistency = 1.0
    risk_score, risk_flag = aggregate(consistency, retrieval_score, critic_score,
                                      entity_risk=entity_risk)
    predicted_hallucinated = risk_flag == "HIGH"
    correct = predicted_hallucinated == sample.is_hallucinated
    return EvalResult(
        sample=sample, risk_score=risk_score, risk_flag=risk_flag,
        retrieval=retrieval_score, critic=critic_score, entity=entity_risk,
        predicted_hallucinated=predicted_hallucinated, correct=correct,
    )


def _eval_nli_only(sample: EvalSample) -> EvalResult:
    """NLI only — no retrieval scoring, no entity check."""
    retrieval_score, citations = layer2_retrieval(sample.answer)  # still need citations for NLI
    critic_score, _ = layer3_critic(sample.answer, citations, question=sample.question)
    # Zero out retrieval and entity contributions
    consistency = 1.0
    entity_risk = 0.0
    # Override weights temporarily
    orig_weights = config.WEIGHTS.copy()
    config.WEIGHTS = {"consistency": 0.0, "retrieval": 0.0, "critic": 1.0, "entity": 0.0}
    try:
        risk_score, risk_flag = aggregate(consistency, 0.5, critic_score, entity_risk=entity_risk)
    finally:
        config.WEIGHTS = orig_weights
    predicted_hallucinated = risk_flag == "HIGH"
    correct = predicted_hallucinated == sample.is_hallucinated
    return EvalResult(
        sample=sample, risk_score=risk_score, risk_flag=risk_flag,
        retrieval=retrieval_score, critic=critic_score, entity=entity_risk,
        predicted_hallucinated=predicted_hallucinated, correct=correct,
    )


def _eval_no_semsim(sample: EvalSample) -> EvalResult:
    """Disable the semantic similarity safety net (revert to pure NLI)."""
    original_thresh = config.SEM_SIM_SUPPORT_THRESH
    config.SEM_SIM_SUPPORT_THRESH = 999.0  # effectively disables boost
    try:
        result = eval_no_llm(sample)
    finally:
        config.SEM_SIM_SUPPORT_THRESH = original_thresh
    return result


# -------------------------------------------------------------------------- #
#  Scoring                                                                    #
# -------------------------------------------------------------------------- #

@dataclass
class AblationRow:
    name:     str
    accuracy: float
    f1:       float
    roc_auc:  float
    ece:      float
    n:        int


def _score(name: str, results: List[EvalResult]) -> AblationRow:
    y_true  = [int(r.sample.is_hallucinated) for r in results]
    y_pred  = [int(r.predicted_hallucinated)  for r in results]
    y_score = [r.risk_score                   for r in results]
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5
    ece = compute_ece(results)
    return AblationRow(name=name, accuracy=acc, f1=f1, roc_auc=auc, ece=ece, n=len(results))


# -------------------------------------------------------------------------- #
#  Main                                                                       #
# -------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="MedCAFAS Ablation Study")
    parser.add_argument("--dataset", type=str, default="pubmedqa",
                        choices=["pubmedqa", "medqa", "all"],
                        help="Dataset to evaluate on (default: pubmedqa)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Samples per dataset (default: 100)")
    args = parser.parse_args()

    print("Pre-loading models ...")
    _get_embedder(); _get_kb(); _get_nli_model(); _get_bm25()
    print("Models ready.\n")

    # Load datasets
    datasets: Dict[str, List[EvalSample]] = {}
    if args.dataset in ("pubmedqa", "all"):
        datasets["PubMedQA"] = load_eval_samples_pubmedqa(args.samples)
    if args.dataset in ("medqa", "all"):
        datasets["MedQA-USMLE"] = load_eval_samples(args.samples)

    # Ablation configurations
    ablations: List[Tuple[str, Callable]] = [
        ("Full pipeline",        _eval_full),
        ("- BM25 (cosine only)", _eval_no_bm25),
        ("- Entity check",       _eval_no_entity),
        ("- NLI critic",         _eval_no_nli),
        ("NLI only",             _eval_nli_only),
        ("- Sem-sim boost",      _eval_no_semsim),
    ]

    all_rows: Dict[str, List[AblationRow]] = {}  # ds_name -> rows

    for ds_name, samples in datasets.items():
        print(f"\n{'='*65}")
        print(f"  Dataset: {ds_name} ({len(samples)} samples)")
        print(f"{'='*65}")

        rows: List[AblationRow] = []
        for abl_name, abl_fn in ablations:
            print(f"\n  Running: {abl_name} ...", flush=True)
            t0 = time.time()
            results = [abl_fn(s) for s in samples]
            dt = time.time() - t0
            row = _score(abl_name, results)
            rows.append(row)
            print(f"  {abl_name:<24} Acc={row.accuracy:.1%}  F1={row.f1:.3f}  "
                  f"AUC={row.roc_auc:.3f}  ECE={row.ece:.3f}  ({dt:.0f}s)")

        all_rows[ds_name] = rows

    # -- Print summary tables -----------------------------------------------
    print("\n\n" + "=" * 75)
    print("  ABLATION STUDY RESULTS")
    print("=" * 75)

    for ds_name, rows in all_rows.items():
        print(f"\n  Dataset: {ds_name}")
        print(f"  {'Configuration':<24} {'Accuracy':>9} {'F1':>7} {'ROC-AUC':>8} {'ECE':>7}")
        print(f"  {'-'*24} {'-'*9} {'-'*7} {'-'*8} {'-'*7}")
        for r in rows:
            print(f"  {r.name:<24} {r.accuracy:>8.1%} {r.f1:>7.3f} {r.roc_auc:>8.3f} {r.ece:>7.3f}")

        # Delta from full pipeline
        full = rows[0]
        print(f"\n  Delta from full pipeline (accuracy):")
        for r in rows[1:]:
            delta = r.accuracy - full.accuracy
            print(f"  {r.name:<24} {delta:>+7.1%}")

    # -- LaTeX table --------------------------------------------------------
    print("\n\n  LaTeX Table (copy-paste into paper):")
    print("  \\begin{table}[h]")
    print("  \\centering")
    print("  \\caption{Ablation study results on MedCAFAS hallucination detection.}")
    print("  \\begin{tabular}{lcccc}")
    print("  \\toprule")
    print("  Configuration & Accuracy & F1 & ROC-AUC & ECE \\\\")
    print("  \\midrule")
    for ds_name, rows in all_rows.items():
        if len(all_rows) > 1:
            print(f"  \\multicolumn{{5}}{{l}}{{\\textit{{{ds_name}}}}} \\\\")
        for r in rows:
            name_tex = r.name.replace("- ", "$-$ ").replace("_", "\\_")
            print(f"  {name_tex} & {r.accuracy:.1%} & {r.f1:.3f} & {r.roc_auc:.3f} & {r.ece:.3f} \\\\")
        if len(all_rows) > 1:
            print("  \\midrule")
    print("  \\bottomrule")
    print("  \\end{tabular}")
    print("  \\end{table}")

    # -- Bar chart ----------------------------------------------------------
    fig, axes = plt.subplots(1, len(all_rows), figsize=(8 * len(all_rows), 6),
                             squeeze=False)

    colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#8b5cf6", "#ec4899"]

    for ax, (ds_name, rows) in zip(axes[0], all_rows.items()):
        names = [r.name for r in rows]
        accs  = [r.accuracy for r in rows]
        f1s   = [r.f1 for r in rows]

        x = np.arange(len(names))
        w = 0.35
        bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#2563eb", alpha=0.85)
        bars2 = ax.bar(x + w/2, f1s,  w, label="F1 Score", color="#dc2626", alpha=0.85)

        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"Ablation Study - {ds_name}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Annotate
        for bar in bars1:
            ax.annotate(f"{bar.get_height():.0%}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8, color="#2563eb")

    plt.tight_layout()
    out = "ablation_study.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] Ablation chart saved to {out}")
    plt.close()

    # -- Save JSON ----------------------------------------------------------
    out_json = "ablation_results.json"
    export = {}
    for ds_name, rows in all_rows.items():
        export[ds_name] = [
            {"name": r.name, "accuracy": r.accuracy, "f1": r.f1,
             "roc_auc": r.roc_auc, "ece": r.ece, "n": r.n}
            for r in rows
        ]
    with open(out_json, "w") as f:
        json.dump(export, f, indent=2)
    print(f"  [OK] Results saved to {out_json}")


if __name__ == "__main__":
    main()
