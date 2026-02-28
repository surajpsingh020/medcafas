"""
system_health.py — Generate Final System Health LaTeX Table
─────────────────────────────────────────────────────────────
Runs MedCAFAS evaluation across all 3 datasets and produces a
publication-ready LaTeX table comparing PubMedQA vs MedQA vs LLM-Dataset.

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, ECE

Usage:
    python system_health.py                    # no-LLM mode, 60 samples each
    python system_health.py --samples 100      # more thorough
    python system_health.py --with-llm         # full pipeline (requires Ollama)
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from eval import (
    EvalResult,
    compute_ece,
    eval_no_llm,
    eval_with_llm,
    load_eval_samples,
    load_eval_samples_llm,
    load_eval_samples_pubmedqa,
    bootstrap_ci,
)
from pipeline import _get_bm25, _get_embedder, _get_kb, _get_nli_model


def compute_all_metrics(results: List[EvalResult]) -> Dict:
    """Compute comprehensive metrics for a set of eval results."""
    y_true = [int(r.sample.is_hallucinated) for r in results]
    y_pred = [int(r.predicted_hallucinated) for r in results]
    y_score = [r.risk_score for r in results]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    ece = compute_ece(results)

    # Bootstrap CIs
    try:
        _, acc_lo, acc_hi = bootstrap_ci(results, "accuracy")
        _, f1_lo, f1_hi = bootstrap_ci(results, "f1")
    except Exception:
        acc_lo = acc_hi = f1_lo = f1_hi = float("nan")

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "ece": round(ece, 4),
        "n_samples": len(results),
        "n_correct": sum(r.correct for r in results),
        "acc_ci": f"[{acc_lo:.3f}, {acc_hi:.3f}]",
        "f1_ci": f"[{f1_lo:.3f}, {f1_hi:.3f}]",
    }


def generate_latex_table(
    metrics: Dict[str, Dict], mode: str = "no-LLM"
) -> str:
    """Generate a publication-ready LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{MedCAFAS System Health — Cross-Dataset Evaluation (" + mode + r")}",
        r"\label{tab:system_health}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
    ]

    # Header
    datasets = list(metrics.keys())
    header = r"\textbf{Metric}"
    for ds in datasets:
        header += rf" & \textbf{{{ds}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Metric rows
    metric_labels = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall (Hall.)"),
        ("f1", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
        ("ece", "ECE ($\\downarrow$)"),
        ("n_samples", "\\# Samples"),
    ]

    for key, label in metric_labels:
        row = label
        for ds in datasets:
            val = metrics[ds].get(key, "—")
            if isinstance(val, float):
                if key == "n_samples":
                    row += f" & {int(val)}"
                elif np.isnan(val):
                    row += " & —"
                else:
                    row += f" & {val:.3f}"
            elif isinstance(val, int):
                row += f" & {val}"
            else:
                row += f" & {val}"
        row += r" \\"
        lines.append(row)

    # CI rows
    lines.append(r"\midrule")
    for key, label in [("acc_ci", "Accuracy 95\\% CI"), ("f1_ci", "F1 95\\% CI")]:
        row = label
        for ds in datasets:
            val = metrics[ds].get(key, "—")
            row += f" & {val}"
        row += r" \\"
        lines.append(row)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MedCAFAS System Health Table")
    parser.add_argument(
        "--samples",
        type=int,
        default=60,
        help="Samples per dataset (default: 60)",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Use full 3-layer pipeline (requires Ollama, very slow)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/system_health.tex",
        help="Output LaTeX file path",
    )
    args = parser.parse_args()

    # Pre-load
    print("Pre-loading models...")
    _get_embedder()
    _get_kb()
    _get_nli_model()
    _get_bm25()
    print("  Models ready.\n")

    eval_fn = eval_with_llm if args.with_llm else eval_no_llm
    mode = "Full Pipeline" if args.with_llm else "No-LLM (Layers 2+3)"

    all_metrics: Dict[str, Dict] = {}

    # ── Dataset 1: PubMedQA ───────────────────────────────────────────────
    print("=" * 60)
    print("  Dataset 1: PubMedQA (paragraph answers, proper NLI test)")
    print("=" * 60)
    samples_pqa = load_eval_samples_pubmedqa(args.samples)
    results_pqa = []
    for i, s in enumerate(samples_pqa, 1):
        label = "HALL" if s.is_hallucinated else "good"
        print(
            f"  [{i:>3}/{len(samples_pqa)}] [{label}] {s.question[:55]}...",
            end=" ",
            flush=True,
        )
        r = eval_fn(s)
        status = "OK" if r.correct else "XX"
        print(f"-> {r.risk_flag:7} {status}")
        results_pqa.append(r)

    all_metrics["PubMedQA"] = compute_all_metrics(results_pqa)
    print(
        f"\n  PubMedQA: Acc={all_metrics['PubMedQA']['accuracy']:.1%}  "
        f"F1={all_metrics['PubMedQA']['f1']:.3f}\n"
    )

    # ── Dataset 2: MedQA-USMLE ───────────────────────────────────────────
    print("=" * 60)
    print("  Dataset 2: MedQA-USMLE (short MCQ answers)")
    print("=" * 60)
    samples_mqa = load_eval_samples(args.samples)
    results_mqa = []
    for i, s in enumerate(samples_mqa, 1):
        label = "HALL" if s.is_hallucinated else "good"
        print(
            f"  [{i:>3}/{len(samples_mqa)}] [{label}] {s.question[:55]}...",
            end=" ",
            flush=True,
        )
        r = eval_fn(s)
        status = "OK" if r.correct else "XX"
        print(f"-> {r.risk_flag:7} {status}")
        results_mqa.append(r)

    all_metrics["MedQA"] = compute_all_metrics(results_mqa)
    print(
        f"\n  MedQA: Acc={all_metrics['MedQA']['accuracy']:.1%}  "
        f"F1={all_metrics['MedQA']['f1']:.3f}\n"
    )

    # ── Dataset 3: LLM (only if --with-llm) ──────────────────────────────
    if args.with_llm:
        print("=" * 60)
        print("  Dataset 3: LLM Decision-Matching (phi3.5 outputs)")
        print("=" * 60)
        samples_llm = load_eval_samples_llm(args.samples)
        results_llm = []
        for i, s in enumerate(samples_llm, 1):
            label = "HALL" if s.is_hallucinated else "good"
            print(
                f"  [{i:>3}/{len(samples_llm)}] [{label}] {s.question[:55]}...",
                end=" ",
                flush=True,
            )
            r = eval_fn(s)
            status = "OK" if r.correct else "XX"
            print(f"-> {r.risk_flag:7} {status}")
            results_llm.append(r)

        all_metrics["LLM (phi3.5)"] = compute_all_metrics(results_llm)
        print(
            f"\n  LLM: Acc={all_metrics['LLM (phi3.5)']['accuracy']:.1%}  "
            f"F1={all_metrics['LLM (phi3.5)']['f1']:.3f}\n"
        )

    # ── Generate LaTeX ────────────────────────────────────────────────────
    latex = generate_latex_table(all_metrics, mode=mode)

    print("\n" + "=" * 60)
    print("  FINAL SYSTEM HEALTH TABLE (LaTeX)")
    print("=" * 60)
    print(latex)

    with open(args.output, "w") as f:
        f.write(latex)
    print(f"\n  LaTeX table saved to {args.output}")

    # Also save raw JSON metrics
    json_path = args.output.replace(".tex", ".json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Raw metrics saved to {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for ds, m in all_metrics.items():
        print(
            f"  {ds:20}  Acc={m['accuracy']:.1%}  F1={m['f1']:.3f}  "
            f"AUC={m['roc_auc']:.3f}  ECE={m['ece']:.3f}"
        )

    # Combined accuracy
    total_correct = sum(m["n_correct"] for m in all_metrics.values())
    total_samples = sum(m["n_samples"] for m in all_metrics.values())
    combined = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n  Combined: {total_correct}/{total_samples} = {combined:.1%}")


if __name__ == "__main__":
    main()
