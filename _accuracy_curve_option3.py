"""
Accuracy curve with Option 3: Full pipeline (eval_with_llm) for LLM dataset.

Uses cached eval_no_llm results for PubMedQA + MedQA-USMLE (Layers 2-3 only),
then evaluates LLM dataset with eval_with_llm (full 3-layer pipeline including
Layer 1 consistency with Ollama).
"""
from __future__ import annotations

import random, time, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval import (
    load_eval_samples, load_eval_samples_pubmedqa, load_eval_samples_llm,
    eval_no_llm, eval_with_llm, compute_ece
)
from pipeline import _get_embedder, _get_kb, _get_nli_model, _get_bm25
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import config

SAMPLE_SIZES = [20, 40, 60, 80, 100]

# Cached eval_no_llm results (Layers 2-3 only, post-Phase 3 with RISK_HIGH=0.30)
CACHED_NO_LLM = {
    "PubMedQA": {
        "acc": [0.500, 0.525, 0.533, 0.525, 0.530],
        "f1":  [0.571, 0.571, 0.552, 0.561, 0.541],
        "auc": [0.500, 0.503, 0.503, 0.521, 0.529],
        "ece": [0.480, 0.420, 0.437, 0.421, 0.418],
    },
    "MedQA-USMLE": {
        "acc": [0.700, 0.750, 0.700, 0.725, 0.740],
        "f1":  [0.667, 0.727, 0.667, 0.697, 0.714],
        "auc": [0.720, 0.786, 0.735, 0.759, 0.784],
        "ece": [0.323, 0.284, 0.312, 0.299, 0.298],
    },
}


def _score(results):
    """Compute metrics from EvalResult list."""
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
    return acc, f1, auc, ece


def main():
    max_n = max(SAMPLE_SIZES)

    print(f"Config: RISK_LOW={config.RISK_LOW}, RISK_HIGH={config.RISK_HIGH}")
    print("\nPre-loading models ...")
    _get_embedder(); _get_kb(); _get_nli_model(); _get_bm25()
    print("Models ready.\n")

    # -- Evaluate LLM dataset with FULL pipeline -----
    print(f"\n{'='*60}")
    print(f"  Dataset: LLM (phi3.5) — Full 3-layer pipeline")
    print(f"  Loading {max_n} samples (requires Ollama)...")
    print(f"{'='*60}")

    all_samples = load_eval_samples_llm(max_n)
    print(f"  Loaded {len(all_samples)} samples total.\n")

    print(f"  Running eval_with_llm (Layer 1 + 2 + 3) on {len(all_samples)} samples ...")
    t0 = time.time()
    all_results = [eval_with_llm(s) for s in all_samples]
    dt = time.time() - t0
    print(f"  Eval done in {dt:.0f}s ({dt/len(all_results):.1f}s/sample).\n")

    # -- Sub-sample for each size ---
    random.seed(42)
    llm_metrics = {"acc": [], "f1": [], "auc": [], "ece": []}
    for n in SAMPLE_SIZES:
        if n >= len(all_results):
            subset = all_results
        else:
            hall  = [r for r in all_results if r.sample.is_hallucinated]
            clean = [r for r in all_results if not r.sample.is_hallucinated]
            half  = n // 2
            subset = hall[:half] + clean[:half]
            random.shuffle(subset)
        acc, f1, auc, ece = _score(subset)
        llm_metrics["acc"].append(acc)
        llm_metrics["f1"].append(f1)
        llm_metrics["auc"].append(auc)
        llm_metrics["ece"].append(ece)
        print(f"  n={n:>3d}  Accuracy={acc:.1%}  F1={f1:.3f}  ROC-AUC={auc:.3f}  ECE={ece:.3f}")

    # -- Combine with cached no-LLM results -----
    results_by_ds = {
        "PubMedQA":     CACHED_NO_LLM["PubMedQA"],
        "MedQA-USMLE":  CACHED_NO_LLM["MedQA-USMLE"],
        "LLM (phi3.5)": llm_metrics,
    }

    n_ds = len(results_by_ds)
    combined = {}
    for metric in ("acc", "f1", "auc", "ece"):
        combined[metric] = [
            sum(results_by_ds[d][metric][i] for d in results_by_ds) / n_ds
            for i in range(len(SAMPLE_SIZES))
        ]

    # -- Summary ---
    print(f"\n\n{'='*70}")
    print("  SUMMARY  (accuracy at each sample size)")
    print("=" * 70)
    header = f"{'Dataset':<18}" + "".join(f"  n={n:<5}" for n in SAMPLE_SIZES)
    print(header)
    print("-" * len(header))
    for ds_name, m in results_by_ds.items():
        row = f"{ds_name:<18}" + "".join(f"  {a:.1%} " for a in m["acc"])
        print(row)
    print("-" * len(header))
    row = f"{'COMBINED':<18}" + "".join(f"  {a:.1%} " for a in combined["acc"])
    print(row)
    print(f"\nTotal eval time: {dt/60:.1f} min")

    # -- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_names = ["Accuracy", "F1 Score", "ROC-AUC", "ECE (lower is better)"]
    metric_keys  = ["acc",      "f1",       "auc",     "ece"]

    ds_colors  = {"PubMedQA": "#2563eb", "MedQA-USMLE": "#dc2626", "LLM (phi3.5)": "#16a34a"}
    ds_markers = {"PubMedQA": "o",       "MedQA-USMLE": "s",       "LLM (phi3.5)": "^"}
    combined_color = "#111827"

    for ax, mname, mkey in zip(axes.flat, metric_names, metric_keys):
        for ds_name, m in results_by_ds.items():
            ax.plot(SAMPLE_SIZES, m[mkey],
                    marker=ds_markers.get(ds_name, "o"),
                    color=ds_colors.get(ds_name, "#666"),
                    linewidth=2, markersize=7, label=ds_name)
        ax.plot(SAMPLE_SIZES, combined[mkey],
                marker="D", color=combined_color, linewidth=2.5,
                markersize=8, linestyle="--", label="Combined (mean)")
        ax.set_xlabel("Number of Evaluation Samples", fontsize=10)
        ax.set_ylabel(mname, fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(SAMPLE_SIZES)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title(mname, fontsize=12, fontweight="bold")

        last_i = len(SAMPLE_SIZES) - 1
        val = combined[mkey][last_i]
        label = f"{val:.1%}" if mkey != "ece" else f"{val:.3f}"
        ax.annotate(label, (SAMPLE_SIZES[last_i], val),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold", color=combined_color)

    ds_list = " + ".join(results_by_ds.keys())
    fig.suptitle(
        f"MedCAFAS Accuracy Curve (Option 3 — Full Pipeline)\n"
        f"{ds_list} | LLM: all 3 layers, Others: Layers 2-3 | "
        f"deberta-v3-base NLI | 50k-doc KB",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=150, bbox_inches="tight")
    print(f"\n[OK] Accuracy curve saved to accuracy_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
