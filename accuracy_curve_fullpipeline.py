"""
accuracy_curve.py (FULL PIPELINE VERSION)  
Uses eval_with_llm for ALL datasets to show real system capability.
Takes much longer (~2-5 hours) due to Ollama calls.
"""
from __future__ import annotations

import argparse, random, time, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval import (
    load_eval_samples, load_eval_samples_pubmedqa, eval_with_llm, compute_ece
)
from pipeline import _get_embedder, _get_kb, _get_nli_model, _get_bm25
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SAMPLE_SIZES = [20, 40, 60, 80, 100]

def _score(results):
    y_true  = [int(r.sample.is_hallucinated) for r in results]
    y_pred  = [int(r.predicted_hallucinated)  for r in results]
    y_score = [r.risk_score                   for r in results]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5
    ece = compute_ece(results)
    return acc, f1, auc, ece

def main():
    max_n = max(SAMPLE_SIZES)
    print("Pre-loading models ...")
    _get_embedder(); _get_kb(); _get_nli_model(); _get_bm25()
    print("Models ready.\n")

    loaders = {
        "PubMedQA":    load_eval_samples_pubmedqa,
        "MedQA-USMLE": load_eval_samples,
    }

    random.seed(42)
    results_by_ds = {}
    wall_start = time.time()

    for ds_name, loader in loaders.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}  (loading {max_n} samples once)")
        print(f"{'='*60}")

        all_samples = loader(max_n)
        print(f"  Loaded {len(all_samples)} samples total.")
        print(f"  Running eval_with_llm on {len(all_samples)} samples (FULL PIPELINE)...")
        t0 = time.time()
        all_results = [eval_with_llm(s) for s in all_samples]
        dt = time.time() - t0
        print(f"  Eval done in {dt:.0f}s ({dt/len(all_samples):.1f}s/sample).\n")

        metrics = {"acc": [], "f1": [], "auc": [], "ece": []}

        for n in SAMPLE_SIZES:
            if n >= len(all_results):
                subset = all_results
            else:
                hall   = [r for r in all_results if r.sample.is_hallucinated]
                clean  = [r for r in all_results if not r.sample.is_hallucinated]
                half   = n // 2
                subset = hall[:half] + clean[:half]
                random.shuffle(subset)

            acc, f1, auc, ece = _score(subset)
            metrics["acc"].append(acc)
            metrics["f1"].append(f1)
            metrics["auc"].append(auc)
            metrics["ece"].append(ece)
            print(f"  n={n:>3d}  Accuracy={acc:.1%}  F1={f1:.3f}  ROC-AUC={auc:.3f}  ECE={ece:.3f}")

        results_by_ds[ds_name] = metrics

    n_ds = len(results_by_ds)
    combined = {}
    for metric in ("acc", "f1", "auc", "ece"):
        combined[metric] = []
        for i in range(len(SAMPLE_SIZES)):
            vals = [results_by_ds[d][metric][i] for d in results_by_ds]
            combined[metric].append(sum(vals) / n_ds)
    
    elapsed = time.time() - wall_start

    # Summary
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
    print(f"\nTotal wall time: {elapsed/60:.1f} min")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_names = ["Accuracy", "F1 Score", "ROC-AUC", "ECE (lower is better)"]
    metric_keys  = ["acc",      "f1",       "auc",     "ece"]

    ds_colors  = {"PubMedQA": "#2563eb", "MedQA-USMLE": "#dc2626"}
    ds_markers = {"PubMedQA": "o",       "MedQA-USMLE": "s"}
    combined_color = "#111827"

    for ax, mname, mkey in zip(axes.flat, metric_names, metric_keys):
        for ds_name, m in results_by_ds.items():
            ax.plot(SAMPLE_SIZES, m[mkey],
                    marker=ds_markers[ds_name],
                    color=ds_colors[ds_name],
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
        ax.set_title(mname,fontsize=12, fontweight="bold")

        last_i = len(SAMPLE_SIZES) - 1
        val = combined[mkey][last_i]
        label = f"{val:.1%}" if mkey != "ece" else f"{val:.3f}"
        ax.annotate(label, (SAMPLE_SIZES[last_i], val),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold", color=combined_color)

    ds_list = " + ".join(results_by_ds.keys())
    fig.suptitle(
        f"MedCAFAS Accuracy Curve (Full Pipeline with Ollama)\n"
        f"{ds_list} | all 3 layers | deberta-v3-base NLI | 50k-doc KB",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=150, bbox_inches="tight")
    print(f"\n[OK] Accuracy curve saved to accuracy_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
