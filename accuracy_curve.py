"""
accuracy_curve.py  --  Comprehensive accuracy curve for MedCAFAS
================================================================

Evaluates the pipeline on ALL available datasets at increasing sample sizes
and produces a multi-panel accuracy + F1 + ROC-AUC + ECE curve.

Datasets:
  1. PubMedQA      -- paragraph-length answers, strong NLI signal
  2. MedQA-USMLE   -- short (2-5 word) answers, harder for NLI
  3. LLM (phi3.5)  -- real phi3.5 outputs, gold-standard labels (optional)

Optimisation: each dataset is loaded ONCE at the maximum sample size and
then sub-sampled for the smaller sizes.  This avoids redundant phi3.5 calls
(100 LLM calls instead of 300).

Usage:
    python accuracy_curve.py               # PubMedQA + MedQA only  (~20 min)
    python accuracy_curve.py --llm         # + LLM dataset           (~45-60 min)
    python accuracy_curve.py --llm --sizes 50,100

Output:
    accuracy_curve.png  (saved in project root)
"""

from __future__ import annotations

import argparse
import random
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval import (
    load_eval_samples,
    load_eval_samples_pubmedqa,
    load_eval_samples_llm,
    eval_no_llm,
    compute_ece,
)
from pipeline import _get_embedder, _get_kb, _get_nli_model, _get_bm25
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _score(results):
    """Compute metrics dict from a list of EvalResults."""
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


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="MedCAFAS accuracy curve generator")
    parser.add_argument("--llm", action="store_true",
                        help="Include LLM dataset (real phi3.5 outputs, requires Ollama)")
    parser.add_argument("--sizes", type=str, default="20,40,60,80,100",
                        help="Comma-separated sample sizes (default: 20,40,60,80,100)")
    args = parser.parse_args()

    sample_sizes = sorted(int(s) for s in args.sizes.split(","))
    max_n = max(sample_sizes)

    # Pre-load all models once
    print("Pre-loading models ...")
    _get_embedder(); _get_kb(); _get_nli_model(); _get_bm25()
    print("Models ready.\n")

    # -- Define datasets (name -> loader) ----------------------------------
    loaders = {
        "PubMedQA":    load_eval_samples_pubmedqa,
        "MedQA-USMLE": load_eval_samples,
    }
    if args.llm:
        loaders["LLM (phi3.5)"] = load_eval_samples_llm

    # -- Load each dataset ONCE at the maximum sample size -----------------
    #    Then evaluate at that full set; sub-sample for smaller sizes.
    random.seed(42)                       # reproducible sub-sampling
    results_by_ds = {}                    # ds_name -> {acc:[], f1:[], ...}
    wall_start = time.time()

    for ds_name, loader in loaders.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}  (loading {max_n} samples once)")
        print(f"{'='*60}")

        # Load the full sample set once
        all_samples = loader(max_n)
        print(f"  Loaded {len(all_samples)} samples total.")

        # Evaluate ALL samples through Layers 2+3 once
        print(f"  Running eval_no_llm on {len(all_samples)} samples ...")
        t0 = time.time()
        all_results = [eval_no_llm(s) for s in all_samples]
        dt = time.time() - t0
        print(f"  Eval done in {dt:.0f}s.\n")

        metrics = {"acc": [], "f1": [], "auc": [], "ece": []}

        for n in sample_sizes:
            if n >= len(all_results):
                subset = all_results
            else:
                # Deterministic balanced sub-sample: pick n/2 hallucinated + n/2 correct
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

            print(f"  n={n:>3d}  Accuracy={acc:.1%}  F1={f1:.3f}  "
                  f"ROC-AUC={auc:.3f}  ECE={ece:.3f}")

        results_by_ds[ds_name] = metrics

    # -- Combined (mean across datasets per sample size index) -------------
    n_ds = len(results_by_ds)
    combined = {}
    for metric in ("acc", "f1", "auc", "ece"):
        combined[metric] = []
        for i in range(len(sample_sizes)):
            vals = [results_by_ds[d][metric][i] for d in results_by_ds]
            combined[metric].append(sum(vals) / n_ds)

    elapsed = time.time() - wall_start

    # -- Summary table -----------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  SUMMARY  (accuracy at each sample size)")
    print("=" * 70)
    header = f"{'Dataset':<18}" + "".join(f"  n={n:<5}" for n in sample_sizes)
    print(header)
    print("-" * len(header))
    for ds_name, m in results_by_ds.items():
        row = f"{ds_name:<18}" + "".join(f"  {a:.1%} " for a in m["acc"])
        print(row)
    print("-" * len(header))
    row = f"{'COMBINED':<18}" + "".join(f"  {a:.1%} " for a in combined["acc"])
    print(row)
    print(f"\nTotal wall time: {elapsed/60:.1f} min")

    # -- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_names = ["Accuracy", "F1 Score", "ROC-AUC", "ECE (lower is better)"]
    metric_keys  = ["acc",      "f1",       "auc",     "ece"]

    ds_colors = {
        "PubMedQA":      "#2563eb",
        "MedQA-USMLE":   "#dc2626",
        "LLM (phi3.5)":  "#16a34a",
    }
    combined_color = "#111827"
    ds_markers = {"PubMedQA": "o", "MedQA-USMLE": "s", "LLM (phi3.5)": "^"}

    for ax, mname, mkey in zip(axes.flat, metric_names, metric_keys):
        for ds_name, m in results_by_ds.items():
            ax.plot(sample_sizes, m[mkey],
                    marker=ds_markers.get(ds_name, "o"),
                    color=ds_colors.get(ds_name, "#666"),
                    linewidth=2, markersize=7, label=ds_name)
        # Combined curve (dashed)
        ax.plot(sample_sizes, combined[mkey],
                marker="D", color=combined_color, linewidth=2.5,
                markersize=8, linestyle="--", label="Combined (mean)")

        ax.set_xlabel("Number of Evaluation Samples", fontsize=10)
        ax.set_ylabel(mname, fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(sample_sizes)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title(mname, fontsize=12, fontweight="bold")

        # Annotate the combined value at the largest sample size
        last_i = len(sample_sizes) - 1
        val = combined[mkey][last_i]
        label = f"{val:.1%}" if mkey != "ece" else f"{val:.3f}"
        ax.annotate(label, (sample_sizes[last_i], val),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold", color=combined_color)

    mode = "all 3 layers (LLM dataset included)" if args.llm else "Layers 2-3 only (no-LLM)"
    ds_list = " + ".join(results_by_ds.keys())
    fig.suptitle(
        f"MedCAFAS Accuracy Curve\n"
        f"{ds_list} | {mode} | deberta-v3-base NLI | 8000-doc KB",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out_path = "accuracy_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Accuracy curve saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
