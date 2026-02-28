"""
baselines.py -- Baseline comparisons for MedCAFAS
=================================================

Compares MedCAFAS against simple baselines to demonstrate the value
of the multi-layer approach.  Essential for research papers.

Baselines:
  1. Random classifier           (coin flip)
  2. Always-positive             (flag everything as hallucinated)
  3. Always-negative             (flag nothing)
  4. BM25-only                   (threshold on BM25 retrieval score)
  5. Cosine-only                 (threshold on FAISS cosine similarity)
  6. NLI-only                    (threshold on NLI entailment score)
  7. Perplexity proxy            (answer length as proxy for complexity)
  8. MedCAFAS (full pipeline)    (our system)

Usage:
    python baselines.py                          # PubMedQA, 100 samples
    python baselines.py --dataset all --samples 80
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


@dataclass
class BaselineRow:
    name:     str
    accuracy: float
    f1:       float
    roc_auc:  float
    ece:      float


def _metrics(y_true, y_pred, y_score=None) -> Tuple[float, float, float, float]:
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score if y_score is not None else y_pred)
    except ValueError:
        auc = 0.5
    # Simple ECE approximation
    if y_score is not None:
        scores = np.array(y_score)
        labels = np.array(y_true)
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (scores >= lo) & (scores < hi)
            if mask.sum() > 0:
                ece += (mask.sum() / len(scores)) * abs(scores[mask].mean() - labels[mask].mean())
    else:
        ece = abs(np.mean(y_pred) - np.mean(y_true))
    return acc, f1, auc, float(ece)


# -------------------------------------------------------------------------- #
#  Baseline implementations                                                   #
# -------------------------------------------------------------------------- #

def baseline_random(samples: List[EvalSample], seed=42) -> BaselineRow:
    """Random 50/50 classification."""
    rng = random.Random(seed)
    y_true = [int(s.is_hallucinated) for s in samples]
    y_pred = [rng.randint(0, 1) for _ in samples]
    y_score = [rng.random() for _ in samples]
    acc, f1, auc, ece = _metrics(y_true, y_pred, y_score)
    return BaselineRow("Random", acc, f1, auc, ece)


def baseline_always_positive(samples: List[EvalSample]) -> BaselineRow:
    """Flag everything as hallucinated."""
    y_true = [int(s.is_hallucinated) for s in samples]
    y_pred = [1] * len(samples)
    y_score = [1.0] * len(samples)
    acc, f1, auc, ece = _metrics(y_true, y_pred, y_score)
    return BaselineRow("Always positive", acc, f1, auc, ece)


def baseline_always_negative(samples: List[EvalSample]) -> BaselineRow:
    """Flag nothing as hallucinated."""
    y_true = [int(s.is_hallucinated) for s in samples]
    y_pred = [0] * len(samples)
    y_score = [0.0] * len(samples)
    acc, f1, auc, ece = _metrics(y_true, y_pred, y_score)
    return BaselineRow("Always negative", acc, f1, auc, ece)


def baseline_length_proxy(samples: List[EvalSample], threshold=50) -> BaselineRow:
    """
    Perplexity proxy: flag answers shorter than threshold words.
    Rationale: hallucinated answers are sometimes shorter or more generic.
    """
    y_true = [int(s.is_hallucinated) for s in samples]
    lengths = [len(s.answer.split()) for s in samples]
    median_len = np.median(lengths)
    # Longer answers are more likely to contain fabricated claims
    y_score = [min(1.0, l / (2 * median_len)) for l in lengths]
    y_pred = [int(l > median_len) for l in lengths]
    acc, f1, auc, ece = _metrics(y_true, y_pred, y_score)
    return BaselineRow("Length proxy", acc, f1, auc, ece)


def baseline_bm25_only(samples: List[EvalSample]) -> BaselineRow:
    """Use only BM25 retrieval score (no neural components)."""
    from rank_bm25 import BM25Okapi
    import pickle

    bm25_data = _get_bm25()
    if not bm25_data or "bm25" not in bm25_data:
        return BaselineRow("BM25-only", 0.5, 0.0, 0.5, 0.5)

    bm25 = bm25_data["bm25"]
    y_true = [int(s.is_hallucinated) for s in samples]
    y_score = []
    y_pred = []

    for s in samples:
        tokens = s.answer.lower().split()
        raw = bm25.get_scores(tokens)
        top_score = float(raw.max()) if len(raw) > 0 else 0.0
        # Normalise to [0,1] — higher = better match = less risky
        norm = min(1.0, top_score / 20.0)  # BM25 scores can be large
        risk = 1.0 - norm
        y_score.append(risk)
        y_pred.append(int(risk > 0.5))

    acc, f1, auc, ece = _metrics(y_true, y_pred, y_score)
    return BaselineRow("BM25-only", acc, f1, auc, ece)


def baseline_cosine_only(samples: List[EvalSample]) -> BaselineRow:
    """Use only FAISS cosine similarity (no BM25, no NLI)."""
    y_true = [int(s.is_hallucinated) for s in samples]
    y_score = []
    y_pred = []

    # Temporarily disable BM25
    orig = config.BM25_WEIGHT
    config.BM25_WEIGHT = 0.0
    try:
        for s in samples:
            ret_score, _ = layer2_retrieval(s.answer)
            risk = 1.0 - ret_score
            y_score.append(risk)
            y_pred.append(int(risk > 0.5))
    finally:
        config.BM25_WEIGHT = orig

    acc, f1, auc, ece = _metrics(y_true, y_pred, y_score)
    return BaselineRow("Cosine-only", acc, f1, auc, ece)


def baseline_nli_only(samples: List[EvalSample]) -> BaselineRow:
    """Use only NLI entailment score (no retrieval scoring)."""
    y_true = [int(s.is_hallucinated) for s in samples]
    y_score = []
    y_pred = []

    for s in samples:
        _, citations = layer2_retrieval(s.answer)  # still need citations for NLI
        critic, _ = layer3_critic(s.answer, citations, question="")
        risk = 1.0 - critic
        y_score.append(risk)
        y_pred.append(int(risk > 0.5))

    acc, f1, auc, ece = _metrics(y_true, y_pred, y_score)
    return BaselineRow("NLI-only", acc, f1, auc, ece)


def baseline_medcafas(samples: List[EvalSample]) -> BaselineRow:
    """Full MedCAFAS pipeline."""
    results = [eval_no_llm(s) for s in samples]
    y_true  = [int(r.sample.is_hallucinated) for r in results]
    y_pred  = [int(r.predicted_hallucinated)  for r in results]
    y_score = [r.risk_score                   for r in results]
    acc, f1, auc, ece = _metrics(y_true, y_pred, y_score)
    return BaselineRow("MedCAFAS (ours)", acc, f1, auc, ece)


# -------------------------------------------------------------------------- #
#  Main                                                                       #
# -------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="MedCAFAS Baseline Comparisons")
    parser.add_argument("--dataset", type=str, default="pubmedqa",
                        choices=["pubmedqa", "medqa", "all"])
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    print("Pre-loading models ...")
    _get_embedder(); _get_kb(); _get_nli_model(); _get_bm25()
    print("Models ready.\n")

    datasets: Dict[str, List[EvalSample]] = {}
    if args.dataset in ("pubmedqa", "all"):
        datasets["PubMedQA"] = load_eval_samples_pubmedqa(args.samples)
    if args.dataset in ("medqa", "all"):
        datasets["MedQA-USMLE"] = load_eval_samples(args.samples)

    all_results: Dict[str, List[BaselineRow]] = {}

    for ds_name, samples in datasets.items():
        print(f"\n{'='*65}")
        print(f"  Baselines on: {ds_name} ({len(samples)} samples)")
        print(f"{'='*65}")

        baselines = [
            ("Random",           lambda s: baseline_random(s)),
            ("Always positive",  lambda s: baseline_always_positive(s)),
            ("Always negative",  lambda s: baseline_always_negative(s)),
            ("Length proxy",     lambda s: baseline_length_proxy(s)),
            ("BM25-only",        lambda s: baseline_bm25_only(s)),
            ("Cosine-only",      lambda s: baseline_cosine_only(s)),
            ("NLI-only",         lambda s: baseline_nli_only(s)),
            ("MedCAFAS (ours)",  lambda s: baseline_medcafas(s)),
        ]

        rows: List[BaselineRow] = []
        for name, fn in baselines:
            print(f"  Running: {name} ...", end=" ", flush=True)
            t0 = time.time()
            row = fn(samples)
            dt = time.time() - t0
            rows.append(row)
            print(f"Acc={row.accuracy:.1%}  F1={row.f1:.3f}  AUC={row.roc_auc:.3f}  ({dt:.0f}s)")

        all_results[ds_name] = rows

    # -- Summary table ------------------------------------------------------
    print("\n\n" + "=" * 75)
    print("  BASELINE COMPARISON RESULTS")
    print("=" * 75)

    for ds_name, rows in all_results.items():
        print(f"\n  Dataset: {ds_name}")
        print(f"  {'Method':<24} {'Accuracy':>9} {'F1':>7} {'ROC-AUC':>8} {'ECE':>7}")
        print(f"  {'-'*24} {'-'*9} {'-'*7} {'-'*8} {'-'*7}")
        for r in rows:
            marker = " **" if r.name == "MedCAFAS (ours)" else ""
            print(f"  {r.name:<24} {r.accuracy:>8.1%} {r.f1:>7.3f} {r.roc_auc:>8.3f} {r.ece:>7.3f}{marker}")

    # -- LaTeX table --------------------------------------------------------
    print("\n\n  LaTeX Table:")
    print("  \\begin{table}[h]")
    print("  \\centering")
    print("  \\caption{Comparison of MedCAFAS against baseline methods.}")
    print("  \\begin{tabular}{lcccc}")
    print("  \\toprule")
    print("  Method & Accuracy & F1 & ROC-AUC & ECE \\\\")
    print("  \\midrule")
    for ds_name, rows in all_results.items():
        if len(all_results) > 1:
            print(f"  \\multicolumn{{5}}{{l}}{{\\textit{{{ds_name}}}}} \\\\")
        for r in rows:
            name = r.name.replace("_", "\\_")
            bold = "\\textbf" if r.name == "MedCAFAS (ours)" else ""
            if bold:
                print(f"  \\textbf{{{name}}} & \\textbf{{{r.accuracy:.1%}}} & "
                      f"\\textbf{{{r.f1:.3f}}} & \\textbf{{{r.roc_auc:.3f}}} & "
                      f"\\textbf{{{r.ece:.3f}}} \\\\")
            else:
                print(f"  {name} & {r.accuracy:.1%} & {r.f1:.3f} & {r.roc_auc:.3f} & {r.ece:.3f} \\\\")
    print("  \\bottomrule")
    print("  \\end{tabular}")
    print("  \\end{table}")

    # -- Bar chart ----------------------------------------------------------
    fig, axes = plt.subplots(1, len(all_results), figsize=(10 * len(all_results), 7),
                             squeeze=False)

    for ax, (ds_name, rows) in zip(axes[0], all_results.items()):
        names = [r.name for r in rows]
        accs  = [r.accuracy for r in rows]

        colors_list = ["#94a3b8"] * (len(rows) - 1) + ["#2563eb"]  # MedCAFAS highlighted
        bars = ax.barh(names, accs, color=colors_list, edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Accuracy", fontsize=11)
        ax.set_title(f"Baseline Comparison - {ds_name}", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        for bar, acc in zip(bars, accs):
            ax.annotate(f"{acc:.1%}",
                        xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = "baselines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] Baseline chart saved to {out}")
    plt.close()

    # -- Save JSON ----------------------------------------------------------
    out_json = "baseline_results.json"
    export = {}
    for ds_name, rows in all_results.items():
        export[ds_name] = [
            {"name": r.name, "accuracy": r.accuracy, "f1": r.f1,
             "roc_auc": r.roc_auc, "ece": r.ece}
            for r in rows
        ]
    with open(out_json, "w") as f:
        json.dump(export, f, indent=2)
    print(f"  [OK] Results saved to {out_json}")


if __name__ == "__main__":
    main()
