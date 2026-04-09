"""
accuracy_curve.py  --  Comprehensive accuracy curve for MedCAFAS
================================================================
Evaluates the pipeline at increasing sample sizes to prove metric stability.
Uses pre-computed Llama 3.1 JSON results for lightning-fast plotting.
"""

import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_ece_raw(y_true, y_score, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    y_true, y_score = np.array(y_true), np.array(y_score)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() > 0:
            bin_conf = y_score[mask].mean()
            bin_acc  = y_true[mask].mean()
            ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)

def main():
    sample_sizes = [20, 40, 60, 80, 100]
    
    print("Loading real LLM answers from data/eval_llm_final.json...")
    with open("data/eval_llm_final.json", "r") as f:
        data = json.load(f)

    # Apply our tuned weights and thresholds
    OLD_W = {"consistency": 0.25, "retrieval": 0.30, "critic": 0.30, "entity": 0.15}
    NEW_W = {"consistency": 0.15, "retrieval": 0.20, "critic": 0.55, "entity": 0.10}
    RISK_HIGH = 0.32

    for r in data:
        old_ret_risk = 1.0 - r['retrieval']
        old_nli_risk = 1.0 - r['nli_critic']
        old_ent_risk = r['entity_risk']
        known_risk_sum = (OLD_W['retrieval'] * old_ret_risk) + (OLD_W['critic'] * old_nli_risk) + (OLD_W['entity'] * old_ent_risk)
        cons_risk = (r['risk_score'] - known_risk_sum) / OLD_W['consistency']
        cons_risk = max(0.0, min(1.0, cons_risk))
        
        r['new_risk'] = (NEW_W['consistency'] * cons_risk) + (NEW_W['retrieval'] * old_ret_risk) + (NEW_W['critic'] * old_nli_risk) + (NEW_W['entity'] * old_ent_risk)

    metrics = {"acc": [], "f1": [], "auc": [], "ece": []}
    random.seed(42)

    print("\n============================================================")
    print(" Evaluating Stability across Sample Sizes (Llama 3.1)")
    print("============================================================")

    for n in sample_sizes:
        hall   = [r for r in data if r['is_hallucinated']]
        clean  = [r for r in data if not r['is_hallucinated']]
        
        if n >= len(data):
            subset = data
        else:
            half = n // 2
            subset = random.sample(hall, min(half, len(hall))) + random.sample(clean, min(half, len(clean)))
        
        y_true = [int(r['is_hallucinated']) for r in subset]
        y_score = [r['new_risk'] for r in subset]
        y_pred = [int(s >= RISK_HIGH) for s in y_score]
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5
        ece = compute_ece_raw(y_true, y_score)
        
        metrics["acc"].append(acc)
        metrics["f1"].append(f1)
        metrics["auc"].append(auc)
        metrics["ece"].append(ece)
        
        print(f"  n={n:>3d}  Accuracy={acc:.1%}  F1={f1:.3f}  ROC-AUC={auc:.3f}  ECE={ece:.3f}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    m_names = ["Accuracy", "F1 Score", "ROC-AUC", "ECE (lower is better)"]
    m_keys  = ["acc", "f1", "auc", "ece"]

    for ax, mname, mkey in zip(axes.flat, m_names, m_keys):
        ax.plot(sample_sizes, metrics[mkey], marker="D", color="#2563eb", linewidth=2.5, markersize=8, label="Llama 3.1 (PubMedQA)")
        ax.set_xlabel("Number of Evaluation Samples", fontsize=10)
        ax.set_ylabel(mname, fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(sample_sizes)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title(mname, fontsize=12, fontweight="bold")
        
        val = metrics[mkey][-1]
        label = f"{val:.1%}" if mkey != "ece" else f"{val:.3f}"
        ax.annotate(label, (sample_sizes[-1], val), textcoords="offset points", xytext=(8, 8), fontsize=10, fontweight="bold", color="#111827")

    fig.suptitle("MedCAFAS Evaluation Metric Stability\nLlama 3.1 | deberta-v3-base NLI | 65k-doc KB", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = "accuracy_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Accuracy curve saved to {out_path}")

if __name__ == "__main__":
    main()