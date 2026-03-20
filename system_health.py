"""
system_health.py — Generate Final System Health LaTeX Table
─────────────────────────────────────────────────────────────
Runs MedCAFAS evaluation on the optimized Llama 3.1 dataset and produces a
publication-ready LaTeX table.
"""

import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from eval import EvalResult, EvalSample, compute_ece, bootstrap_ci

def compute_all_metrics(results):
    y_true = [int(r.sample.is_hallucinated) for r in results]
    y_pred = [int(r.predicted_hallucinated) for r in results]
    y_score = [r.risk_score for r in results]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    ece = compute_ece(results)

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

def generate_latex_table(metrics, mode="Optimized"):
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{MedCAFAS System Health — Final Evaluation (" + mode + r")}",
        r"\label{tab:system_health}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}", r"\toprule",
    ]

    datasets = list(metrics.keys())
    header = r"\textbf{Metric}"
    for ds in datasets: header += rf" & \textbf{{{ds}}}"
    header += r" \\"
    lines.extend([header, r"\midrule"])

    metric_labels = [
        ("accuracy", "Accuracy"), ("precision", "Precision"), 
        ("recall", "Recall (Safety)"), ("f1", "F1 Score"), 
        ("roc_auc", "ROC-AUC"), ("ece", "ECE ($\\downarrow$)"), 
        ("n_samples", "\\# Samples"),
    ]

    for key, label in metric_labels:
        row = label
        for ds in datasets:
            val = metrics[ds].get(key, "—")
            if isinstance(val, float): row += f" & {int(val)}" if key == "n_samples" else (" & —" if np.isnan(val) else f" & {val:.3f}")
            else: row += f" & {val}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\midrule")
    for key, label in [("acc_ci", "Accuracy 95\\% CI"), ("f1_ci", "F1 95\\% CI")]:
        row = label
        for ds in datasets: row += f" & {metrics[ds].get(key, '—')}"
        row += r" \\"
        lines.append(row)

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)

def main():
    print("Loading real LLM answers from data/eval_llm_final.json...")
    with open("data/eval_llm_final.json", "r") as f: data = json.load(f)

    OLD_W = {"consistency": 0.25, "retrieval": 0.30, "critic": 0.30, "entity": 0.15}
    NEW_W = {"consistency": 0.15, "retrieval": 0.20, "critic": 0.55, "entity": 0.10}
    RISK_HIGH = 0.32

    results = []
    for r in data:
        old_ret_risk = 1.0 - r['retrieval']
        old_nli_risk = 1.0 - r['nli_critic']
        old_ent_risk = r['entity_risk']
        known_risk_sum = (OLD_W['retrieval'] * old_ret_risk) + (OLD_W['critic'] * old_nli_risk) + (OLD_W['entity'] * old_ent_risk)
        cons_risk = max(0.0, min(1.0, (r['risk_score'] - known_risk_sum) / OLD_W['consistency']))
        new_risk = (NEW_W['consistency'] * cons_risk) + (NEW_W['retrieval'] * old_ret_risk) + (NEW_W['critic'] * old_nli_risk) + (NEW_W['entity'] * old_ent_risk)
        
        pred_hall = new_risk >= RISK_HIGH
        is_hall = r['is_hallucinated']
        
        sample = EvalSample(question=r['question'], answer=r['answer'], is_hallucinated=is_hall)
        results.append(EvalResult(
            sample=sample, risk_score=new_risk, risk_flag="HIGH" if pred_hall else "LOW",
            retrieval=r['retrieval'], critic=r['nli_critic'], entity=r['entity_risk'],
            predicted_hallucinated=pred_hall, correct=(pred_hall == is_hall)
        ))

    all_metrics = {"Llama 3.1 (PubMedQA)": compute_all_metrics(results)}
    latex = generate_latex_table(all_metrics, mode="Hyperparameter Tuned")

    print("\n" + "=" * 60)
    print("  FINAL SYSTEM HEALTH TABLE (LaTeX)")
    print("=" * 60)
    print(latex)

    with open("system_health.tex", "w") as f: f.write(latex)
    with open("system_health.json", "w") as f: json.dump(all_metrics, f, indent=2)
    print("\n[OK] LaTeX and JSON saved.")

if __name__ == "__main__":
    main()