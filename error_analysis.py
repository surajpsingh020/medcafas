"""
error_analysis.py -- Detailed error analysis for MedCAFAS
=========================================================
Categorises every prediction error to understand WHY the system fails.
Uses pre-computed Llama 3.1 JSON results.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix
from eval import EvalSample, EvalResult

def guess_medical_domain(question: str) -> str:
    q = question.lower()
    domain_keywords = {
        "Pharmacology":   ["drug", "medication", "dose", "prescrib", "contraindic", "side effect", "adverse", "pharmacol", "mg", "treatment", "therapy", "antibiotic", "analgesic", "antihypertens"],
        "Cardiology":     ["heart", "cardiac", "arrhythm", "myocard", "atrial", "ventricl", "coronary", "ecg", "ekg", "hypertens", "angina", "valve", "murmur"],
        "Pathology":      ["biopsy", "histol", "cytol", "neoplasm", "tumor", "cancer", "malign", "benign", "carcinoma", "lymphoma", "metastas"],
        "Surgery":        ["surgical", "operation", "incision", "resect", "laparoscop", "anastomosis", "cholecystectomy", "appendectomy"],
        "Neurology":      ["brain", "neural", "stroke", "seizure", "epilep", "dementia", "alzheimer", "parkinson", "neuropath", "multiple sclerosis", "headache", "migraine"],
        "Infectious":     ["infect", "bacteria", "virus", "fungal", "parasit", "antibiotic", "hiv", "tuberculosis", "hepatitis", "pneumonia", "sepsis"],
        "Endocrinology":  ["diabetes", "thyroid", "insulin", "glucose", "adrenal", "pituitary", "hormone", "cortisol", "hba1c", "metabol"],
        "Gastroenterology": ["liver", "hepat", "gastric", "intestin", "colon", "pancrea", "gerd", "ibs", "ibd", "cirrhosis", "gallbladder"],
        "Pulmonology":    ["lung", "pulmonary", "respiratory", "asthma", "copd", "pneumonia", "bronch", "pleural", "oxygen"],
        "Nephrology":     ["kidney", "renal", "dialysis", "creatinine", "glomerular", "nephr", "proteinuria", "ckd"],
        "Psychiatry":     ["depression", "anxiety", "schizophren", "bipolar", "psychiat", "ssri", "antidepressant", "psychosis", "suicide", "mental health"],
    }
    for domain, keywords in domain_keywords.items():
        if any(kw in q for kw in keywords): return domain
    return "Other"

def categorise_error(r: EvalResult) -> str:
    if r.sample.is_hallucinated and not r.predicted_hallucinated:
        if r.retrieval > 0.75: return "FN-Ret"      
        if r.critic > 0.50:    return "FN-NLI"      
        if r.entity < 0.10:    return "FN-Entity"   
        return "FN-Combo"
    else:
        if r.critic < 0.20:    return "FP-NLI"      
        if r.retrieval < 0.60: return "FP-Ret"      
        if r.entity > 0.30:    return "FP-Entity"   
        return "FP-Combo"

def explain_error(r: EvalResult, category: str) -> str:
    explanations = {
        "FP-NLI":    f"NLI entailment too low ({r.critic:.2f}) despite correct answer. Likely paraphrase mismatch.",
        "FP-Ret":    f"Retrieval score too low ({r.retrieval:.2f}). KB may lack coverage.",
        "FP-Entity": f"Entity check flagged {r.entity:.0%} terms as missing. Too aggressive.",
        "FP-Combo":  f"Multiple signals compounded: ret={r.retrieval:.2f}, nli={r.critic:.2f}, ent={r.entity:.2f}.",
        "FN-Ret":    f"High retrieval score ({r.retrieval:.2f}) for hallucinated answer. Superficial match.",
        "FN-NLI":    f"NLI gave high entailment ({r.critic:.2f}) to a fabricated claim.",
        "FN-Entity": f"Entity check missed fabricated terms (risk={r.entity:.2f}).",
        "FN-Combo":  f"All layers failed: ret={r.retrieval:.2f}, nli={r.critic:.2f}, ent={r.entity:.2f}.",
    }
    return explanations.get(category, "Unknown")

class ErrorInfo:
    def __init__(self, sample, result, error_type, category, domain, reason):
        self.sample = sample
        self.result = result
        self.error_type = error_type
        self.category = category
        self.domain = domain
        self.reason = reason

def main():
    print("Loading real LLM answers from data/eval_llm_final.json...")
    with open("data/eval_llm_final.json", "r") as f: data = json.load(f)

    # Apply our tuned weights and thresholds
    OLD_W = {"consistency": 0.25, "retrieval": 0.30, "critic": 0.30, "entity": 0.15}
    NEW_W = {"consistency": 0.15, "retrieval": 0.20, "critic": 0.55, "entity": 0.10}
    RISK_HIGH = 0.32

    results = []
    for r in data:
        old_ret_risk = 1.0 - r['retrieval']
        old_nli_risk = 1.0 - r['nli_critic']
        old_ent_risk = r['entity_risk']
        known_risk_sum = (OLD_W['retrieval'] * old_ret_risk) + (OLD_W['critic'] * old_nli_risk) + (OLD_W['entity'] * old_ent_risk)
        cons_risk = (r['risk_score'] - known_risk_sum) / OLD_W['consistency']
        cons_risk = max(0.0, min(1.0, cons_risk))
        
        new_risk = (NEW_W['consistency'] * cons_risk) + (NEW_W['retrieval'] * old_ret_risk) + (NEW_W['critic'] * old_nli_risk) + (NEW_W['entity'] * old_ent_risk)
        
        pred_hall = new_risk >= RISK_HIGH
        is_hall = r['is_hallucinated']
        
        sample = EvalSample(question=r['question'], answer=r['answer'], is_hallucinated=is_hall)
        result = EvalResult(
            sample=sample, risk_score=new_risk, risk_flag="HIGH" if pred_hall else "LOW",
            retrieval=r['retrieval'], critic=r['nli_critic'], entity=r['entity_risk'],
            predicted_hallucinated=pred_hall, correct=(pred_hall == is_hall)
        )
        results.append(result)

    y_true = [int(r.sample.is_hallucinated) for r in results]
    y_pred = [int(r.predicted_hallucinated) for r in results]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    correct = [r for r in results if r.correct]
    errors  = [r for r in results if not r.correct]
    fp = [r for r in errors if not r.sample.is_hallucinated and r.predicted_hallucinated]
    fn = [r for r in errors if r.sample.is_hallucinated and not r.predicted_hallucinated]

    error_infos = []
    category_counts = Counter()
    domain_errors   = defaultdict(lambda: {"FP": 0, "FN": 0, "total": 0})
    domain_total    = defaultdict(int)

    for r in results:
        domain = guess_medical_domain(r.sample.question)
        domain_total[domain] += 1
        if not r.correct:
            cat = categorise_error(r)
            err_type = "FP" if not r.sample.is_hallucinated else "FN"
            reason = explain_error(r, cat)
            error_infos.append(ErrorInfo(sample=r.sample, result=r, error_type=err_type, category=cat, domain=domain, reason=reason))
            category_counts[cat] += 1
            domain_errors[domain][err_type] += 1
            domain_errors[domain]["total"] += 1

    ds_name = "Llama 3.1 (PubMedQA)"
    _generate_plots(ds_name, results, error_infos, category_counts, domain_errors, domain_total, cm)

    out_json = "error_analysis_llama.json"
    export = {
        "dataset": ds_name, "accuracy": acc, "n_errors": len(errors), "n_fp": len(fp), "n_fn": len(fn),
        "category_counts": dict(category_counts),
        "errors": [{"error_type": e.error_type, "category": e.category, "domain": e.domain, "reason": e.reason, "question": e.sample.question[:100], "risk_score": e.result.risk_score} for e in error_infos]
    }
    with open(out_json, "w") as f: json.dump(export, f, indent=2)
    print(f"\n[OK] Error analysis saved to {out_json}")


def _generate_plots(ds_name, results, error_infos, category_counts, domain_errors, domain_total, cm):
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Not Hall", "Hall"])
    ax1.set_yticklabels(["Not Hall", "Hall"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=16, fontweight="bold", color="white" if cm[i][j] > cm.max()/2 else "black")

    # 2. Error category pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    if category_counts:
        cats, vals = list(category_counts.keys()), list(category_counts.values())
        ax2.pie(vals, labels=cats, autopct="%1.0f%%", colors=plt.cm.Set3(np.linspace(0, 1, len(cats))), textprops={"fontsize": 9})
    ax2.set_title("Error Categories", fontweight="bold")

    # 3. Score distributions
    ax3 = fig.add_subplot(gs[0, 2])
    correct_scores = [r.risk_score for r in results if r.correct]
    fp_scores = [r.risk_score for r in results if not r.correct and not r.sample.is_hallucinated]
    fn_scores = [r.risk_score for r in results if not r.correct and r.sample.is_hallucinated]
    bins = np.linspace(0, 1, 20)
    if correct_scores: ax3.hist(correct_scores, bins=bins, alpha=0.5, label=f"Correct", color="#16a34a")
    if fp_scores: ax3.hist(fp_scores, bins=bins, alpha=0.7, label=f"FP", color="#dc2626")
    if fn_scores: ax3.hist(fn_scores, bins=bins, alpha=0.7, label=f"FN", color="#f59e0b")
    ax3.axvline(0.32, color="red", linestyle="--", label="HIGH=0.32")
    ax3.set_xlabel("Risk Score")
    ax3.set_ylabel("Count")
    ax3.set_title("Score Distributions", fontweight="bold")
    ax3.legend(fontsize=8)

    # 4. Retrieval vs NLI scatter
    ax4 = fig.add_subplot(gs[1, 0])
    for label, color, marker, filter_fn in [("Correct", "#16a34a", "o", lambda r: r.correct), ("FP", "#dc2626", "x", lambda r: not r.correct and not r.sample.is_hallucinated), ("FN", "#f59e0b", "^", lambda r: not r.correct and r.sample.is_hallucinated)]:
        subset = [r for r in results if filter_fn(r)]
        if subset: ax4.scatter([r.retrieval for r in subset], [r.critic for r in subset], c=color, marker=marker, alpha=0.6, label=label, s=30)
    ax4.set_xlabel("Retrieval Score")
    ax4.set_ylabel("NLI Critic Score")
    ax4.set_title("Retrieval vs NLI", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # 5. Per-domain accuracy
    ax5 = fig.add_subplot(gs[1, 1:])
    domains = sorted(domain_total.keys())
    domain_accs = [(domain_total[d] - domain_errors[d]["total"]) / max(1, domain_total[d]) for d in domains]
    if domains:
        y_pos = np.arange(len(domains))
        bars = ax5.barh(y_pos, domain_accs, color="#2563eb", alpha=0.8)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([f"{d} (n={domain_total[d]})" for d in domains], fontsize=9)
        ax5.set_xlabel("Accuracy")
        ax5.set_title("Per-Domain Accuracy", fontweight="bold")
        ax5.set_xlim(0, 1.05)
        ax5.grid(axis="x", alpha=0.3)
        ax5.invert_yaxis()
        for bar, acc in zip(bars, domain_accs):
            ax5.annotate(f"{acc:.0%}", xy=(bar.get_width(), bar.get_y() + bar.get_height()/2), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=9)

    # 6. Layer contribution
    ax6 = fig.add_subplot(gs[2, :])
    if error_infos:
        fp_errors = [e for e in error_infos if e.error_type == "FP"]
        fn_errors = [e for e in error_infos if e.error_type == "FN"]
        categories = ["Retrieval Risk", "NLI Risk", "Entity Risk"]
        fp_means = [np.mean([1-e.result.retrieval for e in fp_errors]), np.mean([1-e.result.critic for e in fp_errors]), np.mean([e.result.entity for e in fp_errors])] if fp_errors else [0,0,0]
        fn_means = [np.mean([1-e.result.retrieval for e in fn_errors]), np.mean([1-e.result.critic for e in fn_errors]), np.mean([e.result.entity for e in fn_errors])] if fn_errors else [0,0,0]
        x = np.arange(len(categories))
        w = 0.35
        ax6.bar(x - w/2, fp_means, w, label=f"False Positives", color="#dc2626", alpha=0.8)
        ax6.bar(x + w/2, fn_means, w, label=f"False Negatives", color="#f59e0b", alpha=0.8)
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories, fontsize=11)
        ax6.set_ylabel("Mean Risk Contribution")
        ax6.set_title("Layer Risk Scores for Error Types", fontweight="bold")
        ax6.legend(fontsize=10)
        ax6.grid(axis="y", alpha=0.3)
        ax6.set_ylim(0, 1.0)

    fig.suptitle(f"MedCAFAS Error Analysis - {ds_name}", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = "error_analysis_llama.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] Error analysis plots saved to {out}")
    plt.close()

if __name__ == "__main__":
    main()