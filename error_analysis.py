"""
error_analysis.py -- Detailed error analysis for MedCAFAS
=========================================================

Categorises every prediction error to understand WHY the system fails.

Error categories:
  False Positives (FP) - correct answer flagged as hallucinated
    - FP-NLI:    NLI failed to detect entailment (paraphrase problem)
    - FP-Ret:    Low retrieval score despite correct content
    - FP-Entity: Entity check too aggressive
    - FP-Combo:  Multiple signals combined to push score over threshold

  False Negatives (FN) - hallucination missed
    - FN-NLI:    NLI falsely gave high entailment to fabricated claim
    - FN-Ret:    High retrieval score for wrong answer (similar topics)
    - FN-Entity: Entity check missed fabricated terms
    - FN-Combo:  Multiple layers failed to flag

Also produces:
  - Per-medical-domain breakdown (pharmacology, pathology, etc.)
  - Score distribution histograms
  - Confusion matrix heatmap

Usage:
    python error_analysis.py                         # PubMedQA, 100 samples
    python error_analysis.py --dataset all --samples 80
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
    _get_embedder,
    _get_kb,
    _get_nli_model,
    _get_bm25,
)
from sklearn.metrics import accuracy_score, confusion_matrix


# -------------------------------------------------------------------------- #
#  Error categorisation                                                       #
# -------------------------------------------------------------------------- #

@dataclass
class ErrorInfo:
    """Detailed info about a single prediction error."""
    sample:    EvalSample
    result:    EvalResult
    error_type: str    # "FP" or "FN"
    category:  str     # "FP-NLI", "FP-Ret", etc.
    domain:    str     # medical domain guess
    reason:    str     # human-readable explanation


def categorise_error(r: EvalResult) -> str:
    """Determine the primary cause of a prediction error."""
    if r.sample.is_hallucinated and not r.predicted_hallucinated:
        # False Negative: missed a real hallucination
        if r.retrieval > 0.75:
            return "FN-Ret"      # high retrieval tricked the system
        if r.critic > 0.50:
            return "FN-NLI"      # NLI said entailed when it shouldn't be
        if r.entity < 0.10:
            return "FN-Entity"   # entity check didn't catch fabricated terms
        return "FN-Combo"
    else:
        # False Positive: flagged a correct answer
        if r.critic < 0.20:
            return "FP-NLI"      # NLI couldn't detect entailment (paraphrase)
        if r.retrieval < 0.60:
            return "FP-Ret"      # retrieval couldn't find matching evidence
        if r.entity > 0.30:
            return "FP-Entity"   # entity check was too aggressive
        return "FP-Combo"


def guess_medical_domain(question: str) -> str:
    """Heuristic domain classification based on question keywords."""
    q = question.lower()
    domain_keywords = {
        "Pharmacology":   ["drug", "medication", "dose", "prescrib", "contraindic",
                           "side effect", "adverse", "pharmacol", "mg", "treatment",
                           "therapy", "antibiotic", "analgesic", "antihypertens"],
        "Cardiology":     ["heart", "cardiac", "arrhythm", "myocard", "atrial",
                           "ventricl", "coronary", "ecg", "ekg", "hypertens",
                           "angina", "valve", "murmur"],
        "Pathology":      ["biopsy", "histol", "cytol", "neoplasm", "tumor",
                           "cancer", "malign", "benign", "carcinoma", "lymphoma",
                           "metastas"],
        "Surgery":        ["surgical", "operation", "incision", "resect",
                           "laparoscop", "anastomosis", "cholecystectomy",
                           "appendectomy"],
        "Neurology":      ["brain", "neural", "stroke", "seizure", "epilep",
                           "dementia", "alzheimer", "parkinson", "neuropath",
                           "multiple sclerosis", "headache", "migraine"],
        "Infectious":     ["infect", "bacteria", "virus", "fungal", "parasit",
                           "antibiotic", "hiv", "tuberculosis", "hepatitis",
                           "pneumonia", "sepsis"],
        "Endocrinology":  ["diabetes", "thyroid", "insulin", "glucose",
                           "adrenal", "pituitary", "hormone", "cortisol",
                           "hba1c", "metabol"],
        "Gastroenterology": ["liver", "hepat", "gastric", "intestin", "colon",
                              "pancrea", "gerd", "ibs", "ibd", "cirrhosis",
                              "gallbladder"],
        "Pulmonology":    ["lung", "pulmonary", "respiratory", "asthma",
                           "copd", "pneumonia", "bronch", "pleural", "oxygen"],
        "Nephrology":     ["kidney", "renal", "dialysis", "creatinine",
                           "glomerular", "nephr", "proteinuria", "ckd"],
        "Psychiatry":     ["depression", "anxiety", "schizophren", "bipolar",
                           "psychiat", "ssri", "antidepressant", "psychosis",
                           "suicide", "mental health"],
    }

    for domain, keywords in domain_keywords.items():
        if any(kw in q for kw in keywords):
            return domain
    return "Other"


def explain_error(r: EvalResult, category: str) -> str:
    """Generate a human-readable explanation of why this error occurred."""
    explanations = {
        "FP-NLI":    f"NLI entailment too low ({r.critic:.2f}) despite correct answer. "
                     f"Likely paraphrase mismatch between LLM output and KB evidence.",
        "FP-Ret":    f"Retrieval score too low ({r.retrieval:.2f}). KB may lack coverage "
                     f"for this topic, or answer wording differs from KB passages.",
        "FP-Entity": f"Entity check flagged {r.entity:.0%} terms as missing from evidence. "
                     f"Terms may be valid but absent from KB vocabulary.",
        "FP-Combo":  f"Multiple signals compounded: ret={r.retrieval:.2f}, nli={r.critic:.2f}, "
                     f"entity={r.entity:.2f}. No single dominant failure.",
        "FN-Ret":    f"High retrieval score ({r.retrieval:.2f}) for hallucinated answer. "
                     f"KB contains similar-topic passages that superficially match.",
        "FN-NLI":    f"NLI gave high entailment ({r.critic:.2f}) to a fabricated claim. "
                     f"Perturbation may be too subtle for NLI to detect.",
        "FN-Entity": f"Entity check missed fabricated terms (entity_risk={r.entity:.2f}). "
                     f"Hallucination uses real medical terms in wrong context.",
        "FN-Combo":  f"All layers failed: ret={r.retrieval:.2f}, nli={r.critic:.2f}, "
                     f"entity={r.entity:.2f}. Hallucination closely mimics real content.",
    }
    return explanations.get(category, "Unknown error category")


# -------------------------------------------------------------------------- #
#  Main analysis                                                              #
# -------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="MedCAFAS Error Analysis")
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

    for ds_name, samples in datasets.items():
        print(f"\n{'='*70}")
        print(f"  Error Analysis: {ds_name} ({len(samples)} samples)")
        print(f"{'='*70}")

        # Run evaluation
        print("  Running eval_no_llm ...")
        t0 = time.time()
        results = [eval_no_llm(s) for s in samples]
        dt = time.time() - t0
        print(f"  Done in {dt:.0f}s.\n")

        # Basic metrics
        y_true = [int(r.sample.is_hallucinated) for r in results]
        y_pred = [int(r.predicted_hallucinated)  for r in results]
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        correct = [r for r in results if r.correct]
        errors  = [r for r in results if not r.correct]
        fp = [r for r in errors if not r.sample.is_hallucinated and r.predicted_hallucinated]
        fn = [r for r in errors if r.sample.is_hallucinated and not r.predicted_hallucinated]

        print(f"  Accuracy: {acc:.1%}")
        print(f"  Total errors: {len(errors)}/{len(results)}")
        print(f"    False Positives (correct flagged as HALL): {len(fp)}")
        print(f"    False Negatives (HALL missed):             {len(fn)}")

        # -- Categorise errors -----------------------------------------------
        error_infos: List[ErrorInfo] = []
        category_counts = Counter()
        domain_errors   = defaultdict(lambda: {"FP": 0, "FN": 0, "total": 0})
        domain_correct  = defaultdict(int)

        for r in results:
            domain = guess_medical_domain(r.sample.question)
            if r.correct:
                domain_correct[domain] += 1
            else:
                cat = categorise_error(r)
                err_type = "FP" if not r.sample.is_hallucinated else "FN"
                reason = explain_error(r, cat)
                error_infos.append(ErrorInfo(
                    sample=r.sample, result=r, error_type=err_type,
                    category=cat, domain=domain, reason=reason,
                ))
                category_counts[cat] += 1
                domain_errors[domain][err_type] += 1
                domain_errors[domain]["total"] += 1

        # Also count domains for correct ones
        for r in results:
            domain = guess_medical_domain(r.sample.question)
            # already counted above

        # -- Error category breakdown ----------------------------------------
        print(f"\n  Error Category Breakdown:")
        print(f"  {'Category':<14} {'Count':>6} {'%':>7}")
        print(f"  {'-'*14} {'-'*6} {'-'*7}")
        for cat, count in sorted(category_counts.items()):
            pct = count / max(1, len(errors)) * 100
            print(f"  {cat:<14} {count:>6} {pct:>6.1f}%")

        # -- Domain breakdown ------------------------------------------------
        print(f"\n  Per-Domain Accuracy:")
        print(f"  {'Domain':<20} {'Correct':>8} {'FP':>5} {'FN':>5} {'Total':>6} {'Acc':>7}")
        print(f"  {'-'*20} {'-'*8} {'-'*5} {'-'*5} {'-'*6} {'-'*7}")

        domain_total = defaultdict(int)
        for r in results:
            domain = guess_medical_domain(r.sample.question)
            domain_total[domain] += 1

        for domain in sorted(domain_total.keys()):
            total = domain_total[domain]
            fp_d = domain_errors[domain]["FP"]
            fn_d = domain_errors[domain]["FN"]
            correct_d = total - fp_d - fn_d
            acc_d = correct_d / max(1, total)
            print(f"  {domain:<20} {correct_d:>8} {fp_d:>5} {fn_d:>5} {total:>6} {acc_d:>6.1%}")

        # -- Score distribution analysis -------------------------------------
        print(f"\n  Score Distribution (mean +/- std):")
        print(f"  {'Metric':<20} {'Correct':>16} {'FP':>16} {'FN':>16}")
        print(f"  {'-'*20} {'-'*16} {'-'*16} {'-'*16}")

        correct_res = [r for r in results if r.correct]
        for metric_name, getter in [("risk_score", lambda r: r.risk_score),
                                     ("retrieval",  lambda r: r.retrieval),
                                     ("nli_critic",  lambda r: r.critic),
                                     ("entity_risk", lambda r: r.entity)]:
            c_vals = [getter(r) for r in correct_res]
            fp_vals = [getter(r) for r in fp] if fp else [0]
            fn_vals = [getter(r) for r in fn] if fn else [0]
            print(f"  {metric_name:<20} "
                  f"{np.mean(c_vals):>6.3f} +/- {np.std(c_vals):.3f} "
                  f"{np.mean(fp_vals):>6.3f} +/- {np.std(fp_vals):.3f} "
                  f"{np.mean(fn_vals):>6.3f} +/- {np.std(fn_vals):.3f}")

        # -- Top 5 worst FP and FN -------------------------------------------
        print(f"\n  Top 5 False Positives (worst FP - correct answers flagged):")
        fps_sorted = sorted(fp, key=lambda r: r.risk_score, reverse=True)[:5]
        for i, r in enumerate(fps_sorted, 1):
            cat = categorise_error(r)
            print(f"  [{i}] score={r.risk_score:.3f} ret={r.retrieval:.2f} "
                  f"nli={r.critic:.2f} ent={r.entity:.2f} [{cat}]")
            print(f"      Q: {r.sample.question[:80]}")
            print(f"      A: {r.sample.answer[:80]}")

        print(f"\n  Top 5 False Negatives (worst FN - hallucinations missed):")
        fns_sorted = sorted(fn, key=lambda r: r.risk_score)[:5]
        for i, r in enumerate(fns_sorted, 1):
            cat = categorise_error(r)
            print(f"  [{i}] score={r.risk_score:.3f} ret={r.retrieval:.2f} "
                  f"nli={r.critic:.2f} ent={r.entity:.2f} [{cat}]")
            print(f"      Q: {r.sample.question[:80]}")
            print(f"      A: {r.sample.answer[:80]}")

        # -- Generate visualizations ----------------------------------------
        _generate_plots(ds_name, results, error_infos, category_counts,
                        domain_errors, domain_total, cm)

        # -- Save detailed JSON ---------------------------------------------
        out_json = f"error_analysis_{ds_name.lower().replace('-', '_')}.json"
        export = {
            "dataset": ds_name,
            "n_samples": len(results),
            "accuracy": acc,
            "n_errors": len(errors),
            "n_fp": len(fp),
            "n_fn": len(fn),
            "category_counts": dict(category_counts),
            "errors": [
                {
                    "error_type": e.error_type,
                    "category": e.category,
                    "domain": e.domain,
                    "reason": e.reason,
                    "question": e.sample.question[:200],
                    "answer": e.sample.answer[:200],
                    "risk_score": e.result.risk_score,
                    "retrieval": e.result.retrieval,
                    "critic": e.result.critic,
                    "entity": e.result.entity,
                }
                for e in error_infos
            ],
        }
        with open(out_json, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\n  [OK] Error analysis saved to {out_json}")

    print("\n  Done!")


def _generate_plots(ds_name, results, error_infos, category_counts,
                    domain_errors, domain_total, cm):
    """Generate comprehensive error analysis visualizations."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Confusion Matrix (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Not Hall", "Hall"])
    ax1.set_yticklabels(["Not Hall", "Hall"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i][j]), ha="center", va="center",
                     fontsize=16, fontweight="bold",
                     color="white" if cm[i][j] > cm.max()/2 else "black")

    # 2. Error category pie chart (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if category_counts:
        cats = list(category_counts.keys())
        vals = list(category_counts.values())
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(cats)))
        ax2.pie(vals, labels=cats, autopct="%1.0f%%", colors=colors_pie,
                textprops={"fontsize": 9})
    ax2.set_title("Error Categories", fontweight="bold")

    # 3. Score distributions (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    correct_scores = [r.risk_score for r in results if r.correct]
    fp_scores = [r.risk_score for r in results if not r.correct and not r.sample.is_hallucinated]
    fn_scores = [r.risk_score for r in results if not r.correct and r.sample.is_hallucinated]
    bins = np.linspace(0, 1, 20)
    if correct_scores:
        ax3.hist(correct_scores, bins=bins, alpha=0.5, label=f"Correct ({len(correct_scores)})", color="#16a34a")
    if fp_scores:
        ax3.hist(fp_scores, bins=bins, alpha=0.7, label=f"FP ({len(fp_scores)})", color="#dc2626")
    if fn_scores:
        ax3.hist(fn_scores, bins=bins, alpha=0.7, label=f"FN ({len(fn_scores)})", color="#f59e0b")
    ax3.axvline(config.RISK_LOW, color="orange", linestyle="--", label=f"LOW={config.RISK_LOW}")
    ax3.axvline(config.RISK_HIGH, color="red", linestyle="--", label=f"HIGH={config.RISK_HIGH}")
    ax3.set_xlabel("Risk Score")
    ax3.set_ylabel("Count")
    ax3.set_title("Score Distributions", fontweight="bold")
    ax3.legend(fontsize=8)

    # 4. Retrieval vs NLI scatter (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    for label, color, marker, filter_fn in [
        ("Correct",  "#16a34a", "o", lambda r: r.correct),
        ("FP",       "#dc2626", "x", lambda r: not r.correct and not r.sample.is_hallucinated),
        ("FN",       "#f59e0b", "^", lambda r: not r.correct and r.sample.is_hallucinated),
    ]:
        subset = [r for r in results if filter_fn(r)]
        if subset:
            ax4.scatter([r.retrieval for r in subset], [r.critic for r in subset],
                        c=color, marker=marker, alpha=0.6, label=label, s=30)
    ax4.set_xlabel("Retrieval Score")
    ax4.set_ylabel("NLI Critic Score")
    ax4.set_title("Retrieval vs NLI", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # 5. Per-domain accuracy (middle-center + right)
    ax5 = fig.add_subplot(gs[1, 1:])
    domains = sorted(domain_total.keys())
    domain_accs = []
    domain_ns   = []
    for d in domains:
        t = domain_total[d]
        e = domain_errors[d]["total"]
        domain_accs.append((t - e) / max(1, t))
        domain_ns.append(t)

    if domains:
        y_pos = np.arange(len(domains))
        bars = ax5.barh(y_pos, domain_accs, color="#2563eb", alpha=0.8)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([f"{d} (n={n})" for d, n in zip(domains, domain_ns)], fontsize=9)
        ax5.set_xlabel("Accuracy")
        ax5.set_title("Per-Domain Accuracy", fontweight="bold")
        ax5.set_xlim(0, 1.05)
        ax5.grid(axis="x", alpha=0.3)
        ax5.invert_yaxis()
        for bar, acc in zip(bars, domain_accs):
            ax5.annotate(f"{acc:.0%}", xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                         xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=9)

    # 6. Layer contribution to errors (bottom)
    ax6 = fig.add_subplot(gs[2, :])
    if error_infos:
        fp_errors = [e for e in error_infos if e.error_type == "FP"]
        fn_errors = [e for e in error_infos if e.error_type == "FN"]

        fp_ret = [e.result.retrieval for e in fp_errors] or [0]
        fp_nli = [e.result.critic for e in fp_errors] or [0]
        fp_ent = [e.result.entity for e in fp_errors] or [0]
        fn_ret = [e.result.retrieval for e in fn_errors] or [0]
        fn_nli = [e.result.critic for e in fn_errors] or [0]
        fn_ent = [e.result.entity for e in fn_errors] or [0]

        categories = ["Retrieval", "NLI Critic", "Entity Risk"]
        fp_means = [np.mean(fp_ret), np.mean(fp_nli), np.mean(fp_ent)]
        fn_means = [np.mean(fn_ret), np.mean(fn_nli), np.mean(fn_ent)]

        x = np.arange(len(categories))
        w = 0.35
        ax6.bar(x - w/2, fp_means, w, label=f"False Positives (n={len(fp_errors)})", color="#dc2626", alpha=0.8)
        ax6.bar(x + w/2, fn_means, w, label=f"False Negatives (n={len(fn_errors)})", color="#f59e0b", alpha=0.8)
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories, fontsize=11)
        ax6.set_ylabel("Mean Score")
        ax6.set_title("Layer Scores for Error Types", fontweight="bold")
        ax6.legend(fontsize=10)
        ax6.grid(axis="y", alpha=0.3)
        ax6.set_ylim(0, 1.0)

    fig.suptitle(f"MedCAFAS Error Analysis - {ds_name}", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = f"error_analysis_{ds_name.lower().replace('-', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] Error analysis plots saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
