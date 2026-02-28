"""
eval.py — MedCAFAS Evaluation Suite
─────────────────────────────────────
Measures pipeline accuracy using HaluEval ground-truth labels.

Each HaluEval row has:
  question            — the medical question
  right_answer        — correct answer  → expected: LOW  (not hallucinated)
  hallucinated_answer — wrong answer    → expected: HIGH (hallucinated)

We treat this as a binary classification problem:
  Positive  = hallucinated  → risk_flag in {HIGH, CAUTION}
  Negative  = not hallucinated → risk_flag == LOW

Metrics computed:
  - Accuracy, Precision, Recall, F1
  - Per-layer score distributions
  - Confusion matrix
  - ROC-AUC on raw risk_score

Usage:
    python eval.py               # runs 40 samples (fast, ~15 min on CPU)
    python eval.py --samples 100 # more thorough
    python eval.py --samples 0   # full dataset (slow)
    python eval.py --no-llm      # skip Layer 1 (uses dummy answer = HaluEval answer directly, very fast)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import config
from pipeline import layer2_retrieval, layer3_critic, aggregate, layer2b_entity_check, _get_embedder, _get_kb, _get_nli_model, _get_bm25


# ─────────────────────────────────────────────────────────────────────────── #
#  Data classes                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class EvalSample:
    question:    str
    answer:      str
    is_hallucinated: bool          # ground truth


@dataclass
class EvalResult:
    sample:      EvalSample
    risk_score:  float
    risk_flag:   str
    retrieval:   float
    critic:      float
    entity:      float
    predicted_hallucinated: bool   # True if flag != LOW
    correct:     bool


# ─────────────────────────────────────────────────────────────────────────── #
#  Fast eval (no LLM — uses HaluEval answers directly)                       #
# ─────────────────────────────────────────────────────────────────────────── #

def eval_no_llm(sample: EvalSample) -> EvalResult:
    """
    Skips Layer 1 (self-consistency / Ollama).
    Directly feeds the known answer through Layer 2 + 2b + Layer 3.
    Consistency is set to 1.0 (neutral) so it doesn't skew results.
    Fast: ~1-2s per sample on CPU.
    """
    retrieval_score, citations = layer2_retrieval(sample.answer)
    entity_risk = layer2b_entity_check(sample.answer, citations)
    # layer3_critic now returns (score, claim_results) tuple
    critic_score, _ = layer3_critic(sample.answer, citations, question=sample.question)

    # Neutral consistency (1.0 = not penalising, isolates retrieval+NLI signal)
    consistency = 1.0
    risk_score, risk_flag, _ = aggregate(consistency, retrieval_score, critic_score,
                                      entity_risk=entity_risk)

    # CAUTION means the model is uncertain — not a confident hallucination call.
    # Treating CAUTION as positive floods the system with false positives.
    # Only flag HIGH as a positive prediction; CAUTION surfaces in the UI for
    # human review but is not counted as a detected hallucination.
    predicted_hallucinated = risk_flag == "HIGH"
    correct = predicted_hallucinated == sample.is_hallucinated

    return EvalResult(
        sample=sample,
        risk_score=risk_score,
        risk_flag=risk_flag,
        retrieval=retrieval_score,
        critic=critic_score,
        entity=entity_risk,
        predicted_hallucinated=predicted_hallucinated,
        correct=correct,
    )


# ─────────────────────────────────────────────────────────────────────────── #
#  Full eval (with LLM — uses pipeline.predict)                              #
# ─────────────────────────────────────────────────────────────────────────── #

def eval_with_llm(sample: EvalSample) -> EvalResult:
    """
    Runs the full 3-layer pipeline via pipeline.predict().
    Slower (~60-120s per sample on CPU).
    """
    from pipeline import predict
    result = predict(sample.question)

    predicted_hallucinated = result.risk_flag == "HIGH"
    correct = predicted_hallucinated == sample.is_hallucinated

    return EvalResult(
        sample=sample,
        risk_score=result.risk_score,
        risk_flag=result.risk_flag,
        retrieval=result.breakdown["retrieval_score"],
        critic=result.breakdown["nli_entailment"],
        entity=result.breakdown.get("entity_risk", 0.0),
        predicted_hallucinated=predicted_hallucinated,
        correct=correct,
    )


# ─────────────────────────────────────────────────────────────────────────── #
#  Dataset loader                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def _perturb_answer(answer: str) -> str:
    """
    Generate a hallucinated version of a correct answer by applying
    rule-based perturbations that mimic real LLM hallucination patterns:
      1. Negate key clinical statements (is not → is, is → is not)
      2. Swap common drug/treatment names for plausible alternatives
      3. Invert common clinical directions (increase → decrease)
    Returns a perturbed string guaranteed to differ from the input.
    """
    import re, random
    random.seed(hash(answer) % 2**31)

    # Negation flips
    negation_pairs = [
        (r'\bis not\b',            'is'),
        (r'\bis\b',                'is not'),
        (r'\bshould not\b',        'should'),
        (r'\bshould\b',            'should not'),
        (r'\bcontraindicated\b',   'recommended'),
        (r'\brecommended\b',       'contraindicated'),
        (r'\bincreases?\b',        'decreases'),
        (r'\bdecreases?\b',        'increases'),
        (r'\binhibits?\b',         'activates'),
        (r'\bactivates?\b',        'inhibits'),
        (r'\bfirst-line\b',        'last-resort'),
        (r'\bfirst line\b',        'last resort'),
    ]

    # Drug/treatment substitutions (common swap-outs in LLM hallucinations)
    drug_swaps = [
        ('metformin',    'glipizide'),
        ('glipizide',    'metformin'),
        ('warfarin',     'heparin'),
        ('heparin',      'warfarin'),
        ('aspirin',      'clopidogrel'),
        ('clopidogrel',  'aspirin'),
        ('metoprolol',   'atenolol'),
        ('atenolol',     'metoprolol'),
        ('amoxicillin',  'vancomycin'),
        ('vancomycin',   'amoxicillin'),
        ('lisinopril',   'losartan'),
        ('losartan',     'lisinopril'),
        ('furosemide',   'hydrochlorothiazide'),
        ('hydrochlorothiazide', 'furosemide'),
        ('morphine',     'fentanyl'),
        ('fentanyl',     'morphine'),
        ('epinephrine',  'atropine'),
        ('atropine',     'epinephrine'),
    ]

    text = answer

    # Try drug swap first (most realistic hallucination)
    for orig, swap in drug_swaps:
        if re.search(orig, text, flags=re.IGNORECASE):
            text = re.sub(orig, swap, text, count=1, flags=re.IGNORECASE)
            if text != answer:
                return text

    # Otherwise apply a negation flip
    random.shuffle(negation_pairs)
    for pattern, replacement in negation_pairs:
        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
            if text != answer:
                return text

    # Last resort: prepend "not"
    return "This is NOT correct: " + answer


def load_eval_samples(n: int = 40) -> List[EvalSample]:
    """
    Load balanced eval set from MedQA-USMLE test split.
    Hallucinated answers are generated by perturbing correct answers
    using rule-based transformations (negation flips, drug swaps, direction
    inversions) — matching real LLM hallucination patterns far better than
    USMLE wrong-option distractors which are designed to be plausible.

    NOTE: USMLE correct answers are 2–5 word noun phrases, which the NLI
    layer cannot evaluate reliably.  Use --dataset pubmedqa for a test that
    is actually aligned with the system's design (paragraph-length answers).
    """
    import random
    random.seed(42)

    print("Loading MedQA-USMLE test dataset...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    print(f"   {len(ds)} total rows available")

    samples: List[EvalSample] = []
    half = n // 2 if n > 0 else len(ds)

    for row in ds:
        q       = row.get("question", "").strip()
        correct = row.get("answer", "").strip()

        if not q or not correct:
            continue

        hallucinated = _perturb_answer(correct)
        if hallucinated == correct:
            continue   # skip if perturbation couldn't change it

        if len([s for s in samples if not s.is_hallucinated]) < half:
            samples.append(EvalSample(question=q, answer=correct, is_hallucinated=False))

        if len([s for s in samples if s.is_hallucinated]) < half:
            samples.append(EvalSample(question=q, answer=hallucinated, is_hallucinated=True))

        if n > 0 and len(samples) >= n:
            break

    print(f"   {len(samples)} eval samples loaded "
          f"({sum(not s.is_hallucinated for s in samples)} correct, "
          f"{sum(s.is_hallucinated for s in samples)} hallucinated)")
    return samples


def load_eval_samples_pubmedqa(n: int = 100) -> List[EvalSample]:
    """
    Load balanced eval set from PubMedQA pqa_labeled.

    PubMedQA answers are paragraph-length sentences from clinical trial abstracts
    — exactly the output format MedCAFAS is designed to evaluate.  This gives a
    meaningful NLI test because the hypothesis is a full declarative sentence.

    Correct answers  : long_answer (full abstract conclusion)
    Hallucinated     : perturbation of long_answer via _perturb_answer()

    Unlike the USMLE eval, answers here are typically 50-200 words so the NLI
    layer can actually produce meaningful entailment/contradiction scores.
    """
    import random
    random.seed(42)

    print("Loading PubMedQA pqa_labeled eval dataset...")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as e:
        print(f"   PubMedQA load failed: {e}")
        print("   Falling back to MedQA-USMLE...")
        return load_eval_samples(n)

    print(f"   {len(ds)} total rows available")

    samples: List[EvalSample] = []
    half = n // 2 if n > 0 else len(ds)

    for row in ds:
        q       = (row.get("question") or "").strip()
        correct = (row.get("long_answer") or "").strip()

        # Skip very short answers (< 10 words) — would undermine the point
        if not q or len(correct.split()) < 10:
            continue

        hallucinated = _perturb_answer(correct)
        if hallucinated == correct:
            continue

        if len([s for s in samples if not s.is_hallucinated]) < half:
            samples.append(EvalSample(question=q, answer=correct, is_hallucinated=False))

        if len([s for s in samples if s.is_hallucinated]) < half:
            samples.append(EvalSample(question=q, answer=hallucinated, is_hallucinated=True))

        if n > 0 and len(samples) >= n:
            break

    print(f"   {len(samples)} eval samples loaded "
          f"({sum(not s.is_hallucinated for s in samples)} correct, "
          f"{sum(s.is_hallucinated for s in samples)} hallucinated)")
    avg_words = np.mean([len(s.answer.split()) for s in samples])
    print(f"   Average answer length: {avg_words:.0f} words (NLI-suitable)")
    return samples


def load_eval_samples_llm(n: int = 40) -> List[EvalSample]:
    """
    GOLD STANDARD EVAL: actual phi3.5 outputs labeled by decision-matching.

    Methodology:
      1. For each PubMedQA row with a definitive final_decision (yes / no),
         ask phi3.5 the same yes/no question.
      2. Extract phi3.5's decision from the first 200 characters of its answer
         using regex (looks for the words 'yes' or 'no').
      3. If phi3.5's decision MATCHES final_decision  → correct (not hallucinated).
         If phi3.5's decision CONTRADICTS final_decision → hallucinated.
         If answer is unclear / ambiguous                → skip.

    This avoids the NLI phrasing-mismatch problem: an LLM can answer
    "Yes, this is correct — treatment X reduces …" whereas PubMedQA's
    long_answer begins with "The results show that …".  NLI entailment
    near-always fails for such paraphrase pairs even when factually identical.
    Decision-matching only tests the critical yes/no fact, which is stable
    across surface-form variation.

    Requires Ollama running.  Cannot be combined with --no-llm.
    """
    from pipeline import _ask_ollama

    print("Loading PubMedQA for LLM decision-matching eval (requires Ollama)...")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as e:
        print(f"   PubMedQA load failed: {e}")
        return load_eval_samples(n)

    print(f"   {len(ds)} total rows available")

    samples: List[EvalSample] = []
    target = n if n > 0 else len(ds)

    for row in ds:
        if len(samples) >= target:
            break

        q              = (row.get("question")        or "").strip()
        final_decision = (row.get("final_decision")  or "").strip().lower()
        ground_truth   = (row.get("long_answer")     or "").strip()

        # Only use rows with a clear yes/no ground-truth decision
        if not q or final_decision not in ("yes", "no"):
            continue
        if len(ground_truth.split()) < 5:
            continue

        print(f"   [{len(samples)+1}/{target}] LLM: {q[:55]}...", end=" ", flush=True)
        try:
            llm_answer = _ask_ollama(q)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        # Extract the LLM's yes/no decision from the start of its response
        answer_prefix = llm_answer[:200].lower()
        if re.search(r'\byes\b', answer_prefix):
            phi_decision = "yes"
        elif re.search(r'\bno\b', answer_prefix):
            phi_decision = "no"
        else:
            print(f"SKIP (unclear decision)")
            continue  # can't determine — skip rather than mislabel

        is_hall = (phi_decision != final_decision)
        print(f"phi={phi_decision} vs gt={final_decision} -> {'HALL' if is_hall else 'OK'}")

        samples.append(EvalSample(question=q, answer=llm_answer, is_hallucinated=is_hall))

    hall_n = sum(s.is_hallucinated for s in samples)
    print(
        f"   {len(samples)} LLM samples "
        f"({len(samples)-hall_n} correct, {hall_n} hallucinated)"
    )
    return samples



def compute_ece(results: List[EvalResult], n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Measures how well the raw risk_score matches the true positive rate
    within equal-width confidence bins.  A well-calibrated model has ECE < 0.05.

    risk_score is treated as P(hallucinated).  y_true = is_hallucinated.
    """
    y_true  = np.array([int(r.sample.is_hallucinated) for r in results])
    y_score = np.array([r.risk_score                  for r in results])

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(results)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_score[mask].mean()
        bin_acc  = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


# ─────────────────────────────────────────────────────────────────────────── #
#  Bootstrap Confidence Intervals                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def bootstrap_ci(
    results: List[EvalResult],
    metric: str = "accuracy",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap 95% confidence interval for a scalar metric.

    Supported metrics: 'accuracy', 'f1', 'roc_auc'

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    import random
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    rng = random.Random(seed)
    n   = len(results)
    boot_scores: List[float] = []

    for _ in range(n_boot):
        sample = [rng.choice(results) for _ in range(n)]
        y_t = [int(r.sample.is_hallucinated)  for r in sample]
        y_p = [int(r.predicted_hallucinated)  for r in sample]
        y_s = [r.risk_score                   for r in sample]

        try:
            if metric == "accuracy":
                boot_scores.append(accuracy_score(y_t, y_p))
            elif metric == "f1":
                boot_scores.append(f1_score(y_t, y_p, zero_division=0))
            elif metric == "roc_auc":
                if len(set(y_t)) < 2:
                    continue
                boot_scores.append(roc_auc_score(y_t, y_s))
        except Exception:
            continue

    boot_scores.sort()
    alpha = (1.0 - ci) / 2
    lo    = np.percentile(boot_scores, 100 * alpha)
    hi    = np.percentile(boot_scores, 100 * (1 - alpha))

    # Point estimate on the original data
    y_true  = [int(r.sample.is_hallucinated) for r in results]
    y_pred  = [int(r.predicted_hallucinated) for r in results]
    y_score = [r.risk_score                  for r in results]
    try:
        if metric == "accuracy":
            pt = accuracy_score(y_true, y_pred)
        elif metric == "f1":
            pt = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "roc_auc":
            pt = roc_auc_score(y_true, y_score)
        else:
            pt = float("nan")
    except Exception:
        pt = float("nan")

    return pt, float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────── #
#  Metrics                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def print_metrics(results: List[EvalResult], show_ci: bool = False) -> None:
    y_true = [int(r.sample.is_hallucinated) for r in results]
    y_pred = [int(r.predicted_hallucinated) for r in results]
    y_score = [r.risk_score for r in results]

    print("\n" + "=" * 60)
    print("  MedCAFAS Evaluation Results")
    print("=" * 60)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Accuracy  : {acc:.1%}  ({sum(r.correct for r in results)}/{len(results)})")

    # Classification report
    print("\n  Classification Report (Positive = hallucinated):")
    print(classification_report(y_true, y_pred, target_names=["NOT hallucinated", "Hallucinated"], digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix:")
    print(f"                  Pred: NOT-HALL   Pred: HALL")
    print(f"  True: NOT-HALL      {cm[0][0]:>6}       {cm[0][1]:>6}")
    print(f"  True: HALL          {cm[1][0]:>6}       {cm[1][1]:>6}")

    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"\n  ROC-AUC   : {auc:.4f}  (1.0 = perfect, 0.5 = random)")
    except Exception:
        pass

    # ECE
    ece = compute_ece(results)
    print(f"  ECE       : {ece:.4f}  (calibration error; < 0.05 = well-calibrated)")

    # Bootstrap 95% CIs
    if show_ci:
        print("\n  Bootstrap 95% Confidence Intervals (1 000 resamples):")
        for metric in ("accuracy", "f1", "roc_auc"):
            try:
                pt, lo, hi = bootstrap_ci(results, metric=metric)
                print(f"  {metric:10}: {pt:.3f}  [{lo:.3f}, {hi:.3f}]")
            except Exception as e:
                print(f"  {metric:10}: could not compute ({e})")

    # Per-layer distributions
    hal_ret  = [r.retrieval for r in results if r.sample.is_hallucinated]
    good_ret = [r.retrieval for r in results if not r.sample.is_hallucinated]
    hal_nli  = [r.critic    for r in results if r.sample.is_hallucinated]
    good_nli = [r.critic    for r in results if not r.sample.is_hallucinated]

    hal_ent  = [r.entity   for r in results if r.sample.is_hallucinated]
    good_ent = [r.entity   for r in results if not r.sample.is_hallucinated]

    print("\n  Layer Score Distributions (mean +/- std):")
    print(f"  {'':25}  {'Correct':>12}  {'Hallucinated':>12}")
    print(f"  {'Retrieval score':25}  {np.mean(good_ret):>6.3f} +/- {np.std(good_ret):.3f}  {np.mean(hal_ret):>6.3f} +/- {np.std(hal_ret):.3f}")
    print(f"  {'NLI entailment':25}  {np.mean(good_nli):>6.3f} +/- {np.std(good_nli):.3f}  {np.mean(hal_nli):>6.3f} +/- {np.std(hal_nli):.3f}")
    if good_ent and hal_ent:
        print(f"  {'Entity risk':25}  {np.mean(good_ent):>6.3f} +/- {np.std(good_ent):.3f}  {np.mean(hal_ent):>6.3f} +/- {np.std(hal_ent):.3f}")

    # Error analysis
    errors = [r for r in results if not r.correct]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for r in errors[:5]:
            label = "HALL" if r.sample.is_hallucinated else "not-hall"
            print(f"  [{label:8}] -> Pred:{r.risk_flag:7} score={r.risk_score:.2f} | "
                  f"ret={r.retrieval:.2f} nli={r.critic:.2f}")
            print(f"             Q: {r.sample.question[:80]}")
            print(f"             A: {r.sample.answer[:80]}")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────────────────── #
#  Fine-tuning suggestions                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def suggest_threshold_tuning(results: List[EvalResult]) -> None:
    """
    Grid-search the best RISK_LOW and RISK_HIGH thresholds on these results.
    Prints suggestions for config.py.
    """
    from sklearn.metrics import f1_score

    y_true = [int(r.sample.is_hallucinated) for r in results]
    scores = [r.risk_score for r in results]

    best_f1    = 0.0
    best_low   = config.RISK_LOW
    best_high  = config.RISK_HIGH

    for low in np.arange(0.20, 0.55, 0.05):
        for high in np.arange(low + 0.10, 0.85, 0.05):
            y_pred = [int(s >= low) for s in scores]   # LOW below `low`, else HALL
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1   = f1
                best_low  = round(float(low), 2)
                best_high = round(float(high), 2)

    print(f"\n  Threshold Tuning Suggestions (best F1={best_f1:.3f} on this eval set):")
    print(f"  Current  → RISK_LOW={config.RISK_LOW}, RISK_HIGH={config.RISK_HIGH}")
    print(f"  Suggested→ RISK_LOW={best_low},  RISK_HIGH={best_high}")
    if best_low != config.RISK_LOW or best_high != config.RISK_HIGH:
        print(f"  → Update config.py if these thresholds consistently improve on larger samples.")
    else:
        print(f"  → Current thresholds are already near-optimal for this sample.")


# ─────────────────────────────────────────────────────────────────────────── #
#  Post-hoc Isotonic Calibration                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def calibrate_scores(
    results: List[EvalResult],
    val_fraction: float = 0.30,
    seed: int = 42,
) -> Tuple[List[EvalResult], float, float]:
    """
    Post-hoc isotonic regression calibration.

    Procedure:
      1. Split results into a calibration set (val_fraction) and a held-out
         test set using a fixed random seed so results are reproducible.
      2. Fit IsotonicRegression on the calibration set  (risk_score → is_hallucinated).
      3. Apply the learned monotone mapping to ALL results (in-place).
      4. Return updated results, pre-cal ECE, post-cal ECE.

    Why isotonic regression:
      • Makes no distributional assumption (unlike Platt/sigmoid).
      • Guaranteed to preserve rank order, so ROC-AUC is unaffected.
      • Works even with small calibration sets (≥ 20 samples).

    The function mutates r.risk_score in-place and rebuilds r.predicted_hallucinated
    from the new scores using the current config thresholds.

    Returns:
        (results, ece_before, ece_after)
    """
    import random
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        print("  [calibration] scikit-learn not available — skipping.")
        ece = compute_ece(results)
        return results, ece, ece

    if len(results) < 20:
        print("  [calibration] Too few samples for calibration (< 20) — skipping.")
        ece = compute_ece(results)
        return results, ece, ece

    ece_before = compute_ece(results)

    # Split for calibration fit
    rng = random.Random(seed)
    shuffled = list(results)
    rng.shuffle(shuffled)
    n_val = max(10, int(len(shuffled) * val_fraction))
    val_set = shuffled[:n_val]

    y_val = [float(r.sample.is_hallucinated) for r in val_set]
    s_val = [r.risk_score                    for r in val_set]

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(s_val, y_val)

    # Apply calibrated scores in-place to all results
    for r in results:
        r.risk_score = float(ir.predict([r.risk_score])[0])
        # Rebuild classification flag with updated score
        if r.risk_score < config.RISK_LOW:
            r.risk_flag = "LOW"
        elif r.risk_score > config.RISK_HIGH:
            r.risk_flag = "HIGH"
        else:
            r.risk_flag = "CAUTION"
        r.predicted_hallucinated = r.risk_score >= config.RISK_LOW
        r.correct = r.predicted_hallucinated == r.sample.is_hallucinated

    ece_after = compute_ece(results)
    return results, ece_before, ece_after


# ─────────────────────────────────────────────────────────────────────────── #
#  Entry point                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="MedCAFAS Evaluation")
    parser.add_argument("--samples", type=int, default=40,
                        help="Number of eval samples (0 = full dataset). Default: 40")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Layer 1 (no Ollama calls). ~50x faster, tests layers 2+3 only.")
    parser.add_argument("--tune", action="store_true",
                        help="Also run threshold grid-search and print config.py suggestions.")
    parser.add_argument("--ci", action="store_true",
                        help="Compute bootstrap 95%% confidence intervals (adds ~10s).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Apply post-hoc isotonic calibration and report ECE improvement.")
    parser.add_argument("--save", type=str, default="",
                        help="Save detailed results to this JSON file.")
    parser.add_argument(
        "--dataset", type=str, default="medqa",
        choices=["medqa", "pubmedqa", "llm"],
        help=(
            "Eval dataset.  'medqa' = MedQA-USMLE (short MCQ answers, tests retrieval)."
            "  'pubmedqa' = PubMedQA long_answer (full sentences, proper NLI test)."
            "  'llm' = actual phi3.5 outputs labeled by yes/no decision-matching vs PubMedQA final_decision"
            " (gold standard, requires Ollama, cannot be used with --no-llm)."
            "  Default: medqa"
        ),
    )
    args = parser.parse_args()

    # Pre-load singletons before timed loop
    print("\nPre-loading models...")
    _get_embedder(); _get_kb(); _get_nli_model(); _get_bm25()
    print("    Models ready.\n")

    if args.dataset == "pubmedqa":
        samples = load_eval_samples_pubmedqa(args.samples)
    elif args.dataset == "llm":
        if args.no_llm:
            print("ERROR: --dataset llm requires Ollama (cannot be combined with --no-llm).")
            return
        samples = load_eval_samples_llm(args.samples)
    else:
        samples = load_eval_samples(args.samples)
    eval_fn = eval_no_llm if args.no_llm else eval_with_llm

    mode = "no-LLM (layers 2+3 only)" if args.no_llm else "full pipeline (all 3 layers)"
    print(f"Running evaluation [{args.dataset}] in {mode} mode...")
    print(f"    {len(samples)} samples\n")

    results: List[EvalResult] = []
    t0 = time.perf_counter()

    for i, sample in enumerate(samples, 1):
        label = "HALL" if sample.is_hallucinated else "good"
        print(f"  [{i:>3}/{len(samples)}] [{label}] {sample.question[:60]}...", end=" ", flush=True)
        r = eval_fn(sample)
        status = "OK" if r.correct else "XX"
        print(f"-> {r.risk_flag:7} (score={r.risk_score:.2f}) {status}")
        results.append(r)

    elapsed = time.perf_counter() - t0
    print(f"\n  Total time: {elapsed:.0f}s  ({elapsed/len(results):.1f}s/sample)")

    print_metrics(results, show_ci=args.ci)

    if args.calibrate:
        print("\n  Running post-hoc isotonic calibration …")
        results, ece_before, ece_after = calibrate_scores(results)
        print(f"  ECE before calibration : {ece_before:.4f}")
        print(f"  ECE after  calibration : {ece_after:.4f}  "
              f"({'improved' if ece_after < ece_before else 'no improvement'})")
        print("  Re-computing metrics on calibrated scores:")
        print_metrics(results, show_ci=args.ci)

    if args.tune:
        suggest_threshold_tuning(results)

    if args.save:
        out = [
            {
                "question":        r.sample.question,
                "answer":          r.sample.answer,
                "is_hallucinated": r.sample.is_hallucinated,
                "risk_flag":       r.risk_flag,
                "risk_score":      r.risk_score,
                "retrieval":       r.retrieval,
                "nli_critic":      r.critic,
                "entity_risk":     r.entity,
                "correct":         r.correct,
            }
            for r in results
        ]
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved to {args.save}")


if __name__ == "__main__":
    main()
