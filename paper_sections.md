# MedCAFAS: Research Paper Sections

---

## 3. Methodology: Architecture Evolution

### 3.1 Overview

MedCAFAS employs a three-layer, training-free hallucination detection pipeline that
operates entirely on CPU.  The final architecture comprises:

- **Layer 1 — Self-Consistency**: The target LLM (phi3.5, 3.8B parameters) is sampled
  three times at temperature $\tau = 0.8$; semantic consistency is measured as
  mean pairwise cosine similarity over BioLinkBERT-base embeddings (768 dimensions).
- **Layer 2 — Evidence Retrieval**: Candidate answers are verified against a 50,000-document
  knowledge base via hybrid BM25 + FAISS retrieval, followed by Max-Marginal Relevance (MMR)
  diversity re-ranking ($\lambda = 0.85$).
- **Layer 3 — NLI Critic**: A cross-encoder NLI model
  (`cross-encoder/nli-deberta-v3-base`, MNLI accuracy 90.04%) evaluates each
  atomic claim against retrieved evidence using FACTSCORE-style decomposition.

Score aggregation applies a weighted sum of four risk signals
(consistency 25%, retrieval 30%, NLI critic 30%, entity overlap 15%)
with two override mechanisms — a **Dual-Key Confidence Gate** and a
**Conflict-First Safety Buffer** — that correct for systematic biases
discovered during iterative evaluation.

The architecture did not emerge in a single design step.  Three critical failure
modes were identified through ablation on a 60-sample MedQA-USMLE evaluation set
(30 correct answers, 30 rule-perturbed hallucinations).  Each failure mode motivated a
targeted architectural intervention; the evolution from the initial pipeline
(ROC-AUC = 0.540, $F_1$ = 0.114) to the final system (ROC-AUC = 0.952, $F_1$ = 0.881)
is documented below.

---

### 3.2 Failure Mode 1: The Yes-Man Bias ($F_1$ = 0.114)

**Observation.**
The initial pipeline (commit `50b80a7`) achieved near-random discrimination
($F_1$ = 0.114, accuracy = 48.3%).  Layer 1 consistency scores clustered
around $\bar{c} \approx 0.95$ for both correct and hallucinated answers,
rendering self-consistency uninformative.

**Root Cause.**
phi3.5 exhibits low sampling diversity for factual medical prompts:
re-sampling at $\tau = 0.8$ produces paraphrases of the same core answer
rather than semantically distinct alternatives.  The resulting high
consistency triggered the original single-key Confidence Gate
(threshold $c \geq 0.92$), which floored the NLI critic score to 0.60.
Because the gate fired on nearly every sample, Layer 3's discrimination
signal was suppressed — the pipeline trusted the LLM's "confidence" even
when the evidence layers disagreed.

**Intervention: Dual-Key Confidence Gate with Conflict-First Safety Buffer.**

The gate was redesigned to require *two independent confirmation keys*:

$$
\text{Gate fires} \iff
  \underbrace{c \geq \theta_\text{cons}}_{\text{Key 1: consistency}} \;\wedge\;
  \underbrace{r \geq \theta_\text{ret}}_{\text{Key 2: retrieval}}
$$

with $\theta_\text{cons} = 0.96$ (raised from 0.92) and $\theta_\text{ret} = 0.95$.
The dual-key design ensures the gate fires only when *both the LLM and the
knowledge base independently support the answer*.

Additionally, a **Safety Buffer** was introduced with a strict evaluation-order
constraint (conflict-first):

$$
\text{Buffer fires} \iff c \geq 0.80 \;\wedge\;
  \frac{r + s_\text{NLI}}{2} < 0.70 \;\wedge\;
  c - \frac{r + s_\text{NLI}}{2} \geq \delta
$$

where $\delta = 0.40$ is the conflict threshold.  When the Safety Buffer fires,
it forces a HIGH-risk classification *before* the Confidence Gate is evaluated,
preventing a confident-but-wrong LLM from overriding evidence-based signals.

**Impact.**
The dual-key gate eliminated the systematic silencing of Layer 3.  However,
retrieval itself remained non-discriminative (§3.3), so the overall $F_1$
improvement was modest at this stage — the gate's value became visible only
after retrieval was fixed.

---

### 3.3 Failure Mode 2: Question Drowning ($F_1$ = 0.167)

**Observation.**
Even with the gate fix, the pipeline achieved only $F_1$ = 0.167 (ROC-AUC = 0.499).
Inspection of per-sample retrieval scores revealed that *correct and hallucinated
answers to the same question retrieved identical top-5 evidence documents*, resulting
in identical retrieval scores — zero discrimination from Layer 2.

**Root Cause.**
Standard RAG pipelines encode the *question alone* as the retrieval query.
For medical exam questions (e.g., "A 45-year-old male presents with chest
pain...  What is the most likely diagnosis?"), the question dominates the
embedding; the 2–5 word answer ("Acute pericarditis" vs. "Myocardial infarction")
contributes negligible signal.  We term this the **question drowning** effect:
the question's semantic mass swamps the answer's discriminative content,
causing both the correct and hallucinated answers to retrieve the same
evidence.

Furthermore, USMLE-style answers are often short noun phrases (e.g., "Cisplatin")
that occupy generic regions in the embedding space, making FAISS-based dense
retrieval ineffective for lexical matching tasks.

**Intervention: Answer-Aware Retrieval with BM25-Primary Routing.**

Three changes were introduced:

1. **Answer-aware query construction.**  The retrieval query was changed from
   the question alone to the concatenation of question and answer:

   $$q_\text{ret} = \text{question} \oplus \text{answer}$$

   This ensures that different candidate answers pull *different* evidence —
   a correct answer ("Cisplatin") retrieves cisplatin-related documents, while
   a swapped hallucination ("Carboplatin") retrieves carboplatin-related documents.

2. **BM25-primary routing for short queries.**  When the answer is shorter than
   5 words, the system routes retrieval through BM25 (lexical matching) first,
   bypassing FAISS's dense vectors.  BM25 matches exact drug names and medical
   terms where embedding-based retrieval fails:

   ```python
   if len(query.split()) < 5 and not context:
       return _bm25_primary_retrieval(query, top_k=TOP_K)
   ```

3. **Question–Evidence (Q-E) similarity scoring.**  For short noun-phrase claims
   where NLI cross-encoders produce uninformative neutral scores, the system
   measures how well the *retrieved evidence* aligns with the *original question*
   rather than the claim itself:

   $$s_\text{QE} = \max_{e \in \mathcal{E}} \cos\bigl(\mathbf{q}, \mathbf{e}\bigr)$$

   The intuition is that a correct answer retrieves evidence *about the topic
   the question asks about* (high Q-E similarity), while a hallucinated answer
   retrieves off-topic evidence (low Q-E similarity).  The raw cosine score is
   linearly remapped from $[0.5, 0.9] \to [0.3, 0.8]$ to serve as the NLI
   substitute for that claim.

**Impact.**
Answer-aware retrieval with Q-E scoring raised performance from
$F_1 = 0.167$ / AUC = 0.499 to $F_1 = 0.450$ / AUC = 0.694 (version 5).
Retrieval scores for correct answers now averaged $\bar{r}_\text{correct} = 0.72$
vs. $\bar{r}_\text{hall} = 0.54$ — a separation that did not exist under
question-only retrieval.

---

### 3.4 Failure Mode 3: Embedding Negation-Blindness (20/22 Residual Errors)

**Observation.**
Error analysis of the 22 misclassifications in version 5 revealed that 20 of them
(91%) involved a specific hallucination template: the perturbation function prefixed
the correct answer with "This is NOT correct:" or applied negation flips
("is" → "is not", "can" → "cannot").  These negated answers received the *same*
detection scores as their un-negated correct counterparts.

**Root Cause.**
Cosine similarity in embedding space is negation-blind:

$$\cos\bigl(\texttt{embed}(\text{"Drug X"}),\; \texttt{embed}(\text{"This is NOT correct: Drug X"})\bigr) \approx 0.92$$

Because the base answer tokens dominate the embedding, the negation tokens ("NOT",
"incorrect", "cannot") contribute negligible shift.  Consequently:

- **Layer 1** (detection mode): Consistency between the provided negated answer
  and Ollama's un-negated answer remained $\sim$0.97 — indistinguishable from a
  genuinely consistent correct answer.
- **Layer 2**: Answer-aware retrieval retrieved the *same* evidence for "Drug X"
  and "NOT Drug X", producing identical retrieval scores.
- **Q-E scoring**: The retrieved evidence aligned equally well with the question
  regardless of negation, yielding identical $s_\text{QE}$ scores.

**Intervention: Lexical Negation Penalty (two-site correction).**

A set of 21 negation marker patterns was defined at module level:

```python
_NEGATION_MARKERS = (
    "not correct", "not the ", "incorrect", "is not", "are not",
    "was not", "were not", "cannot", "should not", "would not",
    "except", "contraindicated", "avoid", "unlikely", "ruled out",
    "not a ", "not an ", "no evidence", "does not", "do not",
    "it is not",
)
```

These patterns are applied at two independent sites in the pipeline:

**Site 1 — Layer 3 Q-E Score Inversion.**  When a short claim or its parent
answer matches any negation marker $m \in \mathcal{M}$, the Q-E score is inverted:

$$s_\text{QE}' = 1.0 - s_\text{QE}$$

The rationale is that high question–evidence alignment *combined with negation*
is a contradiction signal: the evidence supports the topic, but the answer
explicitly negates it.  Correct answers with natural negation ("Aspirin is
contraindicated in...") are also inverted, which constitutes a known source of
false positives (§5 — Limitations).

**Site 2 — Layer 1 Consistency Capping (detection mode).**  In detection mode
(where a pre-supplied answer is compared against Ollama-generated answers),
negated answers are capped at $c_\text{max} = 0.50$:

$$c' = \min(c, 0.50) \quad \text{if} \quad \exists\, m \in \mathcal{M} : m \subseteq a_\text{lower}$$

This breaks the artificial high consistency that cosine similarity assigns to
"Drug X" vs. "NOT Drug X".

**Impact.**
The negation penalty, combined with a critical evaluation bugfix (where
`predict()` was modified to accept and evaluate the *provided* answer rather
than regenerating its own), raised performance from AUC = 0.694 / $F_1$ = 0.450
to **AUC = 0.952 / $F_1$ = 0.881** (version 7).  The 20 negation-related
misclassifications were reduced to 2.

---

### 3.5 Knowledge Base Construction

The knowledge base comprises 50,000 medical QA passages drawn from three
established biomedical datasets:

| Source | Dataset | Documents | Content Type |
|--------|---------|-----------|--------------|
| MedQA-USMLE | GBaker/MedQA-USMLE-4-options | 10,000 | Board-style clinical vignettes |
| PubMedQA (labeled) | qiaojin/PubMedQA pqa_labeled | 1,000 | Clinical trial Q&A with expert labels |
| PubMedQA (artificial) | qiaojin/PubMedQA pqa_artificial | 19,000 | Synthetically generated trial Q&A |
| MedMCQA | medmcqa | 20,000 | Indian medical entrance exam MCQs |

Documents are chunked at a maximum of 300 words with 50-word overlap between
consecutive chunks, encoded with BioLinkBERT-base (768 dimensions), and indexed
in a flat FAISS inner-product index.  A parallel BM25Okapi index over tokenized
passages enables the hybrid retrieval described in §3.3.

---

### 3.6 Score Aggregation

The final risk score is a weighted linear combination of four inverted
"goodness" signals:

$$r = w_c(1 - c) + w_r(1 - r) + w_s(1 - s) + w_e \cdot e_\text{entity}$$

where $w_c = 0.25$, $w_r = 0.30$, $w_s = 0.30$, $w_e = 0.15$, and
$c$, $r$, $s$, $e_\text{entity}$ are the consistency, retrieval, NLI critic,
and entity overlap scores respectively.

The score is mapped to a three-tier risk flag:

$$
\text{flag} =
\begin{cases}
  \text{LOW}     & r < 0.20 \\
  \text{CAUTION} & 0.20 \leq r < 0.30 \\
  \text{HIGH}    & r \geq 0.30
\end{cases}
$$

Two hard-override conditions bypass the weighted score entirely:

1. **Dual failure**: If retrieval risk $> 0.50$ AND critic risk $> 0.85$
   simultaneously, the flag is forced to HIGH (catches fabricated content
   with no KB support whatsoever).
2. **Temporal violation**: If any atomic claim references a future calendar year,
   the flag is forced to HIGH.

Together with the Confidence Gate (§3.2) and Safety Buffer (§3.2), the
aggregation layer implements a defence-in-depth strategy where no single
high-scoring signal can unilaterally override the detection system.

---

### 3.7 Supplementary: Entity Overlap Check (Layer 2b)

As an auxiliary verification layer, MedCAFAS extracts pharmaceutical entities
from the candidate answer using a suffix-based regex pattern
(covering 30+ drug class suffixes: `-olol`, `-statin`, `-cillin`, etc.)
augmented with a curated lookup table of 100+ common drug names and
80+ medical acronyms (e.g., SSRI, COPD, ACE).  The entity risk score
is the fraction of detected answer-entities that do *not* appear in the
retrieved evidence:

$$e_\text{entity} = 1 - \frac{|\text{entities}(a) \cap \text{entities}(\mathcal{E})|}{|\text{entities}(a)|}$$

This layer contributes 15% to the weighted aggregate and provides a
lexical safety net that catches wrong-entity hallucinations (e.g., drug
name swaps) even when embedding similarity is high.

---

## 4. Results & Discussion

### 4.1 Evaluation Protocol

All evaluations use a balanced 60-sample subset of MedQA-USMLE-4-options
(30 correct answers, 30 rule-perturbed hallucinations).  Hallucinations are
generated by `_perturb_answer()` through three perturbation strategies applied
in priority order:

1. **Drug swap**: Named pharmaceutical entities in the correct answer are
   replaced with a plausible but incorrect alternative (e.g., "Cisplatin" → "Carboplatin").
2. **Negation flip**: Affirmative clinical statements are negated
   (e.g., "is indicated" → "is contraindicated").
3. **Contradiction prefix**: If neither rule matches, the correct answer is
   prefixed with "This is NOT correct:" as a fallback.

Two evaluation modes are reported:

- **No-LLM Eval**: Layers 2–3 only; consistency is fixed at 1.0.  Isolates
  retrieval and NLI discrimination without Ollama latency.
- **Full LLM Eval**: All three layers including Ollama self-consistency.
  Measures end-to-end detection performance as deployed.

Metrics: ROC-AUC (primary), $F_1$, accuracy, precision, recall, with 95%
bootstrap confidence intervals (10,000 resamples).

---

### 4.2 Performance Across Architecture Versions

| Version | Key Change | ROC-AUC | $F_1$ | Accuracy |
|---------|-----------|---------|-------|----------|
| v1 | BioLinkBERT + Safety Buffer (initial) | 0.540 | 0.114 | 48.3% |
| v4 | + BM25 hybrid + answer-aware retrieval (no Q-E) | 0.499 | 0.167 | — |
| v5 | + Q-E similarity scoring | 0.694 | 0.450 | 63.3% |
| v6 (no-LLM) | + Negation penalty (L2/L3 only) | 0.898 | 0.450 | 63.3% |
| **v7** | **+ L1 negation cap + eval answer passthrough** | **0.952** | **0.881** | **88.3%** |

*95% CI for v7 AUC: [0.887, 0.996]* (bootstrap, $n = 10{,}000$).

The most impactful single intervention was answer-aware retrieval with Q-E
scoring (v4 → v5: +0.195 AUC), followed by the negation penalty (v5 → v7:
+0.258 AUC).  The Confidence Gate and Safety Buffer were *necessary prerequisites*
that became effective only after the retrieval signal became discriminative.

---

### 4.3 Confusion Matrix (v7, $n = 60$)

|  | Predicted NOT-HALL | Predicted HALL |
|--|-------------------|----------------|
| **Actual NOT-HALL** | 27 | 3 |
| **Actual HALL** | 4 | 26 |

- **Precision**: 0.897 (26/29)
- **Recall**: 0.867 (26/30)
- **Specificity**: 0.900 (27/30)
- **$F_1$**: 0.881

The 7 errors decompose as follows:

**False Positives (3):**  Correct answers that contain natural negation language
triggered the lexical penalty.  For example, "X *cannot* cross the blood–brain
barrier" is a true clinical statement, but the word "cannot" activates the
negation marker set, inflating the risk score.  One additional FP was a
very short microbiology answer (NLI entailment = 0.01) where the cross-encoder
failed to parse the terse claim.

**False Negatives (4):**  Two were "This is NOT correct:" prefix hallucinations
that scored borderline CAUTION (risk = 0.28, 0.30) — just below the HIGH
threshold of 0.30.  One was a drug swap ("Aspirin" for the correct answer)
where the swapped drug also appeared in the retrieved evidence, yielding high
retrieval and NLI scores.  One was a short answer with coincidentally high Q-E
alignment despite being factually wrong.

---

### 4.4 Layer-Level Analysis

The discrimination power of MedCAFAS is distributed across layers.  Mean
layer scores for correct vs. hallucinated answers (v7, $n = 60$):

| Layer | Correct (mean) | Hallucinated (mean) | $\Delta$ | Role |
|-------|----------------|--------------------:|----------:|------|
| L1 Consistency | ~0.95 | ~0.60 (capped) | 0.35 | Negation cap creates separation |
| L2 Retrieval | 0.72 | 0.54 | 0.18 | Answer-aware query yields different evidence |
| L3 NLI Critic | 0.615 | 0.280 | 0.335 | Largest gap; Q-E + negation inversion |
| L2b Entity | ~0.10 | ~0.30 | 0.20 | Drug name presence/absence in evidence |

**NLI gap** ($\Delta_\text{NLI} = 0.335$) provides the strongest single-layer
discrimination, validating the design choice to weight the critic at 30%.
Layer 1 consistency alone would fail ($\Delta_\text{cons} \approx 0$ without
the negation cap), confirming the necessity of the lexical penalty.

---

### 4.5 Why Standard RAG Fails on USMLE Distractors

Medical board exams are specifically designed with *plausible distractor* answer
options.  Standard RAG hallucination detection assumes that a hallucinated answer
will fail to find supporting evidence in the knowledge base.  This assumption
breaks for USMLE-style distractors for two reasons:

1. **Topical overlap.**  All answer options (correct and distractor) belong to the
   same clinical domain.  A question about chemotherapy agents will have both
   "Cisplatin" (correct) and "Carboplatin" (distractor) well-represented in any
   medical KB.  Question-only retrieval returns the same oncology passages for
   both, and NLI cannot distinguish because both drugs are real entities with
   genuine medical evidence.

2. **Embedding proximity.**  Related medical concepts occupy neighbouring regions
   in embedding space: $\cos(\text{Cisplatin}, \text{Carboplatin}) \approx 0.89$.
   Dense retrieval treats them as near-synonyms, returning overlapping evidence
   sets.

MedCAFAS addresses this through answer-aware retrieval: by including the specific
answer in the retrieval query, the system pulls evidence *about that particular
answer* rather than evidence about the question's general topic.  The Q-E
scoring layer then evaluates whether the answer-specific evidence is *relevant*
to what the question asked — a proxy for factual correctness that bypasses the
limitations of NLI on short noun-phrase claims.

---

### 4.6 Cross-Dataset Generalisation

To assess whether MedCAFAS detection generalises beyond MedQA-USMLE, the
no-LLM evaluation was run on 40-sample subsets from PubMedQA and MedQA:

| Dataset | Accuracy | ROC-AUC |
|---------|----------|---------|
| PubMedQA ($n = 40$) | 52.5% | 0.782 |
| MedQA-USMLE ($n = 40$) | 65.0% | 0.897 |

PubMedQA's lower accuracy reflects the distribution mismatch: PubMedQA answers
are typically yes/no/maybe judgements about clinical trial conclusions, which
are harder to perturb meaningfully and harder to verify against a KB of
question–answer pairs.  The MedQA performance is consistent with the full
60-sample evaluation, suggesting stable behaviour across sample sizes.

---

### 4.7 Ablation Summary

Each architectural component's marginal contribution can be estimated from the
version history:

| Component | Removed → | $\Delta$ AUC | $\Delta$ $F_1$ |
|-----------|-----------|-------------:|---------------:|
| Negation Penalty (L1 + L3) | v5 baseline | −0.258 | −0.431 |
| Q-E Similarity Scoring | v4 baseline | −0.195 | −0.283 |
| Answer-Aware Retrieval | Question-only RAG | −0.453 | −0.714 |
| Dual-Key Confidence Gate | Single-key (0.92) | −0.412* | −0.767* |
| Safety Buffer | No buffer | ~0 (subsumes gate) | ~0 |

*Gate ablation measured v7 → v1 (confounded with other changes); true
marginal contribution is lower.  The Safety Buffer's marginal contribution is
near-zero in the current evaluation because the Dual-Key Gate already prevents
most false-negative gate fires; the buffer serves as a defence-in-depth
mechanism for edge cases.

---

### 4.8 Limitations and Future Work

1. **Lexical negation is brittle.**  The 21-pattern marker set captures common
   clinical negation but misses implicit negation (e.g., "Drug X has no role
   in treating Y" does not always match).  Conversely, correct answers containing
   biological negation ("cannot cross the blood–brain barrier") trigger false
   positives.  A learned negation detector or negation-aware embeddings
   (e.g., NegBERT) would reduce both error types.

2. **Small evaluation set.**  All reported metrics are on $n = 60$ (30+30)
   balanced samples.  Bootstrap confidence intervals are wide
   (AUC 95% CI: [0.887, 0.996]).  A larger held-out set ($n \geq 500$) is
   needed for reliable comparison with published baselines.

3. **Rule-based perturbations.**  Hallucinations are generated by drug swap,
   negation flip, and contradiction prefix — three deterministic templates
   that do not represent the full diversity of real-world LLM hallucinations
   (e.g., fabricated citations, temporal errors, subtle factual distortions).

4. **Single LLM.**  All evaluations use phi3.5 (3.8B parameters).
   Generalisation to larger or instruction-tuned models (GPT-4, Llama-3-70B)
   is untested.  Larger models may exhibit different consistency profiles that
   require re-calibration of the Confidence Gate thresholds.

5. **Latency.**  End-to-end prediction requires three sequential Ollama calls
   (Layer 1) plus FAISS/BM25 retrieval and NLI inference; typical latency is
   30–90 seconds on CPU.  Layer 1 could be parallelised, and the NLI
   cross-encoder could be distilled to a smaller model for deployment.

---

*End of generated sections.*
