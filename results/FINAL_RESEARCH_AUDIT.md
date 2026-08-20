# Final Research Audit

## Status

**Complete as a synthesis of frozen Stages 1–4.** No training, tuning, prediction regeneration, or alteration of frozen experiment artefacts was performed in Stage 5.

## Research question and objectives

The study asks how classical TF-IDF classifiers and RoBERTa trained on ISOT perform in-domain and on an independently audited cross-corpus target. Objectives were to prevent leakage, compare four models, quantify generalisation gaps, test paired accuracy differences, and inspect post-hoc error patterns.

## Data, cleaning, independence and leakage

ISOT (`Fake.csv`, `True.csv`) is the sole training corpus. The final processed corpus has 39,101 unique non-empty cleaned contents after duplicate handling; its stratified 80/10/10 splits have zero exact-content intersections. Text is represented as title + text with whitespace cleaning. WELFake was audited and rejected: all 39,101 unique cleaned ISOT contents occur in it, including every train, validation and test item. Its historical predictions are retained solely as an independence finding, never as a valid external result.

FakeNewsAMT is the external evaluation cohort. The 430 usable non-empty-body records comprise 190 fake and 240 legitimate/real items; 50 fake-labelled title-only strings were excluded by a pre-specified parsing rule. Exact and normalised body/title checks detected zero ISOT overlaps. This supports absence of direct reuse under those representations, not absence of paraphrase, shared events, or source dependence.

## Methodology and results integrity

TF-IDF was fitted on ISOT train only (1–2 grams, `min_df=2`, `max_df=0.95`, 50,000 features); saved Naive Bayes, Logistic Regression and Random Forest models were externally transformed/evaluated without fitting. RoBERTa used ISOT-only fitting, ISOT validation selection, `roberta-base`, length 64, batch 2, accumulation 8, fp16, and seed 42. Epoch 1 remained selected after epoch 2 tied validation Macro-F1, triggering configured patience-one early stopping. ISOT test and FakeNewsAMT were evaluated only after selection.

RoBERTa had the highest ISOT Macro-F1 (0.999742). Naive Bayes had the highest observed FakeNewsAMT Macro-F1 (0.570970); RoBERTa obtained 0.544063. All models declined materially externally; RoBERTa's gap was 45.568 pp. Six exact paired McNemar tests with Holm correction found RoBERTa advantages on ISOT but no significant external differences. These tests concern paired accuracy, not Macro-F1 superiority.

## Explainability, reproducibility, limitations and ethics

Stage 4 provides class/error profiles, representative cases, model-error overlap and baseline feature-weight summaries from frozen outputs. Token-level Integrated Gradients did **not** complete under the managed CPU limit; no attribution claim is made. Stage 4 checked 225 frozen files: 225 unchanged, 0 modified, 0 missing. Hashes, configurations, scripts, datasets, saved models, predictions and metadata are retained; historical documentation and configuration files are flagged where stale.

Demonstrated limitations include WELFake dependence and the 430-item external cohort. Plausible limitations include ISOT age/construction bias, short excerpts, crowdsourced fake examples, label-definition/domain mismatch, residual non-text dependence, limited power, a 64-token RoBERTa input, compute-constrained design, and limited real-world generalisability. Dataset licences/provenance must be respected; do not redistribute source text beyond permitted use. The final conclusion is cross-corpus sensitivity, not that any model has universally failed or succeeded.
