# Final Dissertation Research Story

This study investigated whether fake-news classifiers trained on ISOT retain their performance when evaluated beyond their source corpus. It addressed a methodological gap: a different dataset cannot automatically be treated as independent external evidence.

The project first audited WELFake, which had been considered as an external corpus. The audit found every one of the 39,101 unique cleaned ISOT articles in WELFake, including all train, validation and test records. WELFake was therefore rejected for independent-generalisation claims; retaining that result is itself an important data-validation contribution.

FakeNewsAMT was then selected after deterministic exact and normalised body/title checks found zero ISOT overlaps. The final external cohort contains 430 usable non-empty-body excerpts (190 fake and 240 legitimate/real). It was never used for fitting, tuning, early stopping or checkpoint selection.

Three train-only-TF-IDF baselines and `roberta-base` were trained/selected using ISOT procedures. RoBERTa had the strongest ISOT Macro-F1 (0.999742), followed by Random Forest (0.995879), Logistic Regression (0.987631) and Naive Bayes (0.958528). On FakeNewsAMT, all scores were much lower: Naive Bayes had the highest observed Macro-F1 (0.570970), followed by Logistic Regression (0.550823), RoBERTa (0.544063), and Random Forest (0.540609). The associated gaps were 38.756–45.568 percentage points.

Paired McNemar tests with Holm correction showed RoBERTa advantages on ISOT paired accuracy but no statistically significant external difference versus any baseline. This means the external ordering is descriptive for the 430-record cohort, not established evidence of a general model ranking. RoBERTa was validly early-stopped after epoch 2 tied epoch 1 validation Macro-F1; it must not be described as a fixed three-epoch completion.

Stage 4 supplied descriptive class/error profiles, model-error overlaps and baseline feature summaries from frozen outputs. RoBERTa’s FakeNewsAMT fake recall was 0.736842 and real recall was 0.400000. Integrated Gradients was attempted but did not complete, so there are no token-level attribution findings.

The contribution is a reproducible MSc-scale case study showing why corpus-independence audits and cross-corpus evaluation are necessary before interpreting very strong in-domain fake-news metrics as robustness. Its principal limitations are the small, short-excerpt, crowdsourced FakeNewsAMT cohort; possible residual dependence beyond deterministic overlap checks; dataset age/construction effects; and limited external statistical power. The final conclusion is not that RoBERTa failed, but that its excellent ISOT performance did not demonstrate superior cross-corpus generalisation on this independently screened cohort.
