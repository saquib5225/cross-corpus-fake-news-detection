# Final Research Findings

## 1. Executive summary

This completed study compares three ISOT-trained TF-IDF baselines with ISOT-trained RoBERTa under a leakage-aware in-domain and cross-corpus design. Very high ISOT performance did not transfer to the independent FakeNewsAMT cohort.

## 2–4. Dataset findings, WELFake discovery and FakeNewsAMT selection

WELFake was initially considered, then rejected after the audit found 100% cleaned ISOT-content inclusion. This prevented an invalid external-generalisation claim. FakeNewsAMT was selected after zero detected exact/normalised body/title overlaps. It is an independent cross-corpus evaluation dataset of 430 usable short excerpts (190 fake, 240 legitimate/real), not a large-scale benchmark; its fake examples are crowdsourced and residual dependence remains possible.

## 5–8. Model findings and generalisation

ISOT Macro-F1 values were 0.958528 (Naive Bayes), 0.987631 (Logistic Regression), 0.995879 (Random Forest), and 0.999742 (RoBERTa). FakeNewsAMT Macro-F1 values were 0.570970, 0.550823, 0.540609, and 0.544063 respectively. Naive Bayes therefore had the highest observed external Macro-F1; RoBERTa's exceptionally strong ISOT result does not establish superior cross-corpus generalisation. Its 45.568-pp gap was the largest, only slightly above Random Forest's 45.527 pp.

## 9. Statistical significance

Paired exact McNemar tests (six tests, Holm correction) found significant ISOT accuracy advantages for RoBERTa versus each baseline. No FakeNewsAMT comparison was significant after correction. With 430 external records, observed rank differences require cautious interpretation.

## 10–11. Error analysis and explainability

RoBERTa's external matrix was [[140, 50], [144, 96]], giving fake recall 0.736842 and real recall 0.400000. Stage 4's descriptive profiles, model-error overlaps and feature-weight analyses support discussion of model behaviour, not causal truth assessment. Integrated Gradients was attempted but did not complete; there are no token-attribution findings.

## 12–15. Contribution, limitations and conclusion

The main contribution is methodological: a reproducible audit exposed WELFake dependence and replaced it with a documented independently screened external cohort. The evidence demonstrates that in-domain fake-news classification metrics alone are inadequate evidence of cross-corpus robustness. It does not establish real-world misinformation performance, a universal ranking, or causal explanations.
