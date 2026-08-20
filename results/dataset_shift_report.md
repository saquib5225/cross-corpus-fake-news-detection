# Dataset-shift report

## Scope and method

Full-corpus descriptive statistics are in `tables/dataset_statistics_comparison.csv`. Exact-overlap analysis established that raw WELFake contains all cleaned ISOT articles, so lexical shift is calculated against the **WELFake-only remainder** after exact normalised ISOT content removal. This avoids presenting duplicated training/evaluation records as distribution shift.

Computationally expensive lexical and length analyses use deterministic **10,000-record stratified samples** (5,000 per class where available), seed **42**, sampled without replacement from deduplicated, non-empty content pools. Sampling limits TF-IDF/lexical memory and runtime while maintaining class balance. The sampled lexical measures use English-stopword-filtered unigrams (`min_df=3`, maximum 10,000 features). Exact duplicates and overlaps were evaluated on full corpora, not samples.

## Findings

The actual values are in `tables/dataset_shift_statistics.csv`. ISOT has 39,101 unique non-empty records; the non-overlapping WELFake pool has 24,572. ISOT mean article length is 425.61 words versus 779.35 in non-overlapping WELFake. Unigram vocabulary Jaccard similarity is 0.6900; mean TF-IDF-vector cosine similarity is 0.8764. The length-distribution KS statistic is 0.2998 (p=9.88e-324).

ISOT provides `subject` and `date`; WELFake does not supply comparable fields, so no subject/date shift is claimed. Figures are in `results/figures/` and refer explicitly to the stratified non-overlap samples.
