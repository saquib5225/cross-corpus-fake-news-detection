# Duplicate and overlap report

Exact duplicate counts and split intersections are in `tables/duplicate_analysis.csv`. ISOT has 5,797 duplicate-content rows before cleaning; WELFake has 8,461. The three ISOT processed splits have zero exact content intersections.

Cross-corpus matching is materially different: every one of the 39,101 unique non-empty ISOT content strings is present in WELFake. Split-level counts and label-agreement results are in `tables/dataset_overlap.csv`.

Near-duplicate screening uses punctuation-insensitive, lower-cased title signatures. It is a transparent candidate-screening method, not a semantic similarity claim. Because exact content overlap is already complete, no exhaustive fuzzy/semantic matching can change the main leakage conclusion. Sampling was not used for exact or title-signature matching.
