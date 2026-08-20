# Stage 2A - Frozen traditional-ML evaluation on FakeNewsAMT

## 1. Experimental objective

Measure how the existing ISOT-trained traditional classifiers generalise to a genuinely independent external corpus. This is a frozen-model evaluation; no model, vectorizer, feature selection, preprocessing choice, hyperparameter, or model-selection decision used FakeNewsAMT. WELFake is not used.

## 2. Dataset preparation and independence

FakeNewsAMT source: `polsci/fake-news`, `train-00000-of-00001.parquet`, SHA-256 `c798cb2996925f057c8170086cc873914abe0c057f559204d1ef95e8a2e3e72f`. The release has 480 rows with labels `0=legit`, `1=fake`. The project mapping is therefore `legit -> 1 (real)` and `fake -> 0 (fake)`. The frozen external cohort contains 430 non-empty body records: 240 real and 190 fake. Fifty fake-labelled title-only rows with no body separator were excluded according to the pre-existing Stage 1 rule, not model performance.

Each stored text is split at its first blank line into title and body, then represented as title + body with whitespace collapsed, matching the established ISOT representation. The repeated full-corpus independence check returned: exact body=0, normalised body=0, exact title=0, normalised title=0.

## 3. Training, validation, and testing protocol

- Training: existing 31,280-record ISOT training split only.
- Validation: existing 3,910-record ISOT validation split only; no external data was used for selection.
- Testing: existing 3,911-record ISOT test set and the untouched 430-record FakeNewsAMT cohort.
- Feature extraction: the existing saved TF-IDF vectorizer (`ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `max_features=50,000`, lowercase), fitted on ISOT train only, was loaded and transformed external text without fitting.
- Seed: 42. Evaluation uses saved model artefacts and deterministic preprocessing.

## 4. Models and metrics

Saved ISOT-only Naive Bayes, Logistic Regression and Random Forest models were evaluated. Metrics are accuracy, macro precision/recall/F1, ROC-AUC from the saved class-1 (real) probability, and class-specific precision/recall/F1. Confusion matrices use the order fake, real.

## 5. Results

| Model | Evaluation | Accuracy | Precision | Recall | F1 | Macro_F1 | ROC_AUC | Fake_F1 | Real_F1 | Fake_support | Real_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Naive Bayes | ISOT_test | 0.9588 | 0.9587 | 0.9584 | 0.9585 | 0.9585 | 0.9900 | 0.9550 | 0.9621 | 1791 | 2120 |
| Naive Bayes | FakeNewsAMT_external | 0.5721 | 0.5891 | 0.5860 | 0.5710 | 0.5710 | 0.6159 | 0.5929 | 0.5490 | 190 | 240 |
| Logistic Regression | ISOT_test | 0.9877 | 0.9880 | 0.9873 | 0.9876 | 0.9876 | 0.9986 | 0.9865 | 0.9887 | 1791 | 2120 |
| Logistic Regression | FakeNewsAMT_external | 0.5605 | 0.6012 | 0.5854 | 0.5508 | 0.5508 | 0.6853 | 0.6166 | 0.4850 | 190 | 240 |
| Random Forest | ISOT_test | 0.9959 | 0.9959 | 0.9958 | 0.9959 | 0.9959 | 0.9997 | 0.9955 | 0.9962 | 1791 | 2120 |
| Random Forest | FakeNewsAMT_external | 0.5581 | 0.6179 | 0.5888 | 0.5406 | 0.5406 | 0.6676 | 0.6304 | 0.4509 | 190 | 240 |

## 6. Generalisation gaps

Gap = ISOT-test Macro-F1 minus FakeNewsAMT Macro-F1. A positive gap and relative decrease denote external degradation relative to in-domain Macro-F1.

| Model | ISOT_test_Macro_F1 | FakeNewsAMT_Macro_F1 | Macro_F1_gap_percentage_points | Macro_F1_relative_change_percent |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.9876 | 0.5508 | 43.6807 | 44.2278 |
| Naive Bayes | 0.9585 | 0.5710 | 38.7558 | 40.4326 |
| Random Forest | 0.9959 | 0.5406 | 45.5271 | 45.7155 |

## 7. Observations, limitations, and threats to validity

The results quantify transfer to an independent corpus rather than WELFake, whose overlap invalidated its generalisation claim. FakeNewsAMT is small, consists of short excerpts, and its fake class is crowdsourced; it is not representative of organic web misinformation. The external cohort also excludes 50 title-only fake records because no body is available, leaving modest class imbalance (240/190). Zero exact/normalised title/body matches reduce direct corpus-reuse risk but cannot rule out shared events, paraphrases, or source-level similarities. Results should therefore be interpreted as evidence for this specific external target, with uncertainty reported in any later dissertation analysis.

## 8. Reproducibility and validity decision

The run records model hashes, data hash, fixed seed, split sizes, preprocessing, metrics and saved prediction outputs in Stage 2A-specific files. No WELFake row was loaded; this runner contains no fitting operation. **Stage 2A is reproducible and complete.**
