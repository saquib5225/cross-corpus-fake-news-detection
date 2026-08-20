# Final Model Comparison

## Comparison

| Model               | ISOT Macro_F1      | FakeNewsAMT Macro_F1 | Generalisation Gap |
| ------------------- | ------------------ | -------------------- | ------------------ |
| Naive Bayes         | 0.95852796132523   | 0.5709699809127191   | 0.3875579804125109 |
| Logistic Regression | 0.9876306723061159 | 0.5508232420093848   | 0.4368074302967311 |
| Random Forest       | 0.9958794602524897 | 0.5406086232878253   | 0.4552708369646644 |
| RoBERTa             | 0.9997424774746806 | 0.5440633130014647   | 0.4556791644732159 |

RoBERTa has the highest ISOT-test Macro-F1 (0.9997424774746806). Naive Bayes has the highest FakeNewsAMT Macro-F1 (0.5709699809127191) and the smallest Macro-F1 gap (0.3875579804125109; 38.75579804125109 percentage points). RoBERTa does not improve external Macro-F1 over the baselines. In-domain ordering does not correspond to external ordering: the strongest ISOT score has a lower external Macro-F1 than Naive Bayes.

The large drops quantify cross-corpus sensitivity and are consistent with dataset-specific learning, but do not establish that any architecture is universally superior or inferior. FakeNewsAMT has 430 usable records and is a crowdsourced-excerpt corpus; conclusions are limited to this independent cohort and label/source construction. These results should inform cautious deployment: strong in-domain validation alone is insufficient evidence of cross-corpus robustness.

The RoBERTa model was selected by ISOT validation, frozen, and then evaluated on ISOT test and FakeNewsAMT. FakeNewsAMT did not guide training, tuning, stopping, or checkpoint selection; WELFake was excluded.
