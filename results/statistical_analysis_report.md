# Statistical model comparison

## Methods

Exact two-sided McNemar tests compare RoBERTa with each baseline on the same labelled observations. The null hypothesis is equal probabilities of correctness on discordant paired observations. This is appropriate for paired binary correctness outcomes, but it tests accuracy differences rather than Macro-F1 differences. Six planned tests (three baselines × two datasets) use Holm correction. The 430-item FakeNewsAMT sample limits power and external scope. No causal conclusion follows from significance.

## Results

| Evaluation           | Comparator_model    | RoBERTa_only_correct | Comparator_only_correct | Discordant_pairs | Exact_McNemar_p        | Holm_adjusted_p_across_6_tests | Statistically_significant_at_0.05_after_Holm |
| -------------------- | ------------------- | -------------------- | ----------------------- | ---------------- | ---------------------- | ------------------------------ | -------------------------------------------- |
| ISOT_test            | Naive Bayes         | 161                  | 1                       | 162              | 5.576456291136357e-47  | 3.3458737746818142e-46         | True                                         |
| ISOT_test            | Logistic Regression | 48                   | 1                       | 49               | 1.7763568394002505e-13 | 8.881784197001252e-13          | True                                         |
| ISOT_test            | Random Forest       | 16                   | 1                       | 17               | 0.000274658203125      | 0.0010986328125                | True                                         |
| FakeNewsAMT_external | Naive Bayes         | 62                   | 72                      | 134              | 0.43699054908597257    | 1.0                            | False                                        |
| FakeNewsAMT_external | Logistic Regression | 64                   | 69                      | 133              | 0.7288539386735833     | 1.0                            | False                                        |
| FakeNewsAMT_external | Random Forest       | 65                   | 69                      | 134              | 0.7956287149703885     | 1.0                            | False                                        |

No bootstrap procedure was added: the exact paired test already answers the planned paired-correctness question, while unpaired confidence intervals would not establish model-to-model superiority.
