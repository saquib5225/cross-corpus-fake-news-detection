# Explainability and Detailed Error Analysis

## Scope and method

All analyses use frozen Stage 2/3 predictions. No model was fitted or evaluated anew. RoBERTa token diagnostics use Integrated Gradients over the frozen selected checkpoint input embeddings, four linear interpolation steps, maximum 64 tokens, and seed 42. Four examples were selected deterministically: the highest predicted-class-confidence record in each prediction category (with external row ID as the tie-breaker). It is appropriate for a differentiable transformer classifier and provides local sensitivity diagnostics for the predicted logit; it is not evidence of causal reasoning or factual verification.

## External class-specific performance

| class_label | class_name | support | true_positives | false_positives | false_negatives | precision           | recall             | f1_score            | error_rate         |
| ----------- | ---------- | ------- | -------------- | --------------- | --------------- | ------------------- | ------------------ | ------------------- | ------------------ |
| 0           | fake       | 190     | 140            | 144             | 50              | 0.49295774647887325 | 0.7368421052631579 | 0.5907172995780591  | 0.2631578947368421 |
| 1           | real       | 240     | 96             | 50              | 144             | 0.6575342465753424  | 0.4                | 0.49740932642487046 | 0.6                |

The confusion matrix is supplied in `roberta_external_confusion_matrix.csv`. Labels are 0=fake and 1=real.

## External error profile

| prediction_category | char_length_count | char_length_mean  | word_count_mean    | lexical_diversity_mean | punctuation_count_mean | digit_count_mean  | roberta_probability_real_mean |
| ------------------- | ----------------- | ----------------- | ------------------ | ---------------------- | ---------------------- | ----------------- | ----------------------------- |
| false_negative      | 144               | 783.6805555555555 | 130.57638888888889 | 0.6980938740107925     | 19.625                 | 5.305555555555555 | 0.0273050837808611            |
| true_positive       | 96                | 712.1145833333334 | 114.85416666666669 | 0.6975405683091428     | 16.635416666666668     | 6.145833333333333 | 0.9657393779166666            |
| true_negative       | 140               | 713.1428571428571 | 119.6              | 0.6856558733518497     | 16.521428571428572     | 3.1               | 0.0142602858206214            |
| false_positive      | 50                | 740.24            | 121.78             | 0.6757004836427886     | 16.3                   | 3.86              | 0.9441619108                  |

The aggregate patterns are descriptive. They quantify how the already-labelled FakeNewsAMT items differ by prediction category, rather than explaining the truthfulness of any article. The cross-dataset table `roberta_cross_dataset_error_analysis.csv` documents corresponding ISOT/FakeNewsAMT correct-versus-incorrect profiles. Representative excerpts are limited to stored data and are provided in `representative_error_cases.csv`.

## Model comparison

| comparison                                | records |
| ----------------------------------------- | ------- |
| all_models_wrong                          | 97      |
| roberta_only_wrong                        | 44      |
| roberta_correct_all_baselines_wrong       | 34      |
| RoBERTa_wrong_Naive Bayes_correct         | 72      |
| RoBERTa_correct_Naive Bayes_wrong         | 62      |
| RoBERTa_wrong_Logistic Regression_correct | 69      |
| RoBERTa_correct_Logistic Regression_wrong | 64      |
| RoBERTa_wrong_Random Forest_correct       | 69      |
| RoBERTa_correct_Random Forest_wrong       | 65      |

External error overlaps identify agreement/disagreement on this 430-record cohort only. The Stage 3 paired tests found no Holm-corrected external difference, so these counts are not evidence of general model superiority.

## Baseline features

Logistic Regression weights and Naive Bayes log-probability differences identify TF-IDF features associated with their class scores; Random Forest importances measure split-use importance but are unsigned and not causal. See `baseline_feature_importance.csv`.

## Limitations

FakeNewsAMT consists of short excerpts and contains 430 evaluated records. RoBERTa truncates inputs to 64 tokens, so attributions cover only the tokenized prefix. Four integration steps are a computationally constrained, therefore relatively coarse, approximation. Feature weights and attributions show model behaviour under the supplied labels; they do not establish why an item is true or false, nor causal reasoning. The selected examples are illustrative rather than representative of all errors.


## Attribution feasibility

Integrated Gradients was attempted with the frozen selected checkpoint, seed 42 and a 64-token input limit. The managed CPU execution limit terminated the required repeated transformer gradient passes before a reproducible completion. No partial token attributions or attribution figure are reported. `results/explainability/roberta_attribution_status.json` records this limitation.
