# Dissertation Table and Figure Plan

| Item | Chapter/section | Suggested caption | Purpose / finding | Priority |
|---|---|---|---|---|
| `tables/dataset_master_summary.csv` | Methodology: datasets | “Datasets, roles and independence decisions” | Makes WELFake rejection and FakeNewsAMT role explicit. | Essential |
| `tables/external_dataset_overlap.csv` | Methodology: independence audit | “ISOT–FakeNewsAMT overlap audit” | Documents zero detected direct reuse. | Essential |
| `tables/final_results_master.csv` | Results: model performance | “Final in-domain and external model performance” | Main eight-row metric table. | Essential |
| `tables/final_generalisation_master.csv` | Results: generalisation | “Macro-F1 generalisation gaps” | Quantifies external decline. | Essential |
| `tables/statistical_model_comparison.csv` | Results: statistical tests | “Paired McNemar comparisons with Holm correction” | Separates observed ranking from significance. | Essential |
| `tables/roberta_external_class_performance.csv` | Error analysis | “RoBERTa class-specific FakeNewsAMT performance” | Shows fake/real recall asymmetry. | Optional |
| `figures/final/01_isot_macro_f1.png` | Results: in-domain | “ISOT-test Macro-F1 by model” | RoBERTa leads in-domain. | Essential |
| `figures/final/02_fakenewsamt_macro_f1.png` | Results: external | “FakeNewsAMT Macro-F1 by model” | Naive Bayes highest observed external score. | Essential |
| `figures/final/03_generalisation_gap_pp.png` | Results: generalisation | “ISOT-to-FakeNewsAMT Macro-F1 gap” | All gaps are large. | Essential |
| `figures/dataset_article_length_comparison.png` | Methodology: shift | “Article-length comparison across screened pools” | Supports distribution-context discussion. | Optional |
| `figures/final/05_fakenewsamt_confusion_comparison.png` | Results/error analysis | “FakeNewsAMT confusion matrices by model” | Shows error trade-offs. | Essential |
| `figures/final/06_roberta_external_error_patterns.png` | Error analysis | “RoBERTa FakeNewsAMT prediction categories” | Descriptive error distribution. | Optional |
| `figures/explainability/model_shared_error_comparison.png` | Error analysis | “Shared and model-specific external errors” | Shows overlap, not superiority. | Optional |
| `figures/explainability/baseline_top_feature_comparison.png` | Error analysis | “Illustrative baseline feature-weight patterns” | Behavioural inspection only. | Optional |

Avoid including historical WELFake performance figures/tables as final results. Do not add a token-attribution figure: Integrated Gradients did not complete.
