# Project completion audit

Audit scope: repository state inspected on 18 August 2026. “Completed” requires executed output, not merely code. No experiments were run for this audit.

| Component | Status | Evidence/File | Validated? | Notes |
|---|---|---|---|---|
| 1. Workspace/project audit | COMPLETED | `PROJECT_AUDIT.md` | Yes | Inventory and initial resources recorded. |
| 2. ISOT dataset validation | COMPLETED | `PROJECT_AUDIT.md`, `results/tables/isot_dataset_statistics.csv` | Yes | Counts and empty text observed. |
| 3. WELFake acquisition | COMPLETED | `data/raw/WELFake/WELFake_Dataset.csv` | Yes | 245,086,152-byte source present. |
| 4. ISOT dataset statistics | PARTIALLY COMPLETED | `results/tables/isot_dataset_statistics.csv` | Yes | Row/class summary only; no length/subject/date/vocabulary outputs. |
| 5. WELFake dataset statistics | NOT STARTED | — | No | No report-ready statistics file. |
| 6. Missing-value analysis | PARTIALLY COMPLETED | `PROJECT_AUDIT.md`, `results/data_validation_report.md` | Yes | ISOT empties and WELFake raw missing counts inspected; no complete table. |
| 7. Duplicate analysis | PARTIALLY COMPLETED | `scripts/run_pipeline.py`, `isot_dataset_statistics.csv` | Partial | Text duplicates removed, but no duplicate-analysis report/count breakdown. |
| 8. ISOT/WELFake overlap analysis | NOT STARTED | — | No | No overlap artefact. |
| 9. Data leakage audit | NOT STARTED | — | No | No formal audit. |
| 10. Preprocessing | COMPLETED | `src/preprocessing.py`, processed ISOT CSVs | Yes | Title+text, whitespace cleaning and non-empty/duplicate removal executed. |
| 11. 80/10/10 split | COMPLETED | `data/processed/isot_train.csv`, `isot_validation.csv`, `isot_test.csv` | Yes | Stratified split code and files present. |
| 12. TF-IDF pipeline | COMPLETED | `models/traditional/tfidf_vectorizer.joblib`, `results/baseline_metadata.json` | Yes | Metadata states fit on ISOT train only. |
| 13. Naive Bayes | COMPLETED | `models/traditional/naive_bayes.joblib`, baseline result CSVs | Yes | Executed in-domain and external predictions saved. |
| 14. Logistic Regression | COMPLETED | `models/traditional/logistic_regression.joblib`, baseline result CSVs | Yes | Executed in-domain and external predictions saved. |
| 15. Random Forest | COMPLETED | `models/traditional/random_forest.joblib`, baseline result CSVs | Yes | ISOT F1=.995879; WELFake F1=.832310. |
| 16. Hyperparameter tuning | NOT STARTED | — | No | Fixed values only; no validation search output. |
| 17. In-domain evaluation | COMPLETED | `results/tables/in_domain_baseline_results.csv`, prediction/confusion CSVs | Yes | Real metrics for all three baselines. |
| 18. WELFake external evaluation | COMPLETED | `results/tables/cross_dataset_results.csv`, WELFake predictions | Yes | Corrected label mapping applied before final run. |
| 19. Generalisation-gap calculation | COMPLETED | `results/tables/generalisation_gap.csv` | Yes | Metrics are direct in-domain minus external values. |
| 20. Dataset-shift analysis | NOT STARTED | — | No | No quantitative shift report. |
| 21. RoBERTa implementation | NOT STARTED | — | No | No transformer module/script. |
| 22. RoBERTa training | NOT STARTED | — | No | No checkpoint/log. |
| 23. RoBERTa ISOT evaluation | NOT STARTED | — | No | No output. |
| 24. RoBERTa WELFake evaluation | NOT STARTED | — | No | No output. |
| 25. Explainability | NOT STARTED | — | No | No attribution artefacts. |
| 26. Error analysis | NOT STARTED | — | No | Predictions exist but no systematic analysis. |
| 27. Robustness/multiple seeds | NOT STARTED | — | No | Only seed 42. |
| 28. Calibration | NOT STARTED | — | No | No calibration outputs. |
| 29. Statistical analysis | NOT STARTED | — | No | No tests/report. |
| 30. Model comparison | PARTIALLY COMPLETED | `results/tables/all_baseline_results.csv` | Yes | Baseline comparison only; excludes RoBERTa/final table. |
| 31. Experiment registry | NOT STARTED | — | No | No registry. |
| 32. Experiment metadata | PARTIALLY COMPLETED | `results/baseline_metadata.json` | Partial | One minimal metadata file; no per-experiment hardware/version/timestamps. |
| 33. Dissertation-quality tables | PARTIALLY COMPLETED | `results/tables/*.csv` | Yes | Baseline tables only; no inventory/Markdown/LaTeX/full study tables. |
| 34. Dissertation-quality figures | NOT STARTED | — | No | `results/figures/` is empty. |
| 35. Figure inventory | NOT STARTED | — | No | No file. |
| 36. Table inventory | NOT STARTED | — | No | No file. |
| 37. Architecture diagram | NOT STARTED | — | No | No file. |
| 38. Research workflow diagram | NOT STARTED | — | No | No file. |
| 39. Model cards | NOT STARTED | — | No | No cards. |
| 40. Ethical analysis | NOT STARTED | — | No | No file. |
| 41. Limitations analysis | NOT STARTED | — | No | No file. |
| 42. Failed-experiment documentation | NOT STARTED | — | No | Syntax/install issues not formally documented. |
| 43. Automated tests | NOT STARTED | — | No | `tests/` is empty. |
| 44. Streamlit application | NOT STARTED | — | No | `app/` is empty. |
| 45. Application testing | NOT STARTED | — | No | No application exists. |
| 46. Reproducibility documentation | PARTIALLY COMPLETED | `README.md`, `requirements.txt`, `configs/config.yaml` | Partial | Setup exists; no complete command/data/provenance guidance. |
| 47. Research-question mapping | NOT STARTED | — | No | No mapping file. |
| 48. Hypothesis evaluation | NOT STARTED | — | No | No evidence summary. |
| 49. Final implementation report | NOT STARTED | — | No | No file. |
| 50. Dissertation evidence package | NOT STARTED | — | No | No file. |
| 51. Viva evidence package | NOT STARTED | — | No | No file. |
| 52. Final leakage audit | NOT STARTED | — | No | No PASS/FAIL audit. |
| Historical pre-validation WELFake external results | INVALIDATED | Earlier `cross_dataset_results.csv` revision; `results/data_validation_report.md` | Yes | Opposite WELFake encoding produced RF F1 .149993; superseded by final CSV. |

## Verified Random Forest results

`results/tables/in_domain_baseline_results.csv` records ISOT macro-F1 **0.9958794603**. `results/tables/cross_dataset_results.csv` records WELFake accuracy **0.8327260734**, macro-F1 **0.8323096608**, and ROC-AUC **0.9604780154**. `results/tables/generalisation_gap.csv` records F1 gap **0.1635697995** (16.36 percentage points). These are generated CSV values, not terminal-only values.

## Totals

- COMPLETED: 12
- PARTIALLY COMPLETED: 7
- NOT STARTED: 33
- FAILED: 0
- INVALIDATED: 1
