# Dissertation Evidence Map

## Source hierarchy

Use Stage 5 final reports/tables and the completed Stage 1–4 reports as primary evidence. Historical WELFake outputs and pre-final planning documents are audit material only and cannot override the final audited findings.

## Research chain

| Link | Evidence-supported statement | Primary evidence |
|---|---|---|
| Problem | High in-domain fake-news metrics may not demonstrate cross-corpus robustness. | `FINAL_RESEARCH_AUDIT.md`; `final_results_master.csv` |
| Gap | An external evaluation requires direct corpus-independence checks, not merely a different dataset name. | `data_leakage_audit.md`; `external_dataset_selection_report.md` |
| Aim | Compare ISOT-trained classical TF-IDF models and RoBERTa in-domain and on an independently screened external cohort. | `FINAL_RESEARCH_AUDIT.md`; `methodology_master_summary.csv` |
| Objectives | Audit leakage; compare four models; quantify gaps; test paired accuracy; inspect frozen-output errors. | `FINAL_RESEARCH_AUDIT.md`; Stages 1–4 reports |
| Dataset selection | WELFake is rejected; FakeNewsAMT is accepted for the defined 430-record evaluation cohort. | `data_leakage_audit.md`; `external_dataset_selection_report.md`; `dataset_master_summary.csv` |
| Validation/preprocessing | ISOT has duplicate handling and zero exact split intersections; text is title + text/body with whitespace cleaning. | `duplicate_overlap_report.md`; `training_configuration.json`; `baseline_external_evaluation_report.md` |
| Experimental design | ISOT train/validation/test uses 80/10/10; TF-IDF is fit on train only; external data is post-selection evaluation only. | `methodology_master_summary.csv`; Stage 2A/2B reports |
| Models | Naive Bayes, Logistic Regression, Random Forest, and `roberta-base` are compared. | `final_results_master.csv`; `training_configuration.json` |
| Internal evaluation | RoBERTa has the highest observed ISOT Macro-F1 (0.999742). | `final_results_master.csv`; `final_model_comparison_report.md` |
| Independent external evaluation | Naive Bayes has the highest observed FakeNewsAMT Macro-F1 (0.570970); all models decline. | `final_results_master.csv`; `final_generalisation_master.csv` |
| Statistical comparison | RoBERTa advantages are significant on ISOT paired accuracy but none of its external paired comparisons survives Holm correction. | `statistical_analysis_report.md`; `statistical_model_comparison.csv` |
| Explainability/error analysis | Error profiles, overlaps and baseline feature summaries are descriptive; Integrated Gradients did not complete. | `explainability_report.md`; `roberta_attribution_status.json` |
| Contribution | The defensible contribution is a reproducible cross-corpus audit/evaluation case study, especially the WELFake-independence discovery. | `RESEARCH_CONTRIBUTION.md` |
| Limitations/conclusion | Findings are bounded by the 430-record, short-excerpt, crowdsourced external cohort and residual-dependence uncertainty. | `LIMITATIONS_AND_THREATS.md`; `FINAL_RESEARCH_FINDINGS.md` |

## Final research questions and evidence

| RQ | Experiment/dataset/metrics | Tables and figures | Finding and status | Limitation |
|---|---|---|---|---|
| RQ1. How do the four ISOT-trained models perform in-domain? | ISOT test; accuracy, macro precision/recall/F1, ROC-AUC. | `final_results_master.csv`; `figures/final/01_isot_macro_f1.png` | Fully answered: RoBERTa has highest observed ISOT Macro-F1. | ISOT may contain construction/source artefacts; in-domain results are not external robustness. |
| RQ2. How do these frozen models generalise to an independently screened external corpus? | FakeNewsAMT, 430 usable records; same metrics. | `final_results_master.csv`; `figures/final/02_fakenewsamt_macro_f1.png`, `05_fakenewsamt_confusion_comparison.png` | Conditionally answered for this cohort: all external Macro-F1 values are much lower; Naive Bayes is highest observed. | Direct text/title overlap checks cannot exclude all residual dependence; corpus is small and specialised. |
| RQ3. What is the in-domain-to-external generalisation gap? | Difference in Macro-F1. | `final_generalisation_master.csv`; `03_generalisation_gap_pp.png` | Fully calculated: gaps are 38.756–45.568 pp. | A gap is descriptive for these datasets, not a universal architecture property. |
| RQ4. Are observed external model differences supported by paired significance testing? | Six exact two-sided McNemar tests, Holm correction. | `statistical_model_comparison.csv`; `statistical_analysis_report.md` | Partially answered: no RoBERTa-versus-baseline external paired accuracy difference is significant after Holm. | Tests target paired accuracy, not Macro-F1; 430 records limit power. |
| RQ5. What error patterns are visible in the frozen outputs? | FakeNewsAMT prediction categories, class metrics, overlaps. | Stage 4 tables; `06_roberta_external_error_patterns.png`; `model_shared_error_comparison.png` | Partially answered descriptively; RoBERTa fake recall is 0.736842 and real recall 0.400000. | Not causal analysis; selected examples are illustrative; token attribution unavailable. |

## Objective mapping

| Objective | Method/experiment | Result/conclusion | Completion |
|---|---|---|---|
| Prevent dataset leakage and establish external-set independence | Exact/normalised overlap and duplicate audits | WELFake rejected; FakeNewsAMT passes stated overlap checks. | Complete |
| Establish a reproducible ISOT protocol | Duplicate-safe splits; train-only TF-IDF; saved artefacts/hashes | Frozen artefacts and integrity records retained. | Complete, with no lockfile limitation |
| Compare classical baselines and RoBERTa | Frozen ISOT/FakeNewsAMT predictions | In-domain ordering differs from observed external ordering. | Complete |
| Quantify generalisation | Macro-F1 gap tables | All models show substantial gaps. | Complete |
| Test comparative differences | McNemar + Holm | ISOT significant; external comparisons non-significant. | Complete, for paired accuracy only |
| Provide explainability | Baseline weights and post-hoc error analysis; attempted IG | Descriptive analysis complete; token-level IG incomplete. | Partially complete |

## Validated final model evidence

| Model | ISOT Macro-F1 | FakeNewsAMT Macro-F1 | Gap (pp) | Primary interpretation | Supporting files |
|---|---:|---:|---:|---|---|
| Naive Bayes | 95.853% | 57.097% | 38.756 | Highest observed external Macro-F1 and smallest gap. | `final_results_master.csv`; `final_generalisation_master.csv` |
| Logistic Regression | 98.763% | 55.082% | 43.681 | Strong ISOT result; lower observed external Macro-F1 than Naive Bayes. | same |
| Random Forest | 99.588% | 54.061% | 45.527 | Near-perfect ISOT result with marked external decline. | same |
| RoBERTa | 99.974% | 54.406% | 45.568 | Highest ISOT result; does not show superior observed external Macro-F1. | same; `roberta_evaluation_report.md` |

## Dataset evidence

**ISOT.** Source files are `News_Dataset/Fake.csv` and `True.csv`; raw counts are 23,481 fake and 21,417 real. The final processed corpus contains 39,101 unique non-empty contents. It is split stratified 80/10/10 (31,280/3,910/3,911) with zero exact-content split intersections. Use `PROJECT_AUDIT.md`, `duplicate_overlap_report.md`, `training_configuration.json` and `data_leakage_audit.md`.

**FakeNewsAMT.** The public `polsci/fake-news` mirror of Perez-Rosas et al.'s dataset supplied 480 records; the evaluation cohort is 430 non-empty-body records (190 fake, 240 legitimate/real). Dataset labels map legitimate to project label 1/real and fake to project label 0/fake. Body/title exact and normalised overlaps with ISOT are all zero. Use `external_dataset_selection_report.md`, `baseline_external_evaluation_report.md`, and `external_dataset_overlap.csv`. It remains limited to short excerpts/crowdsourced fakes and possible non-text residual dependence.

**WELFake.** It was investigated as an external candidate. All 39,101 cleaned ISOT unique articles, including every split member, were found in it; it is therefore rejected for independent external evaluation. Discuss it as a methodological/data-independence finding. Label `results/predictions/*_welfake.csv`, `cross_dataset_results.csv`, `generalisation_gap.csv`, and related WELFake figures/results historical and invalid for independent generalisation claims.

## Methodology writing blueprint

1. Research design: comparative, leakage-aware cross-corpus evaluation (`FINAL_RESEARCH_AUDIT.md`).
2. Acquisition and validation: ISOT provenance; candidate audit; FakeNewsAMT selection (`PROJECT_AUDIT.md`, selection report).
3. Leakage/duplicates/independence: exact-content split checks and body/title audits (`data_leakage_audit.md`, overlap reports).
4. Preprocessing/splits: title + text/body, whitespace cleaning, duplicate handling, stratified 80/10/10 (`baseline_external_evaluation_report.md`).
5. TF-IDF/baselines: 1–2 grams, train-only fitting, frozen models (`methodology_master_summary.csv`).
6. RoBERTa: `roberta-base`, length 64, batch 2, accumulation 8, fp16, ISOT-only fitting (`training_configuration.json`).
7. Selection/evaluation: validation-only selection; epoch-1 checkpoint retained after epoch-2 tie; post-selection ISOT then FakeNewsAMT evaluation (`roberta_evaluation_report.md`).
8. Metrics/statistics: accuracy, macro precision/recall/F1, ROC-AUC; paired exact McNemar/Holm (`final_comparison_metadata.json`, statistics report).
9. Explainability/error analysis: frozen-output profiles and weights; IG attempt incomplete (`explainability_report.md`).
10. Reproducibility/ethics: hashes, artefacts, source licences and non-redistribution caution (`REPRODUCIBILITY_CHECKLIST.md`, selection report).

## Results blueprint

Report dataset validation first, then ISOT performance, then FakeNewsAMT, gaps, paired tests, and error analysis. State exact values only from the master tables. Interpret external model ordering as observed—not statistically established after correction. Do not call error profiles causal or describe incomplete IG as an output.

## Discussion blueprint: discipline of inference

| Category | Permitted statement |
|---|---|
| Observed result | ISOT Macro-F1 is extremely high while FakeNewsAMT Macro-F1 is 0.541–0.571. |
| Possible interpretation | The cross-corpus decline is consistent with sensitivity to corpus-specific distributions/label-source differences. |
| Unsupported speculation | A particular word, architecture component, or data source *caused* a given error; one model universally generalises better; results apply to all real-world misinformation. |
