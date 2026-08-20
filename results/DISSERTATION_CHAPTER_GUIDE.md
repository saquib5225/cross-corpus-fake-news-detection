# Dissertation Chapter Guide

## 1. Introduction
Purpose: state the cross-corpus generalisation question and objectives. Use `FINAL_RESEARCH_FINDINGS.md`; do not claim deployment-ready detection.

## 2. Literature Review
Purpose: situate datasets, leakage and transformer/classical comparison. Add only independently verified citations; do not invent citations.

## 3. Research Methodology
Purpose: document ISOT-only training, split design, TF-IDF and RoBERTa protocol. Use `configs/config.yaml`, `training_configuration.json`, and `methodology_master_summary.csv`. Do not describe the stale configuration's RoBERTa length 192/batch 4 as the executed run.

## 4. Dataset Preparation and Validation
Use leakage, duplicate, shift and external-selection reports plus `dataset_master_summary.csv`; include the WELFake overlap table and dataset figures. Do not call WELFake independent.

## 5. Experimental Design
Use Stage 2A/2B reports and model configuration. Explain validation-only selection and external evaluation after selection; do not say FakeNewsAMT tuned any model.

## 6. Results
Use `final_results_master.csv`, `final_generalisation_master.csv`, final Figures 01–05 and statistical table. Do not equate McNemar results with Macro-F1 tests.

## 7. Discussion
Interpret large gaps and non-significant external rankings using `FINAL_RESEARCH_FINDINGS.md`; do not claim RoBERTa failed.

## 8. Explainability and Error Analysis
Use Stage 4 report/tables and Figures 06 plus existing explainability figures. State Integrated Gradients did not complete.

## 9. Limitations / Threats to Validity
Use `LIMITATIONS_AND_THREATS.md`; distinguish observed limitations from plausible threats.

## 10. Conclusion and Future Work
Use the final findings and contribution. Future work may propose replication on more independent corpora; it must not be presented as completed.
