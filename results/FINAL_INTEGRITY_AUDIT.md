# Final Integrity Audit

## Result: PASS for frozen Stages 1–4

Stage 5 generated only derivative documentation, master tables and figures. It did not run training, tuning, evaluation, prediction generation, or any model checkpoint write. The Stage 4 integrity manifest remains the frozen-artifact baseline: 225 files were SHA-256 checked and all 225 were unchanged (0 modified, 0 missing), recorded in `stage4_integrity_check.json`. The selected RoBERTa `model.safetensors` hash is `6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d`, matching Stage 3/4 metadata.

WELFake is excluded from the valid external evaluation; FakeNewsAMT was evaluation-only after model selection. Stage 1–4 integrity records remain present. No conflicting final values were found among the final comparison, RoBERTa evaluation and Stage 2A reports.

## Retained stale or invalid material

`PROJECT_AUDIT.md`, `PROJECT_COMPLETION_AUDIT.md`, `NEXT_STEPS.md` and `configs/config.yaml` contain pre-Stage-1/2B plans or superseded WELFake assumptions. Historical `results/predictions/*_welfake.csv`, `cross_dataset_results.csv`, `generalisation_gap.csv` and WELFake figures/results must be labelled **historical / invalid for independent external generalisation** and excluded from the dissertation's final performance claims. They are retained for auditability, not deleted.
