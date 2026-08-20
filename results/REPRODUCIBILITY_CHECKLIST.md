# Reproducibility Checklist

| Item | Status / evidence |
|---|---|
| Python and libraries | Recorded in `final_comparison_metadata.json` (Python 3.11.9; pandas, NumPy, scikit-learn, SciPy) and Stage 4 metadata (Torch 2.7.1+cu118; Transformers 5.15.0). |
| Seed/configuration | Seed 42; `configs/config.yaml`, `results/roberta/training_configuration.json`; note the config file's RoBERTa settings are stale versus executed Stage 2B settings. |
| Datasets/hashes | Raw ISOT files; FakeNewsAMT SHA-256 recorded in selection/Stage 2A/2B/4 metadata; WELFake retained but rejected. |
| Processing/splits | `src/`, `scripts/`, processed split hashes in `training_configuration.json`. |
| Models/vectorizer/checkpoint | `models/traditional/`, `results/roberta/selected_checkpoint/`; selected model hash recorded in Stage 3/4 metadata. |
| Predictions/evaluation/statistics | `results/predictions/`, `results/roberta/predictions_*`, Stage 2A/2B/3 reports and scripts. |
| Error analysis | Stage 4 script, tables, figures, report and status JSON. |
| Integrity | Stage 3 audit; Stage 4 manifest and `stage4_integrity_check.json` (225/225 unchanged). |

Missing or limited: no environment lockfile with exact package hashes, no complete machine-independent data-download script for every retained candidate, and no completed RoBERTa token-attribution output. Use executed metadata rather than stale README/config wording.
