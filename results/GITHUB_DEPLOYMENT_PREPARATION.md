# GitHub Deployment Preparation

## Preparation status

The Streamlit deployment file set is prepared locally. This workspace is not currently a Git repository: git status and git rev-parse report no .git directory. No repository was initialised, no commit was made, no remote was configured and nothing was pushed in this stage.

## Required deployment structure

The future GitHub application repository must include:

- app.py
- requirements.txt
- .streamlit/config.toml
- models/traditional/naive_bayes.joblib
- models/traditional/logistic_regression.joblib
- models/traditional/random_forest.joblib
- models/traditional/tfidf_vectorizer.joblib
- results/tables/final_results_master.csv
- results/tables/final_generalisation_master.csv
- results/figures/final/05_fakenewsamt_confusion_comparison.png
- results/figures/final/06_roberta_external_error_patterns.png
- results/figures/explainability/roberta_external_correct_incorrect.png
- results/figures/explainability/model_shared_error_comparison.png
- deployment and verification documentation under results/

These are the only local model/result/figure assets read by the application at runtime, in addition to the remote RoBERTa files.

## Git exclusions

A root .gitignore has been added. It excludes Python cache files, virtual environments, local environment/configuration files, Streamlit secrets/cache, credentials, temporary files, logs, datasets, and local RoBERTa checkpoint directories.

It deliberately does not exclude app.py, requirements.txt, .streamlit/config.toml, models/traditional, required frozen tables, required figures or deployment documentation.

The local selected checkpoint at results/roberta/selected_checkpoint/model.safetensors is 498,612,824 bytes and is excluded. The application does not use that local copy at runtime.

## Hugging Face model architecture

RoBERTa is obtained on demand from the public Hugging Face repository below, so neither the 499 MB weight nor any local training checkpoint needs to be committed to GitHub.

| Setting | Value |
|---|---|
| Repository | Cancer5225/fake-news-detection-roberta |
| Immutable revision | 5bd82453a54dfa7e25e41f9323228986bb2b310e |
| Expected SHA-256 | 6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d |
| Access | Public; no token, secret or credential required |
| Download files | config.json, model.safetensors, tokenizer.json, tokenizer_config.json |

The application allows only that repository and exact revision. It hashes model.safetensors before deserialising it and stops loading if the SHA-256 differs.

## Requirements and runtime

requirements.txt already contains all direct runtime dependencies: Streamlit, pandas, joblib, scikit-learn/scipy, Plotly, PyTorch, Transformers and Hugging Face Hub. No dependency change was needed.

Deploy using Python 3.11 in Streamlit Community Cloud Advanced settings, matching the validated local Python 3.11.9 environment. No packages.txt or secret configuration is required.

## Security and portability checks

- No hard-coded Windows drive/local-user path was found in application code.
- Application paths are derived from the repository root with pathlib.
- No committed Streamlit secrets file exists.
- Focused source/configuration/documentation scan found no credential, API key, Hugging Face token, password or bearer secret.
- Broad scans of frozen tokenizer files can match ordinary vocabulary tokens such as secret or password; those are tokenizer vocabulary entries, not credentials.

## Integrity and next step

No research artefact, model, checkpoint, metric, prediction, table, figure, dataset or Hugging Face repository content was modified.

The exact next manual step is to create an empty GitHub repository, initialise Git locally, add that remote, and selectively stage the required deployment file set while reviewing git status. Do not push the ignored datasets, local checkpoints, secrets or runtime artefacts. Do not deploy to Streamlit Community Cloud in this stage.

