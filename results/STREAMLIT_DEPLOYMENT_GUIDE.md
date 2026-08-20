# Streamlit Deployment Guide

## Purpose

This guide deploys the frozen-model research demonstration only. It does not train, tune, evaluate, or modify any research artefact.

## 1. GitHub application repository

Commit application code, `.streamlit/config.toml`, the small classical model artefacts in `models/traditional/`, validated result tables/figures, and documentation. Do **not** commit the 498.6 MB RoBERTa weight through normal Git, do not include WELFake as application input, and do not add any secret file.

## 2. Hugging Face model repository

Create a dedicated model repository only after confirming that public distribution of the derived checkpoint is permitted. Upload unchanged files from `results/roberta/selected_checkpoint/`: `config.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`. Record the full commit SHA. Confirm that `model.safetensors` is exactly 498,612,824 bytes and SHA-256 `6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d`.

## 3. Configuration and secrets

The application pins the verified public repository `Cancer5225/fake-news-detection-roberta` and immutable revision `5bd82453a54dfa7e25e41f9323228986bb2b310e` in source. No Hugging Face token or Streamlit secret is required. Never add `.streamlit/secrets.toml`, tokens, signed URLs, or credentials for this public checkpoint. The app downloads only the pinned checkpoint files, checks the weight SHA-256, then caches the model resource.

## 4. Community Cloud deployment

1. Push the application repository to GitHub.
2. In Streamlit Community Cloud, select the repository, branch, and `app.py` entry point.
3. Add the secrets above in Advanced settings.
4. Deploy and verify the landing page, charts, each navigation section, classical inference, and the SHA-256-verified pinned RoBERTa inference.

## 5. Resource and cold-start considerations

Community Cloud has documented approximate maxima of 2.7 GB RAM and 50 GB storage, and inactive apps can hibernate. A fresh runtime may download the RoBERTa checkpoint again; a warm runtime uses the Hugging Face and Streamlit caches. RoBERTa inference runs on the available CPU unless a different platform is selected, so response time and memory must be tested before public presentation.

## Troubleshooting

- **RoBERTa unavailable:** verify the pinned repository and full immutable revision in `app.py`, confirm that revision contains all four selected-checkpoint files, and confirm the SHA-256 matches.
- **Checksum failure:** stop deployment; do not load or replace the model. Re-verify the uploaded file against the frozen local artefact.
- **Classical model error:** confirm the four `models/traditional/` files remain at repository-relative paths.
- **Resource failure:** use the app without selecting RoBERTa until a suitable hosting resource is confirmed; do not substitute a modified model.

## Local test

Install `requirements.txt` and run `streamlit run app.py`. The public immutable checkpoint requires no local configuration or credentials.
