# Final Streamlit Deployment Readiness Audit

## Overall status: READY

The application deployment configuration is ready for Streamlit Community Cloud. No application-code, dependency, path, model-integrity, credential, or runtime-data blocker was found. This audit did not deploy the application.

## Application and dependency checks

| Check | Status | Evidence |
|---|---|---|
| Entry point and imports | Pass | app.py imports only Python standard-library modules plus joblib, pandas, Plotly and Streamlit. RoBERTa imports are lazy. |
| Python compatibility | Pass | Local validation ran on Python 3.11.9. Select Python 3.11 in Community Cloud Advanced settings to match the tested runtime. |
| Streamlit compatibility | Pass | Local Streamlit 1.62.0 passed startup, HTTP and 13-page rendering tests. |
| requirements.txt | Pass | It contains all direct runtime packages: streamlit, pandas, joblib, scikit-learn/scipy, plotly, torch, transformers and huggingface_hub. pip check reported no broken requirements. |
| External operating-system dependencies | Pass | None are used; packages.txt is not required. |
| Configuration | Pass | .streamlit/config.toml is present at the required repository location and contains only theme/server/client settings. |
| Credentials | Pass | No secrets file is present and no token, password, API key or credential is embedded in application configuration. |
| Runtime data | Pass | No dataset path or dataset read is used by the application. Only frozen aggregate result tables and selected visual figures are read. |

## Required application files and paths

The following application-facing local files exist at paths derived from app.py's repository root, so they are portable to Linux/Community Cloud:

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

All paths use pathlib repository-relative construction. No Windows drive path, local home directory, backslash-only path, local service, or local-only executable dependency is used.

## Model loading and integrity

### Classical models

The three classical model files and frozen TF-IDF vectorizer are present under models/traditional and are loaded only when requested through cached resources. They were previously validated with frozen inference.

### RoBERTa

| Setting | Verified value |
|---|---|
| Repository | Cancer5225/fake-news-detection-roberta |
| Immutable revision | 5bd82453a54dfa7e25e41f9323228986bb2b310e |
| Expected SHA-256 | 6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d |
| Remote access | Public; no token or secret required |
| Files requested | config.json, model.safetensors, tokenizer.json, tokenizer_config.json |

The application rejects any repository/revision other than the verified source pin. It downloads only the four required remote files, calculates model.safetensors SHA-256 before model deserialisation, and raises an error on a mismatch. The 498,612,824-byte checkpoint is not required in the GitHub application repository.

## Startup, memory and caching considerations

- Result tables use Streamlit data caching.
- Classical artefacts use resource caching and load on inference demand.
- RoBERTa is lazy-loaded and resource-cached, so browsing research pages or using only classical models does not download/load the 499 MB transformer checkpoint.
- The first RoBERTa use requires a network download/cold start and increases memory use. Local CPU validation succeeded.
- Community Cloud publishes resource limits that may change; current documentation lists approximately 2.7 GB maximum memory and 50 GB storage. The application should be monitored during its first authorised cloud deployment, especially for RoBERTa cold start and memory use.

## Navigation and research integrity

All 13 navigation sections were previously rendered successfully. The app does not train, tune, fit, save models, regenerate predictions, write to research-result paths, access ISOT/FakeNewsAMT datasets, or access WELFake.

The classical artefacts and two result tables remain at their recorded frozen SHA-256 values:

| File | SHA-256 |
|---|---|
| naive_bayes.joblib | 5811E0DAC3053D5ADEB0CE2CFD15491F0A7A087D4D1006091D1E847BDE9C3877 |
| logistic_regression.joblib | C672F5AC59DC5649D71F8D0610B64AE42ACF151992B43EADD5C244F00F849BB1 |
| random_forest.joblib | FBED2DC545C695D20B5A331F79ED00F13E55A4DB3B67B0E89433D86CFA4F1FD8 |
| tfidf_vectorizer.joblib | B9D28F3985DDA415459A208B75F4D465CA1B7F167EEE9924B5245D347E53B7F3 |
| final_results_master.csv | D0AE7DE39F291701E297C3147216B21C3A23AFD4283723526C71E0F410D67816 |
| final_generalisation_master.csv | CCF6702DFE3A91288F720571A2C9D5F2AE2BBF75FF1786A722C04657A43C067E |

## Deployment handoff

Before clicking Deploy, ensure the GitHub application repository includes the required application-facing files listed above and excludes datasets, frozen research archives not needed at runtime, credentials, and the 499 MB RoBERTa checkpoint. In Community Cloud, select Python 3.11 in Advanced settings, use app.py as the entry point, and do not add a Hugging Face secret for this public model.

## Blockers

None identified in the application deployment configuration. Deployment itself was intentionally not performed.

## Post-deployment status — 20 August 2026

This readiness audit remains a historical pre-deployment record. The GitHub repository was subsequently pushed on the `main` branch at commit `fecfcf91aee3e302b865caf554377921dad16303`, and the Streamlit Community Cloud deployment was successfully completed and manually verified as live. The deployment used Python 3.14. The deployed application retains its 13 navigation sections and four-model frozen inference demonstration. No research artefacts were changed as part of deployment. GitHub's Random Forest file-size message remained a recommendation warning only and did not prevent the repository push or deployment. The existing scientific limitations and frozen-result framing remain unchanged.
