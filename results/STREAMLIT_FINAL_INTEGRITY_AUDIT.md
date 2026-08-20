# Streamlit Final Integrity Audit

## Scope and outcome

Stage 7 application testing was completed on 20 August 2026. This was an inference and presentation test only. No training, tuning, evaluation run, experiment, dataset access, or modification of frozen research results was performed.

## Tests passed

- Python syntax: `python -m py_compile app.py` completed successfully.
- Navigation: all 13 registered sections were confirmed unique and rendered in Streamlit's `AppTest` runtime without an exception: Project Overview; Why This Research Matters; How It Works; Datasets & Independence; Model Laboratory; Try the Models; Model Comparison; Generalisation; Explainability; Challenges & Solutions; Research Contribution; Limitations; and About the Project.
- Streamlit startup: a headless server started successfully and returned HTTP 200 at its local endpoint. The test server was stopped after the probe.
- Classical frozen inference: Naive Bayes, Logistic Regression, and Random Forest each loaded their existing artefact and returned a valid class and probability for a one-character input and a 50,000-character input.
- Empty input: the input-normalisation result is empty and the `Try the Models` page blocks it before any inference call, displaying an instruction to enter an article or headline.
- Result presentation: `final_results_master.csv` loaded with eight expected model/dataset rows and `final_generalisation_master.csv` loaded with four expected model rows; the table- and chart-rendering pages completed successfully.
- RoBERTa safety guard: with `ROBERTA_REPO_ID` and `ROBERTA_REVISION` absent, an isolated call failed before any Hugging Face download or model import with the clear message: `RoBERTa model repository and immutable revision are not configured.`

## Integrity and data-use checks

- `app.py` performs no training, tuning, fitting, or artefact-writing during startup. Classical artefacts are lazy-loaded only for an inference request; RoBERTa is lazy-loaded only after both deployment settings are provided.
- WELFake is not accessed by application code. It is presented only as a rejected dataset in explanatory text.
- FakeNewsAMT is not read as a dataset and is not used for training or tuning. The application reads only frozen aggregate result tables containing its evaluation metrics.
- The 498.6 MB RoBERTa checkpoint was not loaded, downloaded, or inspected in this test. No Hugging Face repository ID or immutable revision is currently configured.
- No frozen research artefact was modified. The application-facing artefacts were read only; their current SHA-256 values are recorded below for a post-test reference point.

| Artefact | SHA-256 |
|---|---|
| `models/traditional/naive_bayes.joblib` | `5811E0DAC3053D5ADEB0CE2CFD15491F0A7A087D4D1006091D1E847BDE9C3877` |
| `models/traditional/logistic_regression.joblib` | `C672F5AC59DC5649D71F8D0610B64AE42ACF151992B43EADD5C244F00F849BB1` |
| `models/traditional/random_forest.joblib` | `FBED2DC545C695D20B5A331F79ED00F13E55A4DB3B67B0E89433D86CFA4F1FD8` |
| `models/traditional/tfidf_vectorizer.joblib` | `B9D28F3985DDA415459A208B75F4D465CA1B7F167EEE9924B5245D347E53B7F3` |
| `results/tables/final_results_master.csv` | `D0AE7DE39F291701E297C3147216B21C3A23AFD4283723526C71E0F410D67816` |
| `results/tables/final_generalisation_master.csv` | `CCF6702DFE3A91288F720571A2C9D5F2AE2BBF75FF1786A722C04657A43C067E` |

## Deployment configuration review

`requirements.txt` contains the application dependencies, `.streamlit/config.toml` provides a valid dark theme and server settings, and the deployment guide correctly specifies a separately hosted immutable RoBERTa model revision with SHA-256 verification. `STREAMLIT_APPLICATION_SUMMARY.md` accurately describes lazy loading and the research boundaries. The README wording was corrected to acknowledge the included application while preserving the no-rerun instruction.

The application is ready for local demonstration, including all presentation pages and the three classical models. It is not fully ready for Streamlit Community Cloud RoBERTa inference because the required Hugging Face model repository ID and full immutable commit SHA have not been created/configured. Until those two values are set as deployment secrets (and a token if the repository is private), the application safely leaves RoBERTa unavailable; all other application functionality remains demonstrable.
