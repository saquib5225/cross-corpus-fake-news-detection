# Streamlit RoBERTa Configuration and Local Validation

## Configuration status: PASS

The existing Streamlit application is configured for the verified frozen public RoBERTa checkpoint. It uses a fixed source pin rather than a branch, tag, `main`, or `latest` reference.

| Setting | Value |
|---|---|
| Hugging Face repository ID | `Cancer5225/fake-news-detection-roberta` |
| Immutable revision | `5bd82453a54dfa7e25e41f9323228986bb2b310e` |
| `model.safetensors` SHA-256 | `6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d` |
| Model size | 498,612,824 bytes |
| Access | Public; no Hugging Face token or Streamlit secret is required |

## Loading and integrity strategy

- The repository ID and full immutable revision are fixed constants in `app.py`.
- RoBERTa remains lazy-loaded: project pages and classical predictions do not load it.
- `st.cache_resource` retains a verified tokenizer/model resource after its first RoBERTa request within a running process.
- `snapshot_download` is called with the full immutable revision and only the four expected checkpoint files are requested.
- Before `AutoModelForSequenceClassification.from_pretrained` is called, `model.safetensors` is SHA-256 hashed and must exactly equal the recorded frozen-checkpoint value. Any mismatch raises an error and prevents model loading.
- No credentials are configured or required because the verified repository is public.

## Local validation results

Validation was run on 20 August 2026 without accessing ISOT, FakeNewsAMT, or WELFake. It performed inference only; no model was trained, tuned, saved, or altered.

| Test | Result |
|---|---|
| Python compilation | Pass |
| Pinned repository/revision configuration | Pass |
| Local frozen checkpoint SHA-256 reference | Pass |
| Remote checkpoint download, pinned-revision and SHA-256 verification | Pass |
| Naive Bayes inference, short and 50,000-character inputs | Pass |
| Logistic Regression inference, short and 50,000-character inputs | Pass |
| Random Forest inference, short and 50,000-character inputs | Pass |
| RoBERTa inference, short and 50,000-character inputs | Pass |
| Empty-input pre-inference block | Pass |
| All four model choices/model switching | Pass |
| Frozen result-table and chart loading | Pass |
| All 13 navigation sections rendered in Streamlit test runtime | Pass |
| Headless Streamlit startup and HTTP smoke test | Pass (HTTP 200) |

## Resource and deployment considerations

The model downloaded and loaded successfully for local CPU inference. The headless application startup before any RoBERTa request used approximately 78 MB working set; this reflects lazy loading, not transformer inference memory. RoBERTa increases download time, local cache/storage use, and runtime memory on its first request. Streamlit Community Cloud deployment should therefore be tested for cold-start time and available RAM before public release.

The application is configuration-ready for Streamlit Community Cloud but has **not** been deployed in this stage. The pinned public model removes the former repository/revision configuration blocker. Deployment remains a separate authorised action.

## Integrity boundary

Only Stage 7 application and deployment documentation was changed. No Stage 1–6 artefact, frozen model, checkpoint, validated metric, prediction, table, figure, report, or dataset was modified.
