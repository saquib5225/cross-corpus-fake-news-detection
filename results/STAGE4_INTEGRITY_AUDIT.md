# Stage 4 Integrity Audit

Stage 4 performed post-hoc analysis only. The script reads frozen prediction files, preserved FakeNewsAMT and ISOT records, frozen traditional-model/vectorizer artefacts, and the selected frozen RoBERTa checkpoint. No `fit`, optimizer, training loop, checkpoint write, prediction-file write, WELFake input, or Stage 1--3 result modification occurs. The selected checkpoint SHA-256 and all sources are recorded in `explainability_metadata.json`. FakeNewsAMT is used only to analyse already-frozen predictions.
