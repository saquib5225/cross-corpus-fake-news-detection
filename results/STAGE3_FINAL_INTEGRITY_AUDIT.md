# Stage 3 Final Integrity Audit

This Stage 3 script is analysis-only: it reads frozen prediction files and metrics and writes new comparison artefacts. It does not import training code, fit models, change predictions, change frozen metrics, or access WELFake. All external analyses use the existing FakeNewsAMT prediction files, which were produced after model selection. RoBERTa's selected checkpoint is not modified; its SHA-256 is recorded in `final_comparison_metadata.json`. Stage 1 and Stage 2A source files are treated as read-only inputs.

Integrity checks passed: labels are aligned across models for ISOT test; FakeNewsAMT labels and external row IDs are aligned across models; no WELFake file is in the source list.
