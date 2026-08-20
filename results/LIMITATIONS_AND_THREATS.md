# Limitations and Threats to Validity

## Demonstrated

- WELFake is dependent on ISOT in this workspace and cannot be an independent external test.
- FakeNewsAMT has only 430 usable records, consists of short excerpts, and contains crowdsourced deceptive-news examples.
- Large ISOT-to-FakeNewsAMT performance drops demonstrate distribution sensitivity for this design.
- Integrated Gradients did not complete; token-level attribution is unavailable.

## Possible or scope-limiting

ISOT provenance/age and dataset construction may encode source, topical or stylistic artefacts. Label definitions differ across corpora; FakeNewsAMT's crowd-authored fakes and short format create domain mismatch. Exact/normalised overlap checks cannot rule out paraphrase, shared wire material, events, or latent source dependence. The external size limits statistical power. RoBERTa uses a 64-token input and early stopping after a tie at epoch 2; these are recorded design/compute constraints, not evidence of an architecture-wide limit. Results do not generalise automatically to evolving, multilingual, long-form, or organically produced misinformation. No causal explanation follows from feature weights or error profiles.
