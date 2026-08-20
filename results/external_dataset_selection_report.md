# External dataset selection report

## Decision

**Recommended external evaluation dataset: FakeNewsAMT, the 480-record `polsci/fake-news` public mirror of the Perez-Rosas et al. `fakeNewsDataset` (COLING 2018).** It is accepted only for the explicitly defined 430-record non-empty-body subset, after the audit below.

It is preferable to WELFake because the audit finds no ISOT article/body or title reuse under any tested deterministic representation; WELFake instead contains every cleaned ISOT article and cannot support an independent-generalisation claim.

## Exact acquisition and usable cohort

- Source/version: `polsci/fake-news`, `data/train-00000-of-00001.parquet`, public Hugging Face dataset mirror retrieved on 18 August 2026 (235,598 bytes; SHA-256 `c798cb2996925f057c8170086cc873914abe0c057f559204d1ef95e8a2e3e72f`).
- Original dataset: FakeNewsAMT / `fakeNewsDataset` described by Perez-Rosas et al. (2018).
- Released records: 480; `label=0` is `legit` and `label=1` is `fake` in the dataset card.
- Usable external cohort: 430 rows with a non-empty body after deterministic title/body parsing; 240 legitimate/real and 190 fake. The 50 excluded records are fake-labelled title-only strings lacking the blank-line body separator. This exclusion must be made before any future evaluation and reported with the test size.
- Representation: NFKC-normalise line endings; the first segment before the first blank-line separator is title, and the remaining text is body. No lexical filtering, fitting, or learned preprocessing is used for the independence test.

## Independence audit

ISOT reference is the full raw `Fake.csv` + `True.csv` corpus. Exact fields are NFKC-normalised, line-ending-normalised and edge-trimmed. Normalised bodies additionally apply Unicode case-folding and whitespace collapse. Normalised titles additionally replace non-word punctuation with spaces and collapse whitespace. Empty strings are excluded. This is deterministic and implemented in `scripts/audit_external_dataset.py`.

| Check | FakeNewsAMT overlap | Candidate unique records | ISOT unique records | Result |
|---|---:|---:|---:|---|
| Exact body text | 0 | 430 | 38,638 | PASS |
| Exact title | 0 | 426 | 38,728 | PASS |
| Normalised body text | 0 | 430 | 38,637 | PASS |
| Normalised title | 0 | 425 | 38,720 | PASS |

Within FakeNewsAMT, the 430 non-empty bodies have zero exact or normalised duplicate rows. Titles have four exact duplicate rows and five normalised duplicate rows; those title duplicates are not cross-corpus overlaps. The audit also found no dataset documentation claiming ISOT reuse. FakeNewsAMT's provenance is independently collected mainstream-news excerpts plus AMT-generated fakes, rather than WELFake's mixture that demonstrably includes ISOT.

## Interpretation and remaining uncertainty

This establishes **no detected direct text/title reuse under the stated checks**, which is the required independence condition for this workspace. It does not prove that two corpora have no paraphrases, shared wire-service origin, or common factual events. The original paper further makes clear that the fake class is crowdsourced and the items are short excerpts (normally two or three paragraphs), not naturally occurring full web articles. Thus the experiment should be framed as **ISOT-to-FakeNewsAMT cross-corpus generalisation to balanced, crowdsourced deceptive-news excerpts**, not as performance on organically published misinformation at web scale.

The 430-item cohort is also much smaller than WELFake; it supports a valid but less precise external metric. Report a confidence interval in the later evaluation stage and do not compare raw accuracy claims as though the test sizes were equivalent. The source's GPL terms and the provenance of quoted mainstream excerpts should be respected; retain source/citation information and do not redistribute text beyond the dataset's permitted research use.

## Safety decision

**Safe to use for the dissertation's independent cross-dataset experiment, with the above scope and limitations disclosed.** No model work begins in this stage. The stored WELFake outputs remain historical, invalidated evidence for independence rather than being deleted.
