# External dataset candidates and independence screening

**Scope.** This stage is discovery and validation only. No model was trained, rerun, or modified. WELFake remains preserved as a documented independence failure and is excluded from selection.

| Dataset | Records | Article Text | Binary Labels | Provenance | ISOT Overlap | Independence Status | Suitability |
|---|---:|---|---|---|---|---|---|
| **FakeNewsAMT / `polsci/fake-news`** | 480 released; 430 non-empty bodies usable | Title-plus-excerpt text | Yes: legit/fake | Perez-Rosas et al. (COLING 2018); Hugging Face mirror | 0 across four checks | **PASS** | **Recommended**, subject to excerpt/artificial-fake limitations |
| MisInfoText Snopes312 strict subset | 312 released; 116 strict binary usable | Full recovered article text and title | Yes, using only `TRUE`/`FALSE` | Torabi Asr & Taboada / Snopes-checked release | 0 across four checks | **PASS** | Independent but too small for the primary external test |
| MisInfoText BuzzFeed v02 | 1,380; 1,154 after `mtrue`/`mfalse` mapping | Article text; no separate title field | Technically, but 1,090 vs 64 | MisInfoText release, pivoted from BuzzFeed fact-check data | 0 body checks; title unavailable | **PASS for tested body independence** | Not selected: severe imbalance and weak binary mapping |
| MisInfoText BuzzFeedTop | Accessible release | Recovered text | No explicit record-level binary label in file | MisInfoText release | Not tested | Not established | Reject: no defensible released binary label field |
| LIAR (binary mirror) | 12,836 | Short political statements, not news bodies | Collapsed true/false available | Wang (2017), PolitiFact claims | Not tested | Not established | Reject for this experiment: claim-level rather than article-level text |
| FakeNewsNet | Metadata lists thousands of items | Public distribution is URLs/titles; complete corpus cannot be redistributed | Yes | Shu et al.; PolitiFact/GossipCop | Not tested | Not established | Reject at this stage: reproducible full-text acquisition is not available from the public release |
| NELA-GT-2019 | 1.12M | Full articles | Proxy mapping possible only | Gruppi, Horne & Adalı (2020) | Not tested | Not established | Reject: labels are source-level reliability assessments, not article-level veracity |
| FakeNewsCorpus | 9.4M public corpus | Extracted article content | Domain/source labels | Several27 / OpenSources-derived corpus | Not tested | Not established | Reject: source-level labels and impractically broad source-label confounding |

## Candidate notes

### FakeNewsAMT

The release is the 480-item `fakeNewsDataset` portion of Perez-Rosas, Kleinberg, Lefevre and Mihalcea's COLING 2018 study, accessed as the public `polsci/fake-news` mirror. It contains `text` and `label` (`0=legit`, `1=fake`). The source paper explains that it has 240 legitimate items across six domains and 240 AMT-generated fake counterparts. The local mirror encodes a title followed by a blank line and excerpt. Splitting on the first blank-line separator retained 430 non-empty bodies: 240 legitimate and 190 fake. The 50 title-only fake rows are excluded from the selected cohort and its audit. The public mirror is GPL-licensed; cite both the dataset card and original paper. The decisive limitation is construct validity: fake stories are crowdsourced and the bodies are excerpts, not an organic web-fake corpus. It is nevertheless a clear, fully auditable, binary, independently sourced external evaluation corpus.

### MisInfoText Snopes312 strict binary subset

The accessible `snopes_checked_v02.csv` contains 312 recovered original articles, their titles, source URLs and Snopes ratings. Its five ratings are 51 `FALSE`, 53 `mostly false`, 72 `mixture`, 71 `mostly true` and 65 `TRUE`. To avoid subjective collapsing, the audit retained only the 51 `FALSE` (fake) and 65 `TRUE` (real) records. All have non-empty article text; labels are fact-checker ratings, but 116 records are too few for the dissertation's principal external estimate. The repository is GPL-3.0; source article copyright remains relevant, so redistribution beyond the published research release should be avoided.

### MisInfoText BuzzFeed v02 and BuzzFeedTop

BuzzFeed v02 is accessible and contains article text, URL/domain and a rating. Its released ratings are 1,090 `mtrue`, 64 `mfalse`, 170 `mixture`, and 56 `nofact`; no records are labelled plain true/false. The provisional `mtrue→real`, `mfalse→fake` mapping yields 1,154 records but is 94.5% real, so it is inappropriate for a macro-F1 external test and was not selected. BuzzFeedTop carries fact-check-site links and recovered text but no explicit record-level label, so no valid binary mapping or audit was claimed.

### Inaccessible or unsuitable established alternatives

LIAR has a binary mirror but is a claim/statement dataset, not article-body data. FakeNewsNet's maintainers state that the complete corpus cannot be distributed because of publisher copyright and Twitter policy; the public CSVs contain IDs, URLs, titles and tweet IDs, so automatic hydration would not create a fixed reproducible corpus. NELA-GT-2019 has article text but its ground truth is assigned at source level; treating every article from a source as true/fake would change the research target. FakeNewsCorpus has the same source-label limitation. None was assumed independent or accepted.

## Sources

- Perez-Rosas et al., [Automatic Detection of Fake News](https://aclanthology.org/C18-1287.pdf); [public FakeNewsAMT mirror](https://huggingface.co/datasets/polsci/fake-news).
- [MisInfoText catalogue](https://github.com/sfu-discourse-lab/MisInfoText) and [checked-data release](https://github.com/sfu-discourse-lab/Misinformation_detection).
- [FakeNewsNet public repository](https://github.com/KaiDMML/FakeNewsNet), [NELA-GT-2019 paper](https://arxiv.org/abs/2003.08444), [FakeNewsCorpus repository](https://github.com/several27/FakeNewsCorpus), and [LIAR binary mirror](https://huggingface.co/datasets/UKPLab/liar).
