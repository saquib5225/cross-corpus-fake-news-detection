# Dissertation Claim Safety Audit

| Dangerous/incorrect claim | Evidence-based alternative |
|---|---|
| “WELFake is an independent external dataset.” | WELFake is rejected because all 39,101 unique cleaned ISOT articles were detected in it. |
| “The 83.23% WELFake Random Forest result demonstrates generalisation.” | It is a historical dependent-corpus result and invalid for independent generalisation evidence. |
| “The WELFake gap was 16.36 pp.” | Do not report it as a valid external gap; final valid gaps use FakeNewsAMT. |
| “RoBERTa completed a fixed three-epoch run.” | It completed two epochs and stopped legitimately after an epoch-2 validation tie under patience-one early stopping; epoch 1 was selected. |
| “Integrated Gradients produced token-level explanations.” | Integrated Gradients was attempted but did not complete; no token-level attribution is reported. |
| “RoBERTa was the best external model.” | Naive Bayes had the highest **observed** FakeNewsAMT Macro-F1; external paired differences were not significant after Holm correction. |
| “RoBERTa generalises better than classical models.” | The completed FakeNewsAMT evaluation did not demonstrate superior RoBERTa external Macro-F1. |
| “Feature weights/error patterns explain why an item is true or false.” | They describe model behaviour under the supplied labels and are not causal or factual-verification evidence. |
| “The models detect fake news universally/in real-world deployment.” | Findings concern ISOT and the defined 430-record FakeNewsAMT cohort only. |
| “FakeNewsAMT represents all real-world misinformation.” | It is a small independent cohort of short excerpts with crowdsourced deceptive-news characteristics. |
| “No overlap proves total dataset independence.” | The audit found no direct reuse under exact/normalised text/title checks; residual paraphrase/source/event dependence remains possible. |
