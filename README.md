# Fake-news classification under cross-corpus shift

## Research question

How do ISOT-trained TF-IDF baselines and RoBERTa perform in-domain and on an independently audited cross-corpus evaluation set?

## Final status

Stages 1–5 are complete. ISOT was used for training/validation/test; Naive Bayes, Logistic Regression, Random Forest and RoBERTa were compared. FakeNewsAMT (430 usable records) was evaluation-only. RoBERTa led ISOT Macro-F1 (0.999742), while Naive Bayes had the highest observed FakeNewsAMT Macro-F1 (0.570970); external paired differences were not significant after Holm correction.

## Critical data-integrity finding

WELFake is **rejected as an independent external evaluation dataset**: the audit detected all 39,101 unique cleaned ISOT articles in it. Retained WELFake predictions/results are historical audit evidence and must not be reported as independent generalisation.

## Reproducibility

Read `results/REPRODUCIBILITY_CHECKLIST.md`, `results/FINAL_RESEARCH_AUDIT.md`, and `results/FINAL_INTEGRITY_AUDIT.md`. Executed source data, models, configurations, predictions, reports, tables and figures are retained. Do not rerun scripts unless reproducing the documented study. `app.py` is a separate frozen-model Streamlit demonstration; it does not rerun the research pipeline.

## Repository layout

- `News_Dataset/`, `data/`: source and processed data
- `models/`, `results/roberta/selected_checkpoint/`: frozen models
- `scripts/`, `src/`: pipeline and analysis code
- `results/`: reports, integrity records, tables and figures

## Limitations and ethics

FakeNewsAMT comprises short excerpts with crowdsourced fake items; it is not a large-scale real-world benchmark. Exact/normalised overlap checks do not rule out all residual dependence. Respect dataset licences and avoid representing model outputs as factual verification.

## Streamlit research demonstration

`app.py` provides a presentation-focused, frozen-model demonstration of the completed study. It includes the validated model comparison, dataset-independence findings, generalisation analysis, limitations, and optional live inference. It is not a factual-verification service.

Run locally after installing `requirements.txt`:

```powershell
streamlit run app.py
```

The three classical models and TF-IDF vectorizer are loaded from this repository. RoBERTa is loaded lazily from the verified public Hugging Face repository `Cancer5225/fake-news-detection-roberta`, pinned in `app.py` to immutable revision `5bd82453a54dfa7e25e41f9323228986bb2b310e`; no token or Streamlit secret is required. Its downloaded weight is SHA-256 verified before it is loaded. See `results/STREAMLIT_DEPLOYMENT_GUIDE.md` before deployment.

## Deployment

The GitHub repository is `https://github.com/saquib5225/cross-corpus-fake-news-detection` on the deployed `main` branch. The Streamlit Community Cloud deployment was successfully completed and manually verified as live on 20 August 2026 using Python 3.14. The application is a research demonstration using frozen research results; live predictions are not factual verification.
