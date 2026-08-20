# Streamlit Application Summary

## Architecture

`app.py` is a single, repository-relative Streamlit application. It reads validated master tables for all displayed metrics, loads classical artefacts lazily with `st.cache_resource`, and loads RoBERTa lazily from the verified public Hugging Face repository at a source-pinned immutable revision. The downloaded RoBERTa weight is SHA-256 checked before loading.

## Navigation

Project Overview; Why This Research Matters; How It Works; Datasets & Independence; Model Laboratory; Try the Models; Model Comparison; Generalisation; Explainability; Challenges & Solutions; Research Contribution; Limitations; and About the Project.

## Research boundaries

The interface presents WELFake as rejected, FakeNewsAMT as evaluation-only, RoBERTa as best observed in-domain, and Naive Bayes as highest observed external Macro-F1. It does not claim completed token-level Integrated Gradients, a fixed three-epoch RoBERTa completion, factual verification, or external statistical superiority.

## Privacy

The interface does not deliberately persist or log submitted article text. It exposes no credentials or internal paths.
