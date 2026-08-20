# Project audit — Fake-news generalisation dissertation

**Audit date:** 18 August 2026  
**Status:** implementation in progress

## Workspace inventory

| Path | Purpose | Status |
|---|---|---|
| `Dissertation_Chapters_1-3_Revised.pdf` | Revised dissertation Chapters 1–3 | Present (223,418 bytes) |
| `News_Dataset/Fake.csv` | ISOT fake-news source file | Present (62,789,876 bytes) |
| `News_Dataset/True.csv` | ISOT real-news source file | Present (53,582,940 bytes) |

No existing source code, experiment outputs, secondary dataset, or prior implementation was found. The PDF is retained as the source project document; no changes are made to it.

## ISOT inspection

The source files were inspected programmatically. Their observed schema is `title`, `text`, `subject`, `date`.

| Class/source | Records | Empty titles | Empty body text |
|---|---:|---:|---:|
| Fake.csv | 23,481 | 0 | 630 |
| True.csv | 21,417 | 0 | 0 |
| Total before cleaning | 44,898 | 0 | 630 |

The observed counts agree with the original project description. Empty body-text records will be removed before duplicate handling and stratified splitting. The common representation will be `title + text`; raw files will remain immutable.

## Original methodology and revised extension

The original project is a TF-IDF fake-news classifier comparison using Multinomial Naive Bayes, Logistic Regression and Random Forest, with an 80/10/10 train/validation/test design and accuracy, precision, recall and F1 reporting.

The dissertation preserves those baselines and extends them with a leakage-controlled ISOT-to-WELFake external evaluation, RoBERTa, distribution-shift measures, model explanations, reproducible error analysis, calibration, and a research prototype.

## Required secondary data

The independent evaluation dataset is **WELFake** (Verma, Agrawal & Prodan, 2021), to be obtained from its original public Zenodo record: https://doi.org/10.5281/zenodo.4561253. The record describes `WELFake_Dataset.csv` (245.1 MB) with `Unnamed: 0`, `title`, `text`, `label`; label 0 is fake and label 1 is real. It is not used for training, fitting, tuning, or model selection.

## Resources and environment

- GPU: NVIDIA GeForce GTX 1650 Ti, 4 GB VRAM (CUDA driver reports CUDA 12.9).
- CPU/RAM and free disk: direct CIM/drive queries were denied by the current sandbox; this limitation is logged and runtime will be measured by the experiment runner.
- Python: no usable runtime was initially installed. Python 3.11 installation was initiated; final availability will be verified before execution.

## Missing components at audit

- Reproducible repository structure and configuration
- WELFake raw data and provenance record
- Data validation, preprocessing, split, leakage checks and overlap analysis
- Baseline and RoBERTa implementations
- Saved models, experiment artefacts, reports, figures, tests and Streamlit application

## Revised research plan

1. Audit and preserve raw data; construct a duplicate-safe 80/10/10 ISOT split.
2. Fit TF-IDF only on ISOT training records; tune classical models only against validation data.
3. Fine-tune RoBERTa only on ISOT train, using only ISOT validation for selection.
4. Evaluate the selected models once on untouched ISOT test and unseen WELFake.
5. Quantify performance gaps, data shift, attribution evidence and error patterns.
6. Produce traceable tables, figures, reports, model cards, tests and the local research prototype.
