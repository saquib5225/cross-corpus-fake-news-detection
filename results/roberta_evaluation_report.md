# RoBERTa Stage 2B Evaluation Report

## Status and protocol

This report is reconstructed solely from the completed Stage 2B logs and CSV artefacts after the original report-rendering step failed because the optional `tabulate` package was unavailable. It does not rerun training or evaluation and does not alter any numerical result.

The run used `roberta-base`, maximum length 64, batch size 2, gradient accumulation 8 (effective batch size 16), fp16, seed 42, and ISOT-only fitting. The runner was configured before the run for 3 maximum epochs and early-stopping patience 1. It completed two epochs and stopped at global optimizer step 3910 after the epoch-2 validation Macro-F1 did not strictly exceed the epoch-1 best score.

The frozen selected model is `results/roberta/selected_checkpoint/`. It was saved after epoch 1 validation (global step 1955; saved 2026-08-20 05:18:42--05:18:45 BST). The runner restores this directory before it evaluates ISOT test and, only afterwards, opens FakeNewsAMT for its one final external evaluation.

## ISOT validation

| Epoch | Global optimizer step | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1955 | 0.9997442455242966 | 0.9997642621404997 | 0.9997206703910615 | 0.999742399449644 | 0.999742399449644 | 0.999999736481501 |
| 2 | 3910 | 0.9997442455242966 | 0.9997642621404997 | 0.9997206703910615 | 0.999742399449644 | 0.999742399449644 | 0.9999997364815011 |

Epoch 2 tied, rather than exceeded, the epoch-1 Macro-F1. Because the selection comparison is strict (`>`), the epoch-1 model remained selected and the configured patience of one non-improving epoch stopped training.

## Final frozen-model evaluation

| Evaluation | Records | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC | Fake F1 | Real F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ISOT test | 3911 | 0.9997443109179238 | 0.9997642621404997 | 0.9997208263539922 | 0.9997424774746806 | 0.9997424774746806 | 0.9999997366286358 | 0.9997207483943032 | 0.9997642065550578 |
| FakeNewsAMT external | 430 | 0.5488372093023256 | 0.5752459965271078 | 0.5684210526315789 | 0.5440633130014647 | 0.5440633130014647 | 0.6015241228070175 | 0.5907172995780591 | 0.49740932642487046 |

### Confusion matrices

ISOT test:

| Actual \\ Predicted | fake | real |
|---|---:|---:|
| fake | 1790 | 1 |
| real | 0 | 2120 |

FakeNewsAMT:

| Actual \\ Predicted | fake | real |
|---|---:|---:|
| fake | 140 | 50 |
| real | 144 | 96 |

## External generalisation

The saved Macro-F1 generalisation gap is `0.4556791644732159`, or `45.567916447321586` percentage points (relative decrease `45.579654234983366%`).

| Model | ISOT test Macro-F1 | FakeNewsAMT Macro-F1 | Gap (percentage points) |
|---|---:|---:|---:|
| Naive Bayes | 0.95852796132523 | 0.5709699809127191 | 38.75579804125109 |
| Logistic Regression | 0.987630672306116 | 0.5508232420093848 | 43.68074302967312 |
| Random Forest | 0.9958794602524896 | 0.5406086232878253 | 45.527083696466434 |
| RoBERTa | 0.9997424774746806 | 0.5440633130014647 | 45.567916447321586 |

## Leakage safeguards

Model fitting used only the frozen ISOT training split. ISOT validation alone governed checkpoint selection and early stopping. The selected checkpoint was frozen before ISOT test and FakeNewsAMT evaluation. FakeNewsAMT was not used for fitting, hyperparameter tuning, early stopping, or checkpoint selection; it was opened only after selection. The saved independence audit reports zero exact and normalised title/body overlaps. WELFake was not opened or used.

## Disclosure

This is a valid early-stopped Stage 2B evaluation, but it is not a fixed three-epoch completion: only two epochs were completed. This must be stated in the dissertation. The original Markdown-rendering failure is repaired by this artifact without changing results.
