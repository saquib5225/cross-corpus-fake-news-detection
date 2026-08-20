# Stage 2B Final Audit

## Determination

Stage 2B training ended normally through the runner's early-stopping branch, not because of a training crash. The log records `TRAINING_COMPLETE epochs=2 global_step=3910` at 2026-08-20T16:31:37.222799+00:00. A later, separate `ImportError` occurred while rendering the Markdown report because `tabulate` was unavailable.

## Early stopping and termination

- The runner defined `EPOCHS = 3` and `EARLY_STOPPING_PATIENCE = 1` before execution.
- The selection rule was the highest ISOT-validation Macro-F1, using a strict improvement condition (`>`).
- Epoch 1 validation Macro-F1 was `0.999742399449644` at global step 1955.
- Epoch 2 validation Macro-F1 was also `0.999742399449644` at global step 3910.
- Epoch 2 was therefore one non-improving epoch under the strict comparison, satisfying patience 1 and causing the loop to stop before epoch 3.

## Best checkpoint and final model selection

The selected checkpoint is `results/roberta/selected_checkpoint/`. Its saved model timestamp (2026-08-20 05:18:42--05:18:45 BST) coincides with epoch-1 validation, so it represents the epoch-1 / global-step-1955 winner. It is a frozen selected-model directory rather than one of the periodic optimizer checkpoints. The latest periodic checkpoint is step 3500 (epoch 2, batch index 12360); it was not selected.

The runner code reloads `selected_checkpoint` before ISOT test evaluation and calls the FakeNewsAMT loader only after that reload. The final ISOT test and FakeNewsAMT values therefore use the frozen epoch-1 selected model.

## Metrics

### ISOT validation

| Epoch | Step | Accuracy | Precision | Recall | F1 / Macro-F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1955 | 0.9997442455242966 | 0.9997642621404997 | 0.9997206703910615 | 0.999742399449644 | 0.999999736481501 |
| 2 | 3910 | 0.9997442455242966 | 0.9997642621404997 | 0.9997206703910615 | 0.999742399449644 | 0.9999997364815011 |

### Final frozen-model results

| Evaluation | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| ISOT test | 0.9997443109179238 | 0.9997642621404997 | 0.9997208263539922 | 0.9997424774746806 | 0.9997424774746806 | 0.9999997366286358 |
| FakeNewsAMT | 0.5488372093023256 | 0.5752459965271078 | 0.5684210526315789 | 0.5440633130014647 | 0.5440633130014647 | 0.6015241228070175 |

Confusion matrices were saved in `confusion_matrices.csv`: ISOT test `[[1790, 1], [0, 2120]]`; FakeNewsAMT `[[140, 50], [144, 96]]`, with rows actual `[fake, real]` and columns predicted `[fake, real]`.

The saved Macro-F1 generalisation gap is `0.4556791644732159` = `45.567916447321586` percentage points.

## Leakage audit

- Model fitting: ISOT training only.
- Model selection and early stopping: ISOT validation only.
- ISOT test: evaluated only after the selected checkpoint was frozen.
- FakeNewsAMT: not used for training, tuning, early stopping, or checkpoint selection; opened only after selected-checkpoint restoration for the final external evaluation.
- The saved independence check reports zero exact and normalised body/title overlaps.
- WELFake was not opened or used.

No test-set or FakeNewsAMT information is present in the training/selection control flow, and the completed protocol is leakage-safe.

## Completion status and required disclosure

Stage 2B is methodologically valid as an early-stopped ISOT-validation-selected experiment and its final evaluations are valid under the recorded protocol. It must not be described as a completed fixed three-epoch run: the preconfigured early stopping ended it after epoch 2. The missing `tabulate` dependency prevented only the original Markdown rendering, not training, selection, testing, figures, CSVs, predictions, or metadata. The replacement report `results/roberta_evaluation_report.md` is derived solely from the preserved artifacts.

No further training is necessary to report the valid early-stopped experiment. Further training would only be necessary if a strict, non-early-stopped three-epoch protocol is a mandatory study requirement; it was not performed here.
