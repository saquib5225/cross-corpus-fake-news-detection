# Stage 2B RoBERTa failure diagnostic

**Diagnostic scope:** failure of the 18 August 2026 Stage 2B process. This report does not start, resume, or modify training and does not access FakeNewsAMT.

## 1. Observed failure

The active Python process (PID 24584, started 2026-08-18 22:44:33) ran for approximately 6 hours 45 minutes and then disappeared at approximately 2026-08-19 05:29. At the final observation it had consumed about 9,651 CPU-seconds and GPU telemetry showed 100% utilisation with about 3.15 GiB of 4 GiB memory in use. A later inspection found no Python process and an idle GPU.

No epoch-end checkpoint, validation log, training log, prediction file, evaluation table, or report was produced. `results/roberta/selected_checkpoint/` exists but is empty. Therefore the process stopped before the first validation/checkpoint boundary.

## 2. Evidence collected

### Files and timestamps

| Artefact | Observation |
|---|---|
| `results/roberta/sanity_test.json` | Present, 149 bytes, 22:45:56; records successful forward and backward passes, batch 2, max length 128, 974.96 MiB allocated and 1080 MiB reserved. |
| `results/roberta/stage2b_stderr.log` | 6,805 bytes; contains the earlier Hugging Face network failure from the first attempt, not a traceback from PID 24584. |
| `results/roberta/stage2b_stdout.log` | Empty, 0 bytes. |
| `results/roberta/selected_checkpoint/` | Present but empty; no checkpoint files. |
| Other `results/roberta` outputs | Absent: no training log, configuration, predictions, evaluation tables, matrices, figures, or report. |
| Temporary/checkpoint artefacts | No relevant current `.tmp`, `.incomplete`, trainer-state, or crash-dump artefact was found in the inspected workspace/model-cache locations. |

### Operating-system evidence

No Python process exit code was recoverable after the process disappeared. Application/System Event Viewer queries found no Python, CUDA, NVIDIA, OOM, or explicit process-termination event tied to PID 24584. There were no application-level Python crash events.

Windows Error Reporting did record several `LiveKernelEvent` 193 entries at 05:29:48–05:29:49, immediately around the observed process disappearance. However, their attached WATCHDOG dump filenames refer to older dates, so these records may be queued/historical reports processed at that time. They are temporal evidence of a kernel/GPU watchdog condition, not proof that this particular Python process caused it.

The process was externally observed to be active before the disappearance, with changing CPU time and GPU memory. This rules out a normal clean completion. The absence of a Python traceback and the absence of a checkpoint write indicate termination occurred outside the runner's normal Python control flow, or during a low-level native/CUDA operation.

## 3. Implementation inspection

`scripts/run_roberta_stage2b.py` has:

- no `try`/`except` around training or evaluation;
- no Python logging framework or periodic flushed progress log;
- checkpoint saving only after a complete epoch and validation pass (`model.save_pretrained` and tokenizer save);
- no Hugging Face `Trainer`, callbacks, or trainer state;
- no timeout handling, signal handling, or process supervision;
- `DataLoader(num_workers=0)` for both training and evaluation, so multiprocessing/DataLoader-worker failure is not implicated;
- normal PyTorch memory calls (`empty_cache`, peak-memory measurement, gradient checkpointing), but no per-step memory watchdog or recovery;
- a single-process custom training loop with fp16 autocast/GradScaler;
- FakeNewsAMT loading only after the selected ISOT-validation checkpoint is restored.

The implementation therefore cannot preserve progress within an epoch and cannot report an externally induced/native crash in a structured way.

## 4. Cause assessment

### Most likely cause

**External/native termination associated with a GPU/kernel watchdog or execution environment, confidence: low-to-moderate.** The strongest evidence is the disappearance without a Python traceback, the active GPU computation immediately beforehand, and `LiveKernelEvent` 193 records at the same wall-clock time. The stale/historical dump filenames prevent a definitive attribution.

### Causes not supported by available evidence

- **Python exception:** not supported; no current-run traceback exists.
- **CUDA out-of-memory:** not supported; no OOM message, and observed memory stayed below 4 GiB.
- **DataLoader failure:** unlikely; workers were disabled (`num_workers=0`) and no traceback exists.
- **Tokenizer/data-processing failure:** unlikely; tokenisation and the smoke test completed, and the failure occurred deep into training.
- **Disk/storage problem:** not supported; no disk error or partial checkpoint is present.
- **Normal script timeout:** not proven. The runner has no internal timeout. An execution-host/session limit remains possible because the process was launched from the execution environment and no process exit code was retained.
- **Sleep/power state or deliberate external kill:** no confirming event was found.

The exact cause **cannot be determined conclusively** from the retained evidence.

## 5. Approximate stopping point

The process ran from 22:44:33 until approximately 05:29, about 6 h 45 min. Because the script writes `training_log.csv` only after the epoch loop and saves checkpoints only after validation, no batch or optimizer-step number is recoverable. The last defensible point is “before epoch-1 validation/checkpoint save.”

## 6. Reproducibility of the failure

The failure is not currently reproducible without starting another training run, which this diagnostic does not do. The original failure mode is therefore **not reproducible from retained logs**. Re-running the current code would also risk losing all progress again because there is no intra-epoch checkpoint or progress log.

## 7. Safety of current implementation and minimum recommended changes

The leakage protocol is methodologically safe: FakeNewsAMT is opened only after ISOT validation-based checkpoint selection, and WELFake is not used. The training implementation is operationally unsafe for a multi-hour run because a termination before epoch end loses all work and leaves no diagnostic traceback.

Minimum changes recommended before any future run (not implemented here):

1. Add per-optimizer-step progress logging with periodic flushes and GPU memory telemetry.
2. Save resumable checkpoints at a bounded step interval (including model, optimizer, scheduler, scaler, epoch/step, and RNG state), while retaining the best ISOT-validation checkpoint separately.
3. Add top-level exception and signal logging that writes a traceback/exit marker without touching FakeNewsAMT.
4. Run under a process/session mechanism whose timeout is explicitly known, or use a detached job with persistent stdout/stderr capture.

These changes do not alter labels, data splits, model-selection criteria, or external-test protocol. No hyperparameter change is required by the evidence.

## 8. Methodological validity of a future rerun

A future rerun would not threaten methodological validity if it uses the same frozen ISOT splits, fixed configuration, ISOT-only validation selection, and delayed FakeNewsAMT access. It must be documented as a new attempt, and no external metric may influence recovery, tuning, or checkpoint selection.

## 9. Recommended next action

Do not interpret Stage 2B as completed and do not fabricate metrics. First implement only the operational safeguards above, then obtain explicit authorization before a new training attempt. Existing Stage 1 and Stage 2A artefacts remain valid and unchanged.

**Diagnostic conclusion:** the run exited before epoch 1 validation with no Python-level error or checkpoint. A GPU/kernel watchdog or execution-environment termination is the leading hypothesis, but the precise cause is unproven.
