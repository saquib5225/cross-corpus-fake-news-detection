# Stage 2B runner hardening

## 1. Previous failure

The previous run exited before epoch-1 validation/checkpointing after a multi-hour GPU run. No Python traceback, CUDA OOM, checkpoint, or progress log was retained. The failure diagnostic identified the operational risk: the runner only saved after epoch validation and had no persistent diagnostics or resumability. The research protocol itself was not changed.

## 2. Engineering weaknesses identified

The original custom loop had no flushed training log, top-level exception file, signal diagnostics, periodic checkpoints, checkpoint integrity checks, atomic writes, or explicit resume path. It used a single DataLoader process (`num_workers=0`) and had no timeout handler or Trainer callbacks. FakeNewsAMT was already delayed until after ISOT validation selection and remains so.

## 3. Changes implemented

`scripts/run_roberta_stage2b.py` was hardened in place:

- Added flushed `FlushLogger` writing to `results/roberta/stage2b_training.log`.
- Added startup configuration snapshots at `results/roberta/stage2b_run_config.json`; existing snapshots are never overwritten and receive a timestamped name.
- Added startup/device/GPU telemetry and per-optimizer-step logging of epoch, batch, global step, loss, learning rate, elapsed time, and GPU memory.
- Added top-level exception capture to `results/roberta/stage2b_error.log` while re-raising the original exception.
- Added SIGINT/SIGTERM diagnostics where Python/Windows exposes them; the log explicitly states that hard OS/GPU resets cannot be caught.
- Added atomic checkpoints under `results/roberta/checkpoints/` using a temporary sibling directory, model/tokenizer files, `model_state.pt`, `trainer_state.pt`, and metadata, followed by integrity validation.
- Added checkpoint validation requiring model state, trainer state, metadata, and required resume fields.
- Added explicit `--resume-from PATH` and `--resume-latest` options. No checkpoint is selected automatically without an explicit resume request.
- Added deterministic epoch-seeded DataLoader construction and saved batch/epoch position so a valid checkpoint can resume without changing the research configuration.
- Added `--self-test`, which exercises synthetic checkpoint creation, atomic placement, discovery, validation, resume-state loading, and flushed logging without loading any research dataset or training a model.

## 4. Resumability

The checkpoint state contains model weights, optimizer state, scheduler state, fp16 scaler state, Python/NumPy/PyTorch/CUDA RNG state, epoch, batch index, and global optimizer step. A resume request first validates the complete checkpoint and rejects incomplete/corrupt directories. Checkpoints are written every 500 optimizer steps (approximately three per planned epoch), balancing recovery granularity against storage overhead. The best validation checkpoint remains selected only by ISOT validation Macro-F1.

## 5. Logging and failure recovery

Each log line is flushed immediately. Normal startup, device information, step progress, validation, checkpoint writes, resume events, completion, and final evaluation are recorded. Unhandled Python exceptions are written with traceback to a dedicated error file and then re-raised. Signal handlers record cooperative termination requests; kernel-level GPU resets remain outside Python's control.

## 6. Tests performed

- Python bytecode/static validation (`py_compile`) passed.
- Configuration/constants test passed: `roberta-base`, seed 42, max length 128, batch 2, accumulation 8, three epochs, and checkpoint interval 500 were confirmed.
- Synthetic checkpoint test passed: atomic checkpoint creation, required-file validation, latest-valid discovery, optimizer/scheduler/scaler/RNG resume-state loading, and flushed log assertion.
- Synthetic top-level exception logging test passed using temporary files.
- Static access-order inspection confirmed the only `pd.read_parquet` call is inside `load_external_after_selection`, after the ISOT validation-selected checkpoint is restored.
- No full training, ISOT evaluation, FakeNewsAMT access, WELFake access, or model experiment was run.

## 7. Methodological safety

The model, preprocessing, labels, split, fixed seed, sequence length, batch/accumulation configuration, validation-only selection, and delayed external evaluation protocol are unchanged. Stage 1 and Stage 2A results were not modified. A future resumed run remains methodologically valid because checkpoint recovery uses only training state and does not consult FakeNewsAMT.

## 8. Readiness and remaining risks

The runner is ready for a separately authorized new training run. The hardening improves observability and recovery but cannot catch a hard process kill or GPU/kernel reset; the most recent checkpoint may still be up to 500 optimizer steps old. A long run should be launched under a process/session mechanism with a known lifetime and persistent stdout/stderr capture.

**Status:** hardening complete; no experiment run.
