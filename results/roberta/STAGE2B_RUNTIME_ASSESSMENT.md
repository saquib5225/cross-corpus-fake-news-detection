# Stage 2B runtime assessment

## Current run and stop status

The authorized fresh run used the hardened runner and was stopped after four optimizer steps. It was stopped by the operator with `Stop-Process -Force` after confirming that no checkpoint write was in progress; this was necessary to avoid continuing for days. The persistent log, run configuration, and all prior files remain. No checkpoint or result was deleted.

The run configuration is preserved in `results/roberta/stage2b_run_config.json`. No FakeNewsAMT data was accessed.

## Measured configuration

| Parameter | Value |
|---|---|
| Model/tokenizer | `roberta-base` |
| Input | Existing title + article content |
| ISOT training records | 31,280 |
| Maximum sequence length | 128 |
| Per-device batch | 2 |
| Gradient accumulation | 8 |
| Effective batch | 16 examples |
| Precision | fp16 autocast + GradScaler |
| Gradient checkpointing | Enabled |
| Epochs | 3 |
| Optimizer/scheduler | AdamW / linear warmup-linear decay |
| Checkpoint interval | Every 500 optimizer steps |
| Hardware | NVIDIA GTX 1650 Ti, 4 GiB, CUDA available |

## Observed measurements

Persistent log entries recorded:

- optimizer step 1: 80.61 seconds elapsed, batch index 8;
- optimizer step 2: 161.43 seconds elapsed, batch index 16;
- optimizer step 3: 247.38 seconds elapsed, batch index 24;
- optimizer step 4: 350.83 seconds elapsed, batch index 32.

The four-step mean is approximately 87.71 seconds per accumulated optimizer step. Each optimizer step processes 16 examples (eight microbatches of two), giving approximately **5.47 training examples/minute** and **0.684 optimizer steps/minute**. The fourth step was slower than the first three, so these are early-run estimates rather than a guaranteed steady-state rate.

There are approximately 1,955 optimizer steps per epoch (`ceil(15,640 microbatches / 8)`) and 5,865 steps for three epochs. At the measured mean, that implies approximately 47.7 hours per epoch and 143 hours for three epochs. The first 500-step checkpoint would be approximately 12.2 hours away at this rate.

## Bottleneck assessment

**Primary bottleneck: GPU model throughput.** During training, `nvidia-smi` repeatedly reported approximately 99–100% GPU utilisation, with about 3.2 GiB of 4 GiB memory in use. The process consumed CPU time but the GPU remained saturated. This is consistent with forward/backward computation dominating the step time.

- **Gradient checkpointing:** contributes materially through activation recomputation during backward; it was enabled to fit the 4 GiB card. It is a likely secondary cause of slowness, but disabling it was not tested because that would require another model run.
- **Sequence length:** 128 is the current cap and attention cost scales approximately quadratically with token length. It contributes to compute cost but was not isolated experimentally.
- **Gradient accumulation:** eight microbatches make one optimizer step, but accumulation itself is not the main per-example bottleneck; it mainly reduces optimizer-update frequency. Reducing accumulation would not materially reduce total examples' forward/backward work.
- **CPU tokenisation/DataLoader:** bulk tokenisation completed before the first step; `num_workers=0` is deterministic. The persistent 100% GPU signal during steps argues against CPU input starvation as the primary cause.
- **fp16:** enabled as requested. No evidence indicates it was disabled or malfunctioning.
- **Windows/PyTorch overhead:** possible secondary overhead, but no evidence isolates it as the dominant cause.
- **CUDA OOM:** not observed; memory remained below the 4 GiB device capacity.

## Methodologically valid alternatives (not run)

All alternatives retain `roberta-base`, seed 42, ISOT train-only fitting, ISOT validation-only selection, and FakeNewsAMT only after final selection. Estimates scale from the measured 128-token configuration and are approximate.

| Configuration | Effective batch | Steps/epoch | Estimated epoch | Estimated 3-epoch total | Trade-off |
|---|---:|---:|---:|---:|---|
| Current: 128 tokens, batch 2, accumulation 8, fp16, checkpointing on, 3 epochs | 16 | 1,955 | ~48 h | ~143 h | Best retention of the approved input and stability settings; impractical on this hardware. |
| A: 96 tokens, batch 2, accumulation 8, fp16, checkpointing on, 3 epochs | 16 | 1,955 | ~27–32 h | ~81–96 h | Less truncation than 64, but still very long; must be smoke-tested for memory and throughput. |
| B: 64 tokens, batch 2, accumulation 8, fp16, checkpointing off, 3 epochs | 16 | 1,955 | ~8–15 h | ~24–45 h | Much more practical, but truncates substantially more articles and removes the current memory safeguard; requires an explicit hardware smoke test before authorization. |

Changing accumulation from 8 to 4 while keeping effective batch 16 would increase optimizer steps but would not remove the dominant forward/backward work, so it is not recommended as a speed solution. Increasing per-device batch is also not recommended without a new memory smoke test because the current run already used approximately 3.2 GiB.

## Recommendation

Configuration B provides the best chance of a realistically executable dissertation experiment, but its 64-token truncation and disabled checkpointing are substantive engineering/method trade-offs. Configuration A preserves more context and the memory-safe checkpointing approach but remains impractical. No alternative is selected or applied here. A future choice requires explicit authorization after a short, non-training hardware smoke test; no full run was started in this assessment.

## Output and validity status

- Preserved: `results/roberta/stage2b_training.log`, `results/roberta/stage2b_run_config.json`, hardened runner, and prior diagnostic.
- Checkpoints created: none.
- Stage 1/Stage 2A results: untouched.
- FakeNewsAMT/WELFake: not accessed.
- New RoBERTa training run: not started after stopping the measured run.
