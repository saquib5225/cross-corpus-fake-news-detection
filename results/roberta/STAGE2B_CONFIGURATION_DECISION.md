# Stage 2B configuration decision

## 1. Previous configuration and runtime

The stopped configuration used `roberta-base`, max length 128, batch 2, gradient accumulation 8, fp16, gradient checkpointing enabled, and 3 epochs. It measured 87.71 seconds per accumulated optimizer step, approximately 5.47 examples/minute, about 47.7 hours per epoch, and about 143 hours for three epochs.

## 2. Alternatives considered

| Alternative | Max length | Batch / accumulation | fp16 | Checkpointing | Epochs | Estimated total |
|---|---:|---:|---|---|---:|---:|
| Previous | 128 | 2 / 8 | Yes | On | 3 | ~143 h |
| A | 96 | 2 / 8 | Yes | On | 3 | ~81–96 h |
| B (selected) | 64 | 2 / 8 | Yes | Off | 3 | ~24–45 h |

All retain `roberta-base`, fixed seed 42, the existing title-plus-content representation, ISOT train-only fitting, ISOT validation-only model selection, and delayed FakeNewsAMT final evaluation. No external data informed this choice.

## 3. Selected configuration

Configuration B is selected:

- model/tokenizer: `roberta-base`;
- maximum sequence length: 64;
- per-device batch: 2;
- gradient accumulation: 8;
- effective batch: 16;
- learning rate: 2e-5;
- fp16: enabled;
- gradient checkpointing: disabled;
- AdamW, linear warmup/decay scheduler, weight decay 0.01, warmup ratio 0.1;
- seed: 42;
- epochs: 3;
- checkpoint interval: every 500 optimizer steps;
- model selection: highest ISOT validation Macro-F1.

The runner is changed only to represent this explicitly authorized configuration and to make gradient-checkpointing conditional. No methodology or labels change.

## 4. Scientific justification

Configuration B remains a standard pretrained RoBERTa classifier trained only on ISOT training data and selected only on ISOT validation data. The 64-token cap is a documented limitation: it truncates more long articles and may reduce contextual coverage. Disabling checkpointing does not change the model or data protocol; it removes activation recomputation to make the run feasible, subject to a pre-training memory smoke test. The limitation will be disclosed in the final report.

## 5. Computational justification and expected runtime

The measured bottleneck was saturated GPU model throughput, with approximately 80–100 seconds per 16-example effective update at length 128 with checkpointing. Reducing the cap to 64 reduces attention and activation work substantially; disabling checkpointing removes backward recomputation. The expected range is approximately 8–15 hours per epoch and 24–45 hours for three epochs. This is still longer than the absence window, but it is the only documented option with a realistic path to completion rather than a roughly 143-hour run. The first 500-step checkpoint is expected within the first several hours, depending on the smoke-test throughput.

## 6. Steps and checkpoint schedule

The number of microbatches and optimizer steps is unchanged by sequence length: approximately 15,640 microbatches and 1,955 optimizer steps per epoch, 5,865 for three epochs. Atomic checkpoints are scheduled at global steps 500, 1000, 1500, and so on. Each contains model, optimizer, scheduler, fp16 scaler, RNG, epoch, batch position, and global step state.

## 7. Memory considerations

The previous configuration used about 3.2 GiB of the 4 GiB GPU. Length 64 and no checkpointing are expected to reduce activation memory, but disabling checkpointing can increase peak memory relative to a checkpointed run. The hardened runner's smoke test and telemetry must pass before the loop proceeds; no batch-size increase is attempted.

## 8. Methodological limitations

The principal limitation is increased truncation at 64 tokens. This is a pre-declared input constraint, not a result-driven adjustment. It applies consistently to ISOT train, validation, test, and the eventual final external evaluation. No FakeNewsAMT metric will influence training, selection, or configuration.

## 9. Leakage and validity confirmation

FakeNewsAMT remains completely unseen until after the best checkpoint is selected on ISOT validation. WELFake is not used. The ISOT split, labels, seed, and research question remain unchanged. Stage 1 and Stage 2A files are not modified.

**Decision:** selected Configuration B; one fresh run may proceed after preflight validation.
