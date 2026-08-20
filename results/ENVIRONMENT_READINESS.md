# Environment readiness verification

**Verification date:** 18 August 2026  
**Scope:** environment preparation only. No model training, data processing, dataset discovery, or evaluation was run.

## Python and hardware

| Item | Verified value | Status |
|---|---|---|
| Python version | 3.11.9 (64-bit) | Available |
| Python executable | `C:\Users\Saquib\AppData\Local\Programs\Python\Python311\python.exe` | Available |
| CPU capacity | 12 logical processors | Available |
| RAM | 17,083,187,200 bytes (15.91 GiB) total; 9,187,438,592 bytes (8.56 GiB) available at check | Available |
| NVIDIA GPU | NVIDIA GeForce GTX 1650 Ti | Detected by `nvidia-smi` |
| GPU memory | 4,096 MiB | Detected by `nvidia-smi` |
| NVIDIA driver | 576.80 | Detected by `nvidia-smi` |
| GPU compute capability | 7.5 | Detected by `nvidia-smi` |

## Required package status

| Package | Version | Status |
|---|---:|---|
| PyTorch | 2.7.1+cu118 | Installed and import/CUDA verified (CUDA 11.8 build) |
| Transformers | 5.15.0 | Installed and import verified |
| Datasets | 5.0.1 | Installed and import verified |
| Accelerate | 1.14.0 | Installed and import verified |
| Tokenizers / Safetensors | Installed as Transformers dependencies | Present |

`pip check` reports no broken requirements. PyTorch 2.7.1 with the official CUDA 11.8 wheel is compatible with Python 3.11 and the detected GTX 1650 Ti (compute capability 7.5); the installed NVIDIA driver is newer than the CUDA 11.8 runtime requirement.

## CUDA verification

The initial foreground import exceeded the command limit because first-time PyTorch CUDA initialisation took longer than the interactive timeout. A longer isolated diagnostic completed successfully: `import torch` took **15.95 seconds**, PyTorch reports CUDA **11.8**, `torch.cuda.is_available()` is `True`, one CUDA device is available, and its name is **NVIDIA GeForce GTX 1650 Ti**. A small tensor was moved to `cuda:0` and summed successfully (`3.0`). The installed CUDA-enabled PyTorch build is usable on this GPU.

## Repository readiness

There is **no RoBERTa implementation** in the repository. The only RoBERTa references are the configuration block in `configs/config.yaml`, dependency declarations, and a README command. The README's `--roberta` command remains unimplemented in `scripts/run_pipeline.py`; it was not changed during this environment-only task.

The existing ISOT preprocessing pipeline is compatible with a future RoBERTa implementation: each processed split supplies `title`, `text`, `label`, and a non-empty combined `content` field. Labels are consistently encoded as 0 = fake and 1 = real. A future tokenizer can consume `content` directly, with ISOT train used for fitting, validation for model selection, and test retained for final in-domain evaluation.

## Readiness decision

The environment is **ready for a future RoBERTa implementation and training stage**: required packages import correctly and PyTorch CUDA is functional. This verification does not start that stage; no model implementation or training was started.

## Existing results

No existing experimental result, model, prediction, table, figure, or dataset was modified. This file is the only requested environment-readiness artefact added to `results/`.
