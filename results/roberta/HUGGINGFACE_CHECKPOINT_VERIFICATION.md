# Hugging Face Frozen RoBERTa Checkpoint Verification

## Verification status: PASS

**Verification timestamp:** 2026-08-20T20:49:10.0568733+01:00  
**Repository ID:** `Cancer5225/fake-news-detection-roberta`  
**Repository URL:** https://huggingface.co/Cancer5225/fake-news-detection-roberta  
**Visible short commit SHA:** `5bd8245`  
**Full immutable commit SHA:** `5bd82453a54dfa7e25e41f9323228986bb2b310e`  
**Commit message supplied for this revision:** `Upload frozen dissertation RoBERTa checkpoint`

## Method

The four local files in `results/roberta/selected_checkpoint/` were SHA-256 hashed. The four corresponding files were then downloaded directly from the Hugging Face `resolve/<full-commit-SHA>/` endpoints into a temporary verification directory and independently SHA-256 hashed. This test did not load the model, train or tune any model, modify the local checkpoint, modify the Hugging Face repository, or modify dissertation results.

The Hugging Face model API was queried at the same full commit. It reports `model.safetensors` as a Git LFS file with the same SHA-256 and byte size shown below.

## File comparison

| File | Local bytes | Local SHA-256 | Remote bytes | Remote SHA-256 | Result |
|---|---:|---|---:|---|---|
| `config.json` | 880 | `848e4294f79d0bed1ec56e91144ad23d4d1c098061bd6256026b62662fefabdc` | 880 | `848e4294f79d0bed1ec56e91144ad23d4d1c098061bd6256026b62662fefabdc` | Exact match |
| `model.safetensors` | 498,612,824 | `6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d` | 498,612,824 | `6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d` | Exact match |
| `tokenizer.json` | 3,558,739 | `da0ac1fdf55ac64d32f34a97e4294a753fcb91a1643b9694197153c42a4e305b` | 3,558,739 | `da0ac1fdf55ac64d32f34a97e4294a753fcb91a1643b9694197153c42a4e305b` | Exact match |
| `tokenizer_config.json` | 404 | `0cbf8382b450d6b257c5c3e847e6f58424f943c68abe2a141c146c22057be9c9` | 404 | `0cbf8382b450d6b257c5c3e847e6f58424f943c68abe2a141c146c22057be9c9` | Exact match |

## Repository inventory and ambiguity check

At the pinned immutable revision, the repository contains only `.gitattributes` and the four expected checkpoint files listed above. No additional model weight file or alternate checkpoint is present. `.gitattributes` is Git LFS configuration metadata, not a model checkpoint.

## Conclusion

The Hugging Face snapshot at `5bd82453a54dfa7e25e41f9323228986bb2b310e` is byte-for-byte identical to the frozen local dissertation checkpoint for every uploaded file. The uploaded `model.safetensors` matches both the independently downloaded SHA-256 calculation and the Hugging Face API's Git LFS SHA-256 metadata. The checkpoint is verified and can be referenced by this full immutable revision in a later, separately authorised deployment stage.
