# Streamlit Deployment Decision

## Decision

Use a dedicated **public Hugging Face Hub model repository** for the exact frozen selected RoBERTa checkpoint, pinned by full commit SHA and verified locally against SHA-256 `6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d`. Keep the three classical models and TF-IDF vectorizer in the GitHub application repository. Streamlit Community Cloud will download the pinned checkpoint once per fresh runtime into the Hugging Face cache and Streamlit will cache the loaded model resource.

## Rationale

This is safer than relying on undocumented Git LFS hydration in Community Cloud, preserves the checkpoint unchanged, is model-native and version-addressable, and provides a direct checksum-based reproducibility control. It does not require a token for a public model repository. If public distribution is not permitted, use the same architecture with a private Hub repository and a read-only token stored exclusively in Streamlit Community Cloud Secrets.

## Preconditions before implementation

1. Obtain approval to publish the derived checkpoint and applicable model/licence metadata.
2. Manually create the Hub model repository and upload only the four unchanged selected-checkpoint files.
3. Record the full resulting commit SHA and confirm the uploaded weight's SHA-256 and 498,612,824-byte size match the frozen local artefact.
4. Confirm a Community Cloud test deployment stays within its current memory/CPU limits.

No upload, account creation, repository modification, model modification, or application implementation has been performed in this decision stage.
