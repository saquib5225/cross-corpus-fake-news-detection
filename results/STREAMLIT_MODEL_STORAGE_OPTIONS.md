# Streamlit Model Storage Options

## Deployment problem

The validated selected RoBERTa weight file is `results/roberta/selected_checkpoint/model.safetensors`, 498,612,824 bytes (about 498.6 MB), SHA-256 `6549528157c51bb2132b8c9fd730ee3f94f15f7eac4cf98121606794f4da590d`. It must be deployed unchanged. The application repository can retain the comparatively small classical artefacts locally: Naive Bayes (1.6 MB), Logistic Regression (0.4 MB), Random Forest (64.4 MB), and TF-IDF vectorizer (2.0 MB).

GitHub enforces a 100 MB normal-Git single-object limit and recommends LFS for binary files; its recommended on-disk repository limit is 10 GB. Therefore this weight cannot be committed normally. [GitHub repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)

Streamlit Community Cloud runs from the repository root and its published approximate limits are up to 2.7 GB memory and 50 GB storage; apps hibernate after 12 hours without traffic. A 498.6 MB model fits the stated disk ceiling, but CPU/RAM cold-start and inference behaviour must be tested after deployment. [Community Cloud limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [repository-root behaviour](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)

## Option comparison

| Option | Feasibility / limits | Complexity, reproducibility and reliability | Security/cost | Dissertation suitability |
|---|---|---|---|---|
| A. Git LFS in the application repository | The checkpoint fits GitHub Free/Pro LFS's 2 GB per-file limit. GitHub Free includes 10 GiB storage and 10 GiB monthly download bandwidth. | Low implementation work if the deployment checkout hydrates LFS. However, current official Streamlit docs do not state that Community Cloud initializes Git LFS during repository checkout. A pointer-only checkout would fail. This must not be assumed. | Public LFS needs no token but consumes owner bandwidth; each cold deployment/download counts. Over quota can disable LFS. | Not recommended as the primary public-demo design because the critical hydration behaviour is not officially guaranteed. |
| B. Generic immutable external object/release asset | A GitHub release asset can be under 2 GiB; 498.6 MB fits. [GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) | App downloads a URL once to local cache, validates SHA-256, then loads locally. A release/tag plus published SHA is reproducible, but a release asset is less model-specific than a Hub commit and the download implementation must be maintained. | Public asset needs no secret; private asset requires a credential stored only in Streamlit secrets. GitHub states no release bandwidth limit, but availability/rate limits remain operational risks. | Viable fallback, especially if institutional policy disallows a model Hub. |
| C. Hugging Face Hub model repository | The Hub is designed for large ML files. Its documented recommendation is files below 200 GB; this 498.6 MB file is well within it. It has no per-model-repository size cap, subject to account storage policy. [Hub storage limits](https://huggingface.co/docs/hub/main/storage-limits), [Hub repositories](https://huggingface.co/docs/hub/en/repositories) | High reproducibility: upload the four selected-checkpoint files unchanged to a dedicated model repository, pin a full immutable commit SHA in the app, download only the required patterns, and independently verify the frozen weight SHA. `snapshot_download`/`hf_hub_download` cache downloaded files, avoiding repeated downloads while the Cloud instance remains alive. [Hub download guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/download) | Public model repo: no token required, public storage is best-effort and must be used responsibly. Private repo: token required; place it solely in Streamlit Community Cloud Secrets, never Git. [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) | **Recommended** for a public dissertation demonstration, subject to permission to publish the derived checkpoint and a post-deployment resource test. |
| D. Hosted inference endpoint/managed GPU service | Technically possible but changes the application architecture from local frozen-model loading to remote inference. | Adds vendor API, network dependency, endpoint configuration and potentially different runtime behaviour. It is unnecessary for the modest demonstration request. | Requires an API token and typically pay-as-you-go compute. | Not recommended: avoids a file download but adds service/deployment complexity and cost without research benefit. |

## Official documentation evidence

- GitHub normal-Git objects over 100 MB are rejected; Git LFS is the stated solution. GitHub Free/Pro LFS permits individual files up to 2 GB. [Repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits), [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- GitHub Free includes 10 GiB each of Git LFS storage and monthly bandwidth; the owner is charged for LFS downloads. [Git LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- GitHub release assets permit files below 2 GiB. [Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
- Community Cloud's stated approximate ceiling is 2.7 GB RAM and 50 GB storage; resource limits can make an app nonfunctional. [Cloud limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- The Hub client downloads to a local cache and supports a fixed `revision`; `allow_patterns` limits downloads to the checkpoint files needed. [Hub file download API](https://huggingface.co/docs/huggingface_hub/main/package_reference/file_download)

## Recommended solution: pinned Hugging Face model repository

Use one dedicated public Hugging Face **model** repository containing only the frozen `selected_checkpoint` files required by `from_pretrained`: `config.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`. Do not upload training checkpoints, data, predictions, WELFake material, or any secret.

After the owner manually uploads the unchanged files, record the full Hub commit SHA and the existing model SHA-256 in a small non-secret application configuration file. The application should call `snapshot_download(repo_id=..., revision=<full-commit-sha>, allow_patterns=[...])` into the normal Hugging Face cache; it must calculate the local `model.safetensors` SHA-256 and reject loading on mismatch. `st.cache_resource` should cache the instantiated tokenizer/model, while the Hub cache prevents repeat downloads on a warm instance. After Community Cloud hibernation, a new instance may need a fresh download; that is expected.

The classical joblib/vectorizer artefacts remain in the GitHub repository and are loaded lazily from repository-relative paths. The application performs inference only; it must not access WELFake or FakeNewsAMT at prediction time.

## Reproducibility controls

1. Preserve original selected-checkpoint files locally and never overwrite them.
2. Publish the exact model SHA-256 and file size in the model card and app configuration.
3. Pin the full immutable Hub commit SHA, never `main`/a floating revision.
4. Download only selected-checkpoint files and load with `local_files_only=True` after acquisition.
5. Log only non-sensitive operational status (for example, checksum success); never submitted article text.
6. Keep validated metrics as static values read from `final_results_master.csv`, not recalculated by the app.

## Security and privacy

A public model repository needs no credential. If the model must be private, use a least-privilege read token in Community Cloud's secret manager; do not commit `.streamlit/secrets.toml`, embed a token in a URL, or print it to logs. Streamlit advises storing deployment secrets in its settings rather than Git. [Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

The app should not retain user article text, write it to files, or expose paths, URLs with credentials, or model internals. Verify that distributing a fine-tuned checkpoint and the base-model licence is permitted before selecting public hosting.

## Cost and remaining risks

The 498.6 MB file is comfortably within Hub file guidance. Free public Hub storage is documented as best-effort, not a service-level guarantee; private free storage is documented as 100 GB. GitHub LFS free bandwidth may be consumed quickly by repeated cold downloads. Community Cloud resource limits and hibernation mean the cold-start download/load path must be tested before claiming deployment readiness. The model will run on CPU unless a different platform is selected, so response time may be material. These are deployment risks, not model changes.
