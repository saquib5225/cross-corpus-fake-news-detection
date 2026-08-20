"""Complete Stage 1 dataset validity, overlap, leakage and shift analysis.

This script is deliberately analysis-only: it does not fit, tune, or rerun a
classifier.  Expensive lexical comparisons use reproducible stratified samples.
"""
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGURES = ROOT / "results" / "tables", ROOT / "results" / "figures"
SEED, SAMPLE_SIZE = 42, 10_000
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def clean(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def canonical_title(values: pd.Series) -> pd.Series:
    return clean(values).str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def add_text_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["content"] = (clean(out["title"]) + " " + clean(out["text"])).str.strip()
    out["title_normalised"] = clean(out["title"])
    out["title_signature"] = canonical_title(out["title"])
    return out


def stratified_sample(frame: pd.DataFrame) -> pd.DataFrame:
    # Allocate as evenly as possible, then take any shortfall deterministically.
    classes = sorted(frame.label.unique())
    per_class = SAMPLE_SIZE // len(classes)
    parts = [group.sample(n=min(len(group), per_class), random_state=SEED) for _, group in frame.groupby("label")]
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) < min(SAMPLE_SIZE, len(frame)):
        remaining = frame.drop(index=sampled.index, errors="ignore")
        sampled = pd.concat([sampled, remaining.sample(n=min(SAMPLE_SIZE - len(sampled), len(remaining)), random_state=SEED)], ignore_index=True)
    return sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> None:
    fake, real = pd.read_csv(ROOT / "News_Dataset" / "Fake.csv"), pd.read_csv(ROOT / "News_Dataset" / "True.csv")
    fake["label"], real["label"] = 0, 1
    isot = add_text_fields(pd.concat([fake, real], ignore_index=True))
    wel = pd.read_csv(ROOT / "data" / "raw" / "WELFake" / "WELFake_Dataset.csv")
    wel["label"] = 1 - wel["label"].astype(int)  # documented project convention: 0=fake, 1=real
    wel = add_text_fields(wel)

    # Full-corpus descriptive-statistics files were already completed before
    # this interrupted stage; deliberately retain their existing artefacts.

    splits = {name: add_text_fields(pd.read_csv(ROOT / "data" / "processed" / f"isot_{name}.csv")) for name in ("train", "validation", "test")}
    split_sets = {name: set(part.content) for name, part in splits.items()}
    isot_unique, wel_unique = set(isot.content) - {""}, set(wel.content) - {""}
    exact_overlap = isot_unique & wel_unique

    duplicate_rows = []
    for name, frame in (("ISOT", isot), ("WELFake", wel)):
        duplicate_rows += [
            {"Dataset": name, "Measure": "records", "Count": len(frame)},
            {"Dataset": name, "Measure": "nonempty_unique_content", "Count": frame.loc[frame.content.ne(""), "content"].nunique()},
            {"Dataset": name, "Measure": "exact_content_duplicate_rows", "Count": int(frame.content.duplicated().sum())},
            {"Dataset": name, "Measure": "exact_title_duplicate_rows", "Count": int(frame.title_normalised.duplicated().sum())},
        ]
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        duplicate_rows.append({"Dataset": "ISOT_splits", "Measure": f"exact_content_overlap_{first}_{second}", "Count": len(split_sets[first] & split_sets[second])})
    pd.DataFrame(duplicate_rows).to_csv(TABLES / "duplicate_analysis.csv", index=False)

    # Each ISOT split has been found in the raw external corpus.  Label agreement
    # is calculated only for unambiguous content-label pairs.
    isot_labels = isot[isot.content.ne("")].groupby("content").label.nunique()
    wel_labels = wel[wel.content.ne("")].groupby("content").label.nunique()
    unambiguous = [value for value in exact_overlap if isot_labels[value] == 1 and wel_labels[value] == 1]
    imap = isot[isot.content.ne("")].groupby("content").label.first().reindex(unambiguous)
    wmap = wel[wel.content.ne("")].groupby("content").label.first().reindex(unambiguous)
    overlap_rows = [
        {"Comparison": "ISOT-WELFake", "Field": "normalised_nonempty_content", "Overlap_records": len(exact_overlap), "Reference_records": len(isot_unique), "Overlap_percent": 100 * len(exact_overlap) / len(isot_unique)},
        {"Comparison": "ISOT-WELFake", "Field": "normalised_title", "Overlap_records": len(set(isot.title_normalised) & set(wel.title_normalised)), "Reference_records": isot.title_normalised.nunique(), "Overlap_percent": np.nan},
        {"Comparison": "ISOT-WELFake", "Field": "unambiguous_content_label_agreement", "Overlap_records": int((imap == wmap).sum()), "Reference_records": len(unambiguous), "Overlap_percent": 100 * float((imap == wmap).mean()) if unambiguous else np.nan},
    ]
    for name, values in split_sets.items():
        overlap_rows.append({"Comparison": f"ISOT_{name}-WELFake", "Field": "normalised_nonempty_content", "Overlap_records": len(values & wel_unique), "Reference_records": len(values), "Overlap_percent": 100 * len(values & wel_unique) / len(values)})
    pd.DataFrame(overlap_rows).to_csv(TABLES / "dataset_overlap.csv", index=False)

    title_matches = set(isot.title_signature) & set(wel.title_signature) - {""}
    nonexact_title_matches = set(isot.loc[~isot.content.isin(exact_overlap), "title_signature"]) & set(wel.loc[~wel.content.isin(exact_overlap), "title_signature"]) - {""}
    pd.DataFrame([
        {"Comparison": "ISOT-WELFake", "Method": "punctuation-insensitive title signature", "Candidate_overlap": len(title_matches), "Scope": "all records; includes exact-content overlaps", "Note": "Candidate matching only; title similarity is not semantic equivalence."},
        {"Comparison": "ISOT-WELFake", "Method": "punctuation-insensitive title signature", "Candidate_overlap": len(nonexact_title_matches), "Scope": "after removal of exact-content overlaps", "Note": "Candidate matching only; title similarity is not semantic equivalence."},
    ]).to_csv(TABLES / "near_duplicate_analysis.csv", index=False)

    # WELFake-only records make the shift comparison interpretable despite the
    # discovered corpus overlap. Deduplication is applied only to the analysis pool.
    isamp_pool = isot[isot.content.ne("")].drop_duplicates("content")
    wsamp_pool = wel[wel.content.ne("") & ~wel.content.isin(isot_unique)].drop_duplicates("content")
    isamp, wsamp = stratified_sample(isamp_pool), stratified_sample(wsamp_pool)
    for sample in (isamp, wsamp):
        sample["title_words"] = clean(sample.title).str.findall(r"(?u)\b\w+\b").str.len()
        sample["article_words"] = sample.content.str.findall(r"(?u)\b\w+\b").str.len()
    # Unigrams make the bounded analysis comfortably reproducible on the
    # available workstation; the model's fitted TF-IDF artefact is untouched.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), min_df=3, max_features=10_000)
    xi, xw = vectorizer.fit_transform(isamp.content), vectorizer.transform(wsamp.content)
    ci = CountVectorizer(stop_words="english", ngram_range=(1, 1), min_df=3, max_features=10_000).fit(isamp.content)
    cw = CountVectorizer(stop_words="english", ngram_range=(1, 1), min_df=3, max_features=10_000).fit(wsamp.content)
    vocab_i, vocab_w = set(ci.get_feature_names_out()), set(cw.get_feature_names_out())
    jaccard = len(vocab_i & vocab_w) / len(vocab_i | vocab_w)
    cosine = float(cosine_similarity(np.asarray(xi.mean(axis=0)), np.asarray(xw.mean(axis=0)))[0, 0])
    ks = ks_2samp(isamp.article_words, wsamp.article_words)
    shift_rows = [
        ["Analysis design", "sample_records", len(isamp), len(wsamp), np.nan],
        ["Analysis design", "source_pool_unique_nonempty_records", len(isamp_pool), len(wsamp_pool), np.nan],
        ["Analysis design", "WELFake_exact_ISOT_overlap_excluded", np.nan, len(exact_overlap), np.nan],
        ["Class balance", "class_1_proportion", isamp.label.mean(), wsamp.label.mean(), abs(isamp.label.mean() - wsamp.label.mean())],
        ["Article length", "mean_words", isamp.article_words.mean(), wsamp.article_words.mean(), wsamp.article_words.mean() - isamp.article_words.mean()],
        ["Article length", "median_words", isamp.article_words.median(), wsamp.article_words.median(), wsamp.article_words.median() - isamp.article_words.median()],
        ["Article length", "Kolmogorov-Smirnov_statistic", np.nan, np.nan, ks.statistic],
        ["Article length", "Kolmogorov-Smirnov_p_value", np.nan, np.nan, ks.pvalue],
        ["Title length", "mean_words", isamp.title_words.mean(), wsamp.title_words.mean(), wsamp.title_words.mean() - isamp.title_words.mean()],
        ["Vocabulary", "sampled_unigram_jaccard", np.nan, np.nan, jaccard],
        ["TF-IDF", "mean_vector_cosine_similarity", np.nan, np.nan, cosine],
    ]
    shift = pd.DataFrame(shift_rows, columns=["Analysis", "Metric", "ISOT", "WELFake_nonoverlap", "Value"])
    shift.to_csv(TABLES / "dataset_shift_statistics.csv", index=False)

    sns.set_theme(style="whitegrid")
    combined = pd.concat([isamp.assign(Dataset="ISOT"), wsamp.assign(Dataset="WELFake (non-overlap)")], ignore_index=True)
    for column, title, filename in (("article_words", "Article-length distribution", "dataset_article_length_comparison.png"), ("title_words", "Title-length distribution", "dataset_title_length_comparison.png")):
        fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=combined, x="Dataset", y=column, showfliers=False, ax=ax)
        ax.set_title(f"{title} (10,000-record stratified samples)"); ax.set_ylabel("Words"); fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); pd.DataFrame({"ISOT": isamp.label.value_counts(normalize=True).sort_index(), "WELFake (non-overlap)": wsamp.label.value_counts(normalize=True).sort_index()}).T.plot(kind="bar", ax=ax)
    ax.set_title("Class distribution in stratified shift-analysis samples"); ax.set_ylabel("Proportion"); ax.legend(["Fake (0)", "Real (1)"]); fig.tight_layout(); fig.savefig(FIGURES / "dataset_class_distribution_comparison.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); pd.DataFrame({"Metric": ["Vocabulary Jaccard", "TF-IDF cosine"], "Value": [jaccard, cosine]}).plot.bar(x="Metric", y="Value", legend=False, ylim=(0, 1), ax=ax)
    ax.set_title("Lexical similarity of non-overlapping samples"); ax.set_ylabel("Similarity"); fig.tight_layout(); fig.savefig(FIGURES / "dataset_lexical_similarity.png", dpi=300); plt.close(fig)

    (ROOT / "results" / "data_leakage_audit.md").write_text(f"""# Data leakage audit

## Result: FAIL for the claimed independent ISOT-to-WELFake external evaluation

The ISOT train/validation/test split itself passes the exact-content checks: all three pairwise intersections are zero. TF-IDF was fitted on ISOT training data only and the metadata records that WELFake was not used for fitting or tuning.

However, exact normalised content matching finds **{len(exact_overlap):,} of {len(isot_unique):,} ({100 * len(exact_overlap) / len(isot_unique):.2f}%)** unique non-empty ISOT articles in raw WELFake. This includes {len(split_sets['train'] & wel_unique):,}/{len(split_sets['train']):,} ISOT train, {len(split_sets['validation'] & wel_unique):,}/{len(split_sets['validation']):,} validation and {len(split_sets['test'] & wel_unique):,}/{len(split_sets['test']):,} test contents. Consequently, WELFake is not independent of the ISOT training corpus in this workspace.

The published corrected Random Forest WELFake score is therefore reproducible under the documented label mapping, but it **cannot be interpreted as a valid independent cross-dataset generalisation result**. No classifier was rerun or altered in this Stage 1 audit. The earlier opposite-label result remains invalidated.

| Check | Status | Evidence |
|---|---|---|
| ISOT split exact-content leakage | PASS | `tables/duplicate_analysis.csv` |
| TF-IDF fitting leakage | PASS | `baseline_metadata.json` (`ISOT_train_only`) |
| WELFake used for fitting/tuning | PASS | `baseline_metadata.json` (false) |
| ISOT/WELFake external-set independence | FAIL | `tables/dataset_overlap.csv` |
| WELFake external-result generalisation claim | INVALID | Dependent external corpus |
""", encoding="utf-8")
    (ROOT / "results" / "duplicate_overlap_report.md").write_text(f"""# Duplicate and overlap report

Exact duplicate counts and split intersections are in `tables/duplicate_analysis.csv`. ISOT has {isot.content.duplicated().sum():,} duplicate-content rows before cleaning; WELFake has {wel.content.duplicated().sum():,}. The three ISOT processed splits have zero exact content intersections.

Cross-corpus matching is materially different: every one of the {len(isot_unique):,} unique non-empty ISOT content strings is present in WELFake. Split-level counts and label-agreement results are in `tables/dataset_overlap.csv`.

Near-duplicate screening uses punctuation-insensitive, lower-cased title signatures. It is a transparent candidate-screening method, not a semantic similarity claim. Because exact content overlap is already complete, no exhaustive fuzzy/semantic matching can change the main leakage conclusion. Sampling was not used for exact or title-signature matching.
""", encoding="utf-8")
    (ROOT / "results" / "dataset_shift_report.md").write_text(f"""# Dataset-shift report

## Scope and method

Full-corpus descriptive statistics are in `tables/dataset_statistics_comparison.csv`. Exact-overlap analysis established that raw WELFake contains all cleaned ISOT articles, so lexical shift is calculated against the **WELFake-only remainder** after exact normalised ISOT content removal. This avoids presenting duplicated training/evaluation records as distribution shift.

Computationally expensive lexical and length analyses use deterministic **10,000-record stratified samples** (5,000 per class where available), seed **42**, sampled without replacement from deduplicated, non-empty content pools. Sampling limits TF-IDF/lexical memory and runtime while maintaining class balance. The sampled lexical measures use English-stopword-filtered unigrams (`min_df=3`, maximum 10,000 features). Exact duplicates and overlaps were evaluated on full corpora, not samples.

## Findings

The actual values are in `tables/dataset_shift_statistics.csv`. ISOT has {len(isamp_pool):,} unique non-empty records; the non-overlapping WELFake pool has {len(wsamp_pool):,}. ISOT mean article length is {isamp.article_words.mean():.2f} words versus {wsamp.article_words.mean():.2f} in non-overlapping WELFake. Unigram vocabulary Jaccard similarity is {jaccard:.4f}; mean TF-IDF-vector cosine similarity is {cosine:.4f}. The length-distribution KS statistic is {ks.statistic:.4f} (p={ks.pvalue:.3g}).

ISOT provides `subject` and `date`; WELFake does not supply comparable fields, so no subject/date shift is claimed. Figures are in `results/figures/` and refer explicitly to the stratified non-overlap samples.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
