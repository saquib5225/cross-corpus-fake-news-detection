"""Stage 2A: frozen traditional-model evaluation on independent FakeNewsAMT.

This runner intentionally loads the already fitted ISOT-only TF-IDF vectorizer
and classifiers.  It never calls ``fit`` or uses FakeNewsAMT for any training,
tuning, feature selection, or model selection.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
PREDICTIONS = ROOT / "results" / "predictions"
FIGURES = ROOT / "results" / "figures" / "external_evaluation"
MODELS = ROOT / "models" / "traditional"
SEED = 42


def exact(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalised_body(value: object) -> str:
    return re.sub(r"\s+", " ", exact(value).casefold()).strip()


def normalised_title(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalised_body(value))).strip()


def split_title_body(value: object) -> tuple[str, str]:
    title, separator, body = exact(value).partition("\n\n")
    return title.strip(), body.strip() if separator else ""


def project_content(title: object, body: object) -> str:
    """Match the existing ISOT title + text representation without refitting."""
    return re.sub(r"\s+", " ", f"{exact(title)} {exact(body)}").strip()


def metrics(y_true: pd.Series, prediction: np.ndarray, probability_real: np.ndarray) -> tuple[dict, np.ndarray]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="macro", zero_division=0
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true, prediction, labels=[0, 1], zero_division=0
    )
    result = {
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": macro_precision,
        "Recall": macro_recall,
        "F1": macro_f1,
        "Macro_F1": macro_f1,
        "ROC_AUC": roc_auc_score(y_true, probability_real),
        "Fake_precision": per_precision[0], "Fake_recall": per_recall[0], "Fake_F1": per_f1[0], "Fake_support": int(per_support[0]),
        "Real_precision": per_precision[1], "Real_recall": per_recall[1], "Real_F1": per_f1[1], "Real_support": int(per_support[1]),
    }
    return result, confusion_matrix(y_true, prediction, labels=[0, 1])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small report table without an optional formatting dependency."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def load_external() -> pd.DataFrame:
    path = ROOT / "data" / "external_candidates" / "FakeNewsAMT" / "train-00000-of-00001.parquet"
    raw = pd.read_parquet(path).copy()
    title_body = raw["text"].map(split_title_body).tolist()
    raw["title"] = [pair[0] for pair in title_body]
    raw["body"] = [pair[1] for pair in title_body]
    raw["raw_label"] = raw["label"].astype(int)
    # Dataset card: 0=legit, 1=fake. Project convention: 0=fake, 1=real.
    raw["label"] = 1 - raw["raw_label"]
    external = raw[raw["body"].ne("")].copy().reset_index(names="external_row_id")
    external["content"] = [project_content(title, body) for title, body in zip(external["title"], external["body"])]
    if len(raw) != 480 or len(external) != 430 or external["label"].value_counts().to_dict() != {1: 240, 0: 190}:
        raise RuntimeError("Unexpected FakeNewsAMT version, label mapping, or usable-record count.")
    return external


def verify_independence(external: pd.DataFrame) -> dict:
    isot = pd.concat([pd.read_csv(ROOT / "News_Dataset" / "Fake.csv"),
                      pd.read_csv(ROOT / "News_Dataset" / "True.csv")], ignore_index=True)
    checks = {
        "exact_body_overlap": len(({exact(x) for x in isot["text"]} - {""}) & ({exact(x) for x in external["body"]} - {""})),
        "normalised_body_overlap": len(({normalised_body(x) for x in isot["text"]} - {""}) & ({normalised_body(x) for x in external["body"]} - {""})),
        "exact_title_overlap": len(({exact(x) for x in isot["title"]} - {""}) & ({exact(x) for x in external["title"]} - {""})),
        "normalised_title_overlap": len(({normalised_title(x) for x in isot["title"]} - {""}) & ({normalised_title(x) for x in external["title"]} - {""})),
    }
    if any(checks.values()):
        raise RuntimeError(f"External independence check failed: {checks}")
    return checks


def make_figures(results: pd.DataFrame, matrices: dict[str, np.ndarray], gaps: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    order = ["Naive Bayes", "Logistic Regression", "Random Forest"]
    display = results.copy()
    display["Evaluation"] = display["Evaluation"].map({"ISOT_test": "ISOT test", "FakeNewsAMT_external": "FakeNewsAMT"})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=display, x="Model", y="Accuracy", hue="Evaluation", order=order, ax=ax)
    ax.set_ylim(0, 1); ax.set_title("Accuracy: ISOT test versus independent FakeNewsAMT")
    ax.set_ylabel("Accuracy"); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(FIGURES / "isot_vs_fakenewsamt_accuracy.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=display, x="Model", y="Macro_F1", hue="Evaluation", order=order, ax=ax)
    ax.set_ylim(0, 1); ax.set_title("Macro-F1: in-domain versus independent external evaluation")
    ax.set_ylabel("Macro-F1"); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(FIGURES / "isot_vs_fakenewsamt_macro_f1.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    for row, model in enumerate(order):
        for column, evaluation in enumerate(["ISOT_test", "FakeNewsAMT_external"]):
            ax = axes[row, column]
            sns.heatmap(matrices[f"{model}|{evaluation}"], annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Predicted fake", "Predicted real"],
                        yticklabels=["Actual fake", "Actual real"], ax=ax)
            ax.set_title(f"{model}: {'ISOT test' if evaluation == 'ISOT_test' else 'FakeNewsAMT'}")
    fig.tight_layout(); fig.savefig(FIGURES / "confusion_matrices_isot_vs_fakenewsamt.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=gaps, x="Model", y="Macro_F1_gap_percentage_points", order=order, color="#c44e52", ax=ax)
    ax.axhline(0, color="black", linewidth=1); ax.set_title("External generalisation gap (ISOT Macro-F1 − FakeNewsAMT Macro-F1)")
    ax.set_ylabel("Macro-F1 gap (percentage points)"); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(FIGURES / "macro_f1_generalisation_gap.png", dpi=220); plt.close(fig)


def write_report(results: pd.DataFrame, gaps: pd.DataFrame, dataset_stats: pd.DataFrame, independence: dict, metadata: dict) -> None:
    formatted_results = results[["Model", "Evaluation", "Accuracy", "Precision", "Recall", "F1", "Macro_F1", "ROC_AUC",
                                 "Fake_F1", "Real_F1", "Fake_support", "Real_support"]].copy()
    for column in ["Accuracy", "Precision", "Recall", "F1", "Macro_F1", "ROC_AUC", "Fake_F1", "Real_F1"]:
        formatted_results[column] = formatted_results[column].map(lambda value: f"{value:.4f}")
    formatted_gaps = gaps[["Model", "ISOT_test_Macro_F1", "FakeNewsAMT_Macro_F1", "Macro_F1_gap_percentage_points", "Macro_F1_relative_change_percent"]].copy()
    for column in formatted_gaps.columns[1:]:
        formatted_gaps[column] = formatted_gaps[column].map(lambda value: f"{value:.4f}")
    report = f"""# Stage 2A - Frozen traditional-ML evaluation on FakeNewsAMT

## 1. Experimental objective

Measure how the existing ISOT-trained traditional classifiers generalise to a genuinely independent external corpus. This is a frozen-model evaluation; no model, vectorizer, feature selection, preprocessing choice, hyperparameter, or model-selection decision used FakeNewsAMT. WELFake is not used.

## 2. Dataset preparation and independence

FakeNewsAMT source: `polsci/fake-news`, `train-00000-of-00001.parquet`, SHA-256 `{metadata['external_sha256']}`. The release has 480 rows with labels `0=legit`, `1=fake`. The project mapping is therefore `legit -> 1 (real)` and `fake -> 0 (fake)`. The frozen external cohort contains 430 non-empty body records: 240 real and 190 fake. Fifty fake-labelled title-only rows with no body separator were excluded according to the pre-existing Stage 1 rule, not model performance.

Each stored text is split at its first blank line into title and body, then represented as title + body with whitespace collapsed, matching the established ISOT representation. The repeated full-corpus independence check returned: exact body={independence['exact_body_overlap']}, normalised body={independence['normalised_body_overlap']}, exact title={independence['exact_title_overlap']}, normalised title={independence['normalised_title_overlap']}.

## 3. Training, validation, and testing protocol

- Training: existing 31,280-record ISOT training split only.
- Validation: existing 3,910-record ISOT validation split only; no external data was used for selection.
- Testing: existing 3,911-record ISOT test set and the untouched 430-record FakeNewsAMT cohort.
- Feature extraction: the existing saved TF-IDF vectorizer (`ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `max_features=50,000`, lowercase), fitted on ISOT train only, was loaded and transformed external text without fitting.
- Seed: {SEED}. Evaluation uses saved model artefacts and deterministic preprocessing.

## 4. Models and metrics

Saved ISOT-only Naive Bayes, Logistic Regression and Random Forest models were evaluated. Metrics are accuracy, macro precision/recall/F1, ROC-AUC from the saved class-1 (real) probability, and class-specific precision/recall/F1. Confusion matrices use the order fake, real.

## 5. Results

{markdown_table(formatted_results)}

## 6. Generalisation gaps

Gap = ISOT-test Macro-F1 minus FakeNewsAMT Macro-F1. A positive gap and relative decrease denote external degradation relative to in-domain Macro-F1.

{markdown_table(formatted_gaps)}

## 7. Observations, limitations, and threats to validity

The results quantify transfer to an independent corpus rather than WELFake, whose overlap invalidated its generalisation claim. FakeNewsAMT is small, consists of short excerpts, and its fake class is crowdsourced; it is not representative of organic web misinformation. The external cohort also excludes 50 title-only fake records because no body is available, leaving modest class imbalance (240/190). Zero exact/normalised title/body matches reduce direct corpus-reuse risk but cannot rule out shared events, paraphrases, or source-level similarities. Results should therefore be interpreted as evidence for this specific external target, with uncertainty reported in any later dissertation analysis.

## 8. Reproducibility and validity decision

The run records model hashes, data hash, fixed seed, split sizes, preprocessing, metrics and saved prediction outputs in Stage 2A-specific files. No WELFake row was loaded; this runner contains no fitting operation. **Stage 2A is reproducible and complete.**
"""
    (ROOT / "results" / "baseline_external_evaluation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    for directory in (TABLES, PREDICTIONS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    external = load_external()
    independence = verify_independence(external)
    isot_test = pd.read_csv(ROOT / "data" / "processed" / "isot_test.csv")
    if len(isot_test) != 3911:
        raise RuntimeError("Unexpected ISOT test split size; refusing to evaluate.")

    vectorizer_path = MODELS / "tfidf_vectorizer.joblib"
    vectorizer = joblib.load(vectorizer_path)
    x_isot = vectorizer.transform(isot_test["content"])
    x_external = vectorizer.transform(external["content"])
    model_files = {"Naive Bayes": "naive_bayes.joblib", "Logistic Regression": "logistic_regression.joblib", "Random Forest": "random_forest.joblib"}
    rows, matrix_rows, matrices = [], [], {}
    metadata = {"seed": SEED, "tfidf_fitted_on": "ISOT_train_only", "external_used_for_training_or_tuning": False,
                "isot_train_records": 31280, "isot_validation_records": 3910, "isot_test_records": 3911,
                "fakenewsamt_released_records": 480, "fakenewsamt_evaluated_records": 430,
                "external_sha256": sha256(ROOT / "data" / "external_candidates" / "FakeNewsAMT" / "train-00000-of-00001.parquet"),
                "vectorizer_sha256": sha256(vectorizer_path), "independence_checks": independence,
                "models": {name: sha256(MODELS / filename) for name, filename in model_files.items()}}
    for model_name, filename in model_files.items():
        model = joblib.load(MODELS / filename)
        for evaluation, frame, features, output_stem in [("ISOT_test", isot_test, x_isot, "isot"),
                                                         ("FakeNewsAMT_external", external, x_external, "fakenewsamt")]:
            started = time.perf_counter()
            prediction = model.predict(features)
            probability_real = model.predict_proba(features)[:, 1]
            inference_seconds = time.perf_counter() - started
            result, matrix = metrics(frame["label"], prediction, probability_real)
            rows.append({"Model": model_name, "Evaluation": evaluation, "Records": len(frame),
                         "Inference_seconds": inference_seconds, **result})
            matrices[f"{model_name}|{evaluation}"] = matrix
            for actual_index, actual_name in enumerate(["fake", "real"]):
                for predicted_index, predicted_name in enumerate(["fake", "real"]):
                    matrix_rows.append({"Model": model_name, "Evaluation": evaluation,
                                        "Actual_class": actual_name, "Predicted_class": predicted_name,
                                        "Count": int(matrix[actual_index, predicted_index])})
            predictions = pd.DataFrame({"true_label": frame["label"], "prediction": prediction,
                                        "probability_real": probability_real})
            if evaluation == "FakeNewsAMT_external":
                predictions.insert(0, "external_row_id", frame["external_row_id"].to_numpy())
            predictions.to_csv(PREDICTIONS / f"stage2a_{model_name.lower().replace(' ', '_')}_{output_stem}.csv", index=False)

    results = pd.DataFrame(rows)
    gaps = results.pivot(index="Model", columns="Evaluation", values="Macro_F1").reset_index()
    gaps["Macro_F1_gap"] = gaps["ISOT_test"] - gaps["FakeNewsAMT_external"]
    gaps["Macro_F1_gap_percentage_points"] = 100 * gaps["Macro_F1_gap"]
    gaps["Macro_F1_relative_change_percent"] = 100 * gaps["Macro_F1_gap"] / gaps["ISOT_test"]
    gaps = gaps.rename(columns={"ISOT_test": "ISOT_test_Macro_F1", "FakeNewsAMT_external": "FakeNewsAMT_Macro_F1"})
    stats = pd.DataFrame([
        {"Dataset": "FakeNewsAMT", "Measure": "released_records", "Value": 480},
        {"Dataset": "FakeNewsAMT", "Measure": "title_only_excluded_records", "Value": 50},
        {"Dataset": "FakeNewsAMT", "Measure": "evaluated_nonempty_body_records", "Value": 430},
        {"Dataset": "FakeNewsAMT", "Measure": "fake_records_project_label_0", "Value": 190},
        {"Dataset": "FakeNewsAMT", "Measure": "real_records_project_label_1", "Value": 240},
        {"Dataset": "FakeNewsAMT", "Measure": "exact_body_overlap_with_ISOT", "Value": independence["exact_body_overlap"]},
        {"Dataset": "FakeNewsAMT", "Measure": "normalised_body_overlap_with_ISOT", "Value": independence["normalised_body_overlap"]},
        {"Dataset": "FakeNewsAMT", "Measure": "exact_title_overlap_with_ISOT", "Value": independence["exact_title_overlap"]},
        {"Dataset": "FakeNewsAMT", "Measure": "normalised_title_overlap_with_ISOT", "Value": independence["normalised_title_overlap"]},
    ])
    stats.to_csv(TABLES / "fakenewsamt_dataset_statistics.csv", index=False)
    results.to_csv(TABLES / "baseline_external_evaluation.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(TABLES / "baseline_confusion_matrices.csv", index=False)
    gaps.to_csv(TABLES / "baseline_generalisation_gap.csv", index=False)
    (TABLES / "baseline_external_evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_figures(results, matrices, gaps)
    write_report(results, gaps, stats, independence, metadata)


if __name__ == "__main__":
    main()
