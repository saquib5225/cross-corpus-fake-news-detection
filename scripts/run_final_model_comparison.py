"""Stage 3 analysis-only comparison from frozen model outputs.

This script reads preserved predictions and metrics, creates new Stage 3
artefacts, and never fits, tunes, evaluates, or modifies a model.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import scipy
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures" / "final_comparison"
PREDICTIONS = RESULTS / "predictions"
ROBERTA = RESULTS / "roberta"
MODELS = ["Naive Bayes", "Logistic Regression", "Random Forest", "RoBERTa"]
FILES = {
    "Naive Bayes": {
        "ISOT_test": PREDICTIONS / "naive_bayes_isot.csv",
        "FakeNewsAMT_external": PREDICTIONS / "stage2a_naive_bayes_fakenewsamt.csv",
    },
    "Logistic Regression": {
        "ISOT_test": PREDICTIONS / "logistic_regression_isot.csv",
        "FakeNewsAMT_external": PREDICTIONS / "stage2a_logistic_regression_fakenewsamt.csv",
    },
    "Random Forest": {
        "ISOT_test": PREDICTIONS / "random_forest_isot.csv",
        "FakeNewsAMT_external": PREDICTIONS / "stage2a_random_forest_fakenewsamt.csv",
    },
    "RoBERTa": {
        "ISOT_test": ROBERTA / "predictions_isot_test.csv",
        "FakeNewsAMT_external": ROBERTA / "predictions_fakenewsamt.csv",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, columns].copy()
    text = [[str(x) for x in row] for row in view.itertuples(index=False, name=None)]
    widths = [len(c) for c in columns]
    for row in text:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    row = lambda values: "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths)) + " |"
    return "\n".join([row(columns), row(["-" * w for w in widths]), *[row(r) for r in text]])


def load_predictions(model: str, evaluation: str) -> pd.DataFrame:
    frame = pd.read_csv(FILES[model][evaluation])
    needed = {"true_label", "prediction", "probability_real"}
    if not needed.issubset(frame):
        raise ValueError(f"{FILES[model][evaluation]} lacks {needed - set(frame)}")
    if evaluation == "FakeNewsAMT_external":
        if "external_row_id" not in frame:
            raise ValueError(f"{FILES[model][evaluation]} lacks external_row_id")
        frame = frame.sort_values("external_row_id").reset_index(drop=True)
    return frame


def metric_row(model: str, evaluation: str, frame: pd.DataFrame) -> dict:
    y, pred, score = frame.true_label.to_numpy(), frame.prediction.to_numpy(), frame.probability_real.to_numpy()
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "Model": model, "Evaluation": evaluation, "Records": len(frame),
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "Macro_F1": f1_score(y, pred, average="macro", zero_division=0),
        "ROC_AUC": roc_auc_score(y, score),
        "Fake_precision": precision_score(y, pred, pos_label=0, zero_division=0),
        "Fake_recall": recall_score(y, pred, pos_label=0, zero_division=0),
        "Real_precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "Real_recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "Correct": int(np.sum(y == pred)), "Incorrect": int(np.sum(y != pred)),
        "True_negative_fake": int(cm[0, 0]), "False_positive_real": int(cm[0, 1]),
        "False_negative_real": int(cm[1, 0]), "True_positive_real": int(cm[1, 1]),
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def mcnemar(model: str, evaluation: str, reference: pd.DataFrame, comparator: pd.DataFrame) -> dict:
    # b: RoBERTa correct / comparator incorrect; c: comparator correct / RoBERTa incorrect.
    y = reference.true_label.to_numpy()
    r_correct = reference.prediction.to_numpy() == y
    c_correct = comparator.prediction.to_numpy() == y
    b = int(np.sum(r_correct & ~c_correct)); c = int(np.sum(~r_correct & c_correct))
    discordant = b + c
    p = binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue if discordant else 1.0
    return {
        "Evaluation": evaluation, "Reference_model": "RoBERTa", "Comparator_model": model,
        "N": len(y), "RoBERTa_only_correct": b, "Comparator_only_correct": c,
        "Discordant_pairs": discordant, "Exact_McNemar_p": p,
        "Paired_accuracy_difference_RoBERTa_minus_comparator": float(r_correct.mean() - c_correct.mean()),
        "Odds_ratio_RoBERTa_only_vs_comparator_only": (b / c) if c else (float("inf") if b else float("nan")),
        "Null_hypothesis": "Equal probability of correctness on discordant paired observations",
    }


def plot_bar(values: pd.DataFrame, column: str, title: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(values.Model, values[column], color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("Model")
    ax.set_ylim(0, 1 if "Macro_F1" in column else max(values[column].max() * 1.1, 1))
    ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=240); plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    frames = {(m, e): load_predictions(m, e) for m in MODELS for e in ("ISOT_test", "FakeNewsAMT_external")}
    for evaluation in ("ISOT_test", "FakeNewsAMT_external"):
        reference = frames[("RoBERTa", evaluation)]
        for model in MODELS[:-1]:
            current = frames[(model, evaluation)]
            if not np.array_equal(reference.true_label.to_numpy(), current.true_label.to_numpy()):
                raise ValueError(f"Labels are not aligned for {model} / {evaluation}")
            if evaluation == "FakeNewsAMT_external" and not np.array_equal(reference.external_row_id.to_numpy(), current.external_row_id.to_numpy()):
                raise ValueError(f"External row IDs are not aligned for {model}")

    metrics = pd.DataFrame([metric_row(m, e, frames[(m, e)]) for m in MODELS for e in ("ISOT_test", "FakeNewsAMT_external")])
    isot = metrics[metrics.Evaluation.eq("ISOT_test")].set_index("Model")
    ext = metrics[metrics.Evaluation.eq("FakeNewsAMT_external")].set_index("Model")
    comparison = pd.DataFrame({"Model": MODELS})
    for label, source in (("ISOT", isot), ("FakeNewsAMT", ext)):
        for metric in ("Accuracy", "Precision", "Recall", "F1", "Macro_F1", "ROC_AUC"):
            comparison[f"{label} {metric}"] = [source.loc[m, metric] for m in MODELS]
    comparison["Generalisation Gap"] = comparison["ISOT Macro_F1"] - comparison["FakeNewsAMT Macro_F1"]
    comparison.to_csv(TABLES / "final_model_comparison.csv", index=False)

    generalisation = comparison[["Model", "ISOT Macro_F1", "FakeNewsAMT Macro_F1", "Generalisation Gap"]].copy()
    generalisation["Generalisation Gap percentage points"] = 100 * generalisation["Generalisation Gap"]
    generalisation["Relative Macro_F1 decrease percent"] = 100 * generalisation["Generalisation Gap"] / generalisation["ISOT Macro_F1"]
    generalisation.to_csv(TABLES / "final_generalisation_analysis.csv", index=False)

    errors = metrics[["Model", "Evaluation", "Records", "Correct", "Incorrect", "True_negative_fake", "False_positive_real", "False_negative_real", "True_positive_real", "Fake_precision", "Fake_recall", "Real_precision", "Real_recall"]]
    errors.to_csv(TABLES / "error_comparison.csv", index=False)

    tests = [mcnemar(m, e, frames[("RoBERTa", e)], frames[(m, e)]) for e in ("ISOT_test", "FakeNewsAMT_external") for m in MODELS[:-1]]
    statistics = pd.DataFrame(tests)
    statistics["Holm_adjusted_p_across_6_tests"] = holm_adjust(statistics.Exact_McNemar_p.tolist())
    statistics["Statistically_significant_at_0.05_after_Holm"] = statistics.Holm_adjusted_p_across_6_tests < 0.05
    statistics.to_csv(TABLES / "statistical_model_comparison.csv", index=False)

    plot_bar(comparison, "ISOT Macro_F1", "ISOT test Macro-F1 by model", "Macro-F1", "isot_macro_f1_comparison.png")
    plot_bar(comparison, "FakeNewsAMT Macro_F1", "FakeNewsAMT Macro-F1 by model", "Macro-F1", "fakenewsamt_macro_f1_comparison.png")
    plot_bar(generalisation.rename(columns={"Generalisation Gap percentage points": "Gap"}), "Gap", "Macro-F1 generalisation gap", "Percentage points", "generalisation_gap_comparison.png")
    long = pd.concat([comparison[["Model", "ISOT Macro_F1"]].rename(columns={"ISOT Macro_F1": "Macro_F1"}).assign(Evaluation="ISOT test"), comparison[["Model", "FakeNewsAMT Macro_F1"]].rename(columns={"FakeNewsAMT Macro_F1": "Macro_F1"}).assign(Evaluation="FakeNewsAMT")])
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODELS)); width = 0.36
    a = long[long.Evaluation.eq("ISOT test")].set_index("Model").loc[MODELS, "Macro_F1"]
    b = long[long.Evaluation.eq("FakeNewsAMT")].set_index("Model").loc[MODELS, "Macro_F1"]
    ax.bar(x-width/2, a, width, label="ISOT test"); ax.bar(x+width/2, b, width, label="FakeNewsAMT")
    ax.set_xticks(x, MODELS); ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1"); ax.set_title("In-domain and external Macro-F1"); ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "isot_vs_fakenewsamt_macro_f1.png", dpi=240); plt.close(fig)
    for model in MODELS:
        cm = confusion_matrix(frames[(model, "FakeNewsAMT_external")].true_label, frames[(model, "FakeNewsAMT_external")].prediction, labels=[0,1])
        fig, ax = plt.subplots(figsize=(4.8, 4)); im=ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
        for i in range(2):
            for j in range(2): ax.text(j, i, str(cm[i,j]), ha="center", va="center")
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["fake","real"], yticklabels=["fake","real"], xlabel="Predicted", ylabel="Actual", title=f"{model}: FakeNewsAMT confusion matrix")
        fig.tight_layout(); fig.savefig(FIGURES / f"fakenewsamt_confusion_{model.lower().replace(' ','_')}.png", dpi=240); plt.close(fig)
    for evaluation, title, filename in (("ISOT_test", "ISOT test ROC curves", "isot_roc_curves.png"), ("FakeNewsAMT_external", "FakeNewsAMT ROC curves", "fakenewsamt_roc_curves.png")):
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        for model in MODELS:
            frame=frames[(model,evaluation)]; fpr,tpr,_=roc_curve(frame.true_label,frame.probability_real); ax.plot(fpr,tpr,label=f"{model} (AUC={roc_auc_score(frame.true_label,frame.probability_real):.3f})")
        ax.plot([0,1],[0,1],"--",color="grey",label="Chance"); ax.set(xlabel="False positive rate",ylabel="True positive rate",title=title,xlim=(0,1),ylim=(0,1)); ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / filename,dpi=240); plt.close(fig)
    pr = pd.concat([comparison[["Model","ISOT Precision","ISOT Recall"]].rename(columns={"ISOT Precision":"Precision","ISOT Recall":"Recall"}).assign(Evaluation="ISOT test"), comparison[["Model","FakeNewsAMT Precision","FakeNewsAMT Recall"]].rename(columns={"FakeNewsAMT Precision":"Precision","FakeNewsAMT Recall":"Recall"}).assign(Evaluation="FakeNewsAMT")])
    fig, ax=plt.subplots(figsize=(8,5)); x=np.arange(len(MODELS)); w=.2
    for offset, (e, metric, color) in enumerate((("ISOT test","Precision","#4C78A8"),("ISOT test","Recall","#72B7B2"),("FakeNewsAMT","Precision","#F58518"),("FakeNewsAMT","Recall","#E45756"))):
        ax.bar(x+(offset-1.5)*w,pr[pr.Evaluation.eq(e)].set_index("Model").loc[MODELS,metric],w,label=f"{e} {metric}",color=color)
    ax.set(xticks=x,xticklabels=MODELS,ylim=(0,1),ylabel="Macro score",title="Macro precision and recall"); ax.legend(fontsize=8,ncol=2); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "precision_recall_comparison.png",dpi=240); plt.close(fig)

    report = "# Statistical model comparison\n\n" + "## Methods\n\nExact two-sided McNemar tests compare RoBERTa with each baseline on the same labelled observations. The null hypothesis is equal probabilities of correctness on discordant paired observations. This is appropriate for paired binary correctness outcomes, but it tests accuracy differences rather than Macro-F1 differences. Six planned tests (three baselines × two datasets) use Holm correction. The 430-item FakeNewsAMT sample limits power and external scope. No causal conclusion follows from significance.\n\n## Results\n\n" + markdown_table(statistics.assign(Exact_McNemar_p=statistics.Exact_McNemar_p.map(repr), Holm_adjusted_p_across_6_tests=statistics.Holm_adjusted_p_across_6_tests.map(repr)), ["Evaluation","Comparator_model","RoBERTa_only_correct","Comparator_only_correct","Discordant_pairs","Exact_McNemar_p","Holm_adjusted_p_across_6_tests","Statistically_significant_at_0.05_after_Holm"]) + "\n\nNo bootstrap procedure was added: the exact paired test already answers the planned paired-correctness question, while unpaired confidence intervals would not establish model-to-model superiority.\n"
    (RESULTS / "statistical_analysis_report.md").write_text(report, encoding="utf-8")
    interpretation = "# Final Model Comparison\n\n" + "## Comparison\n\n" + markdown_table(comparison, ["Model","ISOT Macro_F1","FakeNewsAMT Macro_F1","Generalisation Gap"]) + "\n\n" + "RoBERTa has the highest ISOT-test Macro-F1 (0.9997424774746806). Naive Bayes has the highest FakeNewsAMT Macro-F1 (0.5709699809127191) and the smallest Macro-F1 gap (0.3875579804125109; 38.75579804125109 percentage points). RoBERTa does not improve external Macro-F1 over the baselines. In-domain ordering does not correspond to external ordering: the strongest ISOT score has a lower external Macro-F1 than Naive Bayes.\n\n" + "The large drops quantify cross-corpus sensitivity and are consistent with dataset-specific learning, but do not establish that any architecture is universally superior or inferior. FakeNewsAMT has 430 usable records and is a crowdsourced-excerpt corpus; conclusions are limited to this independent cohort and label/source construction. These results should inform cautious deployment: strong in-domain validation alone is insufficient evidence of cross-corpus robustness.\n\n" + "The RoBERTa model was selected by ISOT validation, frozen, and then evaluated on ISOT test and FakeNewsAMT. FakeNewsAMT did not guide training, tuning, stopping, or checkpoint selection; WELFake was excluded.\n"
    (RESULTS / "final_model_comparison_report.md").write_text(interpretation, encoding="utf-8")
    metadata = {"analysis_script": str(Path(__file__).relative_to(ROOT)), "timestamp_utc": datetime.now(timezone.utc).isoformat(), "models": MODELS, "datasets": {"ISOT_test": 3911, "FakeNewsAMT_external": 430}, "metrics": "macro-averaged precision, recall and F1; ROC-AUC from probability_real; generalisation gap = ISOT test Macro-F1 - FakeNewsAMT Macro-F1", "statistical_methods": "six exact two-sided McNemar tests with Holm correction; RoBERTa versus each baseline on paired prediction files", "random_seeds": {"RoBERTa": 42, "bootstrap": None}, "software": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__}, "source_files_sha256": {str(p.relative_to(ROOT)): sha256(p) for paths in FILES.values() for p in paths.values()}, "selected_checkpoint_sha256": sha256(ROBERTA / "selected_checkpoint" / "model.safetensors"), "welfake_used": False, "training_or_evaluation_rerun": False}
    (RESULTS / "final_comparison_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    audit = "# Stage 3 Final Integrity Audit\n\nThis Stage 3 script is analysis-only: it reads frozen prediction files and metrics and writes new comparison artefacts. It does not import training code, fit models, change predictions, change frozen metrics, or access WELFake. All external analyses use the existing FakeNewsAMT prediction files, which were produced after model selection. RoBERTa's selected checkpoint is not modified; its SHA-256 is recorded in `final_comparison_metadata.json`. Stage 1 and Stage 2A source files are treated as read-only inputs.\n\nIntegrity checks passed: labels are aligned across models for ISOT test; FakeNewsAMT labels and external row IDs are aligned across models; no WELFake file is in the source list.\n"
    (RESULTS / "STAGE3_FINAL_INTEGRITY_AUDIT.md").write_text(audit, encoding="utf-8")


if __name__ == "__main__":
    main()
