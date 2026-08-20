"""Stage 2B: leakage-safe RoBERTa training and final external evaluation.

ISOT train is the sole fitting set and ISOT validation is the sole model-
selection set.  The FakeNewsAMT file is deliberately not opened until after
the selected checkpoint has been restored for final evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import signal
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "roberta"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures" / "roberta"
MODEL_DIR = OUT / "selected_checkpoint"
SEED = 42
MODEL_NAME = "roberta-base"
PRETRAINED_SOURCE = ROOT / "models" / "roberta-base"
MAX_LENGTH = 64
PER_DEVICE_BATCH = 2
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 1
CHECKPOINT_INTERVAL_STEPS = 500
GRADIENT_CHECKPOINTING = False
LOG_PATH = OUT / "stage2b_training.log"
ERROR_PATH = OUT / "stage2b_error.log"
CHECKPOINT_ROOT = OUT / "checkpoints"


class FlushLogger:
    """Small line logger that flushes every write and survives abrupt exits."""
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)
    def write(self, message: str):
        line = f"{datetime.now(timezone.utc).isoformat()} | {message}\n"
        self.handle.write(line); self.handle.flush()
    def close(self): self.handle.close()


def unique_run_config_path() -> Path:
    base = OUT / "stage2b_run_config.json"
    if not base.exists(): return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = OUT / f"stage2b_run_config_{stamp}.json"
    suffix = 1
    while candidate.exists():
        candidate = OUT / f"stage2b_run_config_{stamp}_{suffix}.json"; suffix += 1
    return candidate


def gpu_snapshot(device: torch.device | None = None) -> dict:
    result = {"cuda_available": bool(torch.cuda.is_available())}
    if result["cuda_available"]:
        index = device.index if device is not None and device.index is not None else torch.cuda.current_device()
        result.update({"gpu_name": torch.cuda.get_device_name(index), "allocated_mb": round(torch.cuda.memory_allocated(index)/1024**2, 2),
                       "reserved_mb": round(torch.cuda.memory_reserved(index)/1024**2, 2), "peak_allocated_mb": round(torch.cuda.max_memory_allocated(index)/1024**2, 2)})
    return result


def _checkpoint_files_valid(path: Path) -> bool:
    return path.is_dir() and (path / "model_state.pt").exists() and (path / "trainer_state.pt").exists() and (path / "checkpoint_metadata.json").exists()


def validate_checkpoint(path: Path) -> bool:
    if not _checkpoint_files_valid(path): return False
    try:
        metadata = json.loads((path / "checkpoint_metadata.json").read_text(encoding="utf-8"))
        state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
        required = {"epoch", "batch_index", "global_step", "rng_state", "optimizer_state", "scheduler_state", "scaler_state"}
        return required.issubset(state) and metadata.get("global_step") == state["global_step"]
    except Exception:
        return False


def find_latest_valid_checkpoint(root: Path = CHECKPOINT_ROOT) -> Path | None:
    candidates = sorted(root.glob("checkpoint-step-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1, reverse=True)
    return next((p for p in candidates if validate_checkpoint(p)), None)


def save_checkpoint_atomic(model, tokenizer, optimizer, scheduler, scaler, epoch: int, batch_index: int, global_step: int, logger: FlushLogger, device: torch.device) -> Path:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    final = CHECKPOINT_ROOT / f"checkpoint-step-{global_step}"
    temporary = CHECKPOINT_ROOT / f".checkpoint-step-{global_step}.tmp-{os.getpid()}"
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary); tokenizer.save_pretrained(temporary)
    torch.save(model.state_dict(), temporary / "model_state.pt")
    state = {"epoch": epoch, "batch_index": batch_index, "global_step": global_step, "rng_state": torch.get_rng_state(),
             "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
             "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict()}
    torch.save(state, temporary / "trainer_state.pt")
    (temporary / "checkpoint_metadata.json").write_text(json.dumps({"global_step": global_step, "epoch": epoch, "batch_index": batch_index, "gpu": gpu_snapshot(device)}, indent=2), encoding="utf-8")
    if final.exists(): shutil.rmtree(final)
    os.replace(temporary, final)
    if not validate_checkpoint(final): raise RuntimeError(f"Atomic checkpoint validation failed: {final}")
    logger.write(f"CHECKPOINT_SAVED path={final} epoch={epoch} batch_index={batch_index} global_step={global_step} gpu={gpu_snapshot(device)}")
    return final


def load_resume_state(path: Path, model, optimizer, scheduler, scaler, device: torch.device) -> dict:
    if not validate_checkpoint(path): raise ValueError(f"Invalid or incomplete checkpoint: {path}")
    model.load_state_dict(torch.load(path / "model_state.pt", map_location=device, weights_only=True))
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer_state"]); scheduler.load_state_dict(state["scheduler_state"]); scaler.load_state_dict(state["scaler_state"])
    torch.set_rng_state(state["rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None: torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    random.setstate(state["python_rng_state"]); np.random.set_state(state["numpy_rng_state"])
    return state


def register_signals(logger: FlushLogger):
    def handler(signum, _frame):
        logger.write(f"TERMINATION_SIGNAL signal={signum}; hard OS/GPU termination cannot be caught here")
        raise KeyboardInterrupt(f"received signal {signum}")
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name): signal.signal(getattr(signal, name), handler)


def self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="stage2b_hardening_") as temporary:
        root = Path(temporary); global CHECKPOINT_ROOT
        old_root = CHECKPOINT_ROOT; CHECKPOINT_ROOT = root / "checkpoints"; CHECKPOINT_ROOT.mkdir()
        logger = FlushLogger(root / "training.log"); logger.write("SELF_TEST_START")
        class DummyModel(torch.nn.Linear):
            def save_pretrained(self, destination):
                Path(destination, "config.json").write_text("{}", encoding="utf-8")
        class DummyTokenizer:
            def save_pretrained(self, destination): Path(destination, "tokenizer_config.json").write_text("{}", encoding="utf-8")
        tiny = DummyModel(3, 2); opt = AdamW(tiny.parameters(), lr=1e-3); sch = get_linear_schedule_with_warmup(opt, 0, 2); scale = torch.amp.GradScaler("cuda", enabled=False)
        path = save_checkpoint_atomic(tiny, DummyTokenizer(), opt, sch, scale, 1, 8, 8, logger, torch.device("cpu"))
        assert validate_checkpoint(path) and find_latest_valid_checkpoint(CHECKPOINT_ROOT) == path
        target = torch.nn.Linear(3, 2); target_opt = AdamW(target.parameters(), lr=1e-3); target_sch = get_linear_schedule_with_warmup(target_opt, 0, 2)
        resumed = load_resume_state(path, target, target_opt, target_sch, scale, torch.device("cpu"))
        assert resumed["global_step"] == 8 and "CHECKPOINT_SAVED" in (root / "training.log").read_text(encoding="utf-8")
        logger.close(); CHECKPOINT_ROOT = old_root
    print("Stage 2B hardening self-test passed; no training or external data access performed.")


def seed_everything() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def content_frame(path: Path, expected_count: int, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"title", "text", "content", "label"}
    if len(frame) != expected_count or not required.issubset(frame.columns):
        raise RuntimeError(f"Unexpected {split} split schema or size.")
    if set(frame["label"].unique()) != {0, 1} or frame["content"].isna().any() or (frame["content"].str.strip() == "").any():
        raise RuntimeError(f"Invalid labels or empty content in {split}.")
    return frame


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: AutoTokenizer):
        self.labels = frame["label"].astype(int).to_list()
        self.rows = tokenizer(frame["content"].astype(str).to_list(), truncation=True, max_length=MAX_LENGTH)
    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.rows.items()}
        item["labels"] = self.labels[index]
        return item


def metric_bundle(y: np.ndarray, prediction: np.ndarray, prob_real: np.ndarray) -> tuple[dict, np.ndarray]:
    macro = precision_recall_fscore_support(y, prediction, average="macro", zero_division=0)
    cls = precision_recall_fscore_support(y, prediction, labels=[0, 1], zero_division=0)
    return ({"Accuracy": accuracy_score(y, prediction), "Precision": macro[0], "Recall": macro[1],
             "F1": macro[2], "Macro_F1": macro[2], "ROC_AUC": roc_auc_score(y, prob_real),
             "Fake_precision": cls[0][0], "Fake_recall": cls[1][0], "Fake_F1": cls[2][0], "Fake_support": int(cls[3][0]),
             "Real_precision": cls[0][1], "Real_recall": cls[1][1], "Real_F1": cls[2][1], "Real_support": int(cls[3][1])},
            confusion_matrix(y, prediction, labels=[0, 1]))


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); labels, pred, probs = [], [], []
    for batch in loader:
        y = batch.pop("labels").numpy()
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        probability = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
        labels.extend(y); probs.extend(probability); pred.extend((probability >= 0.5).astype(int))
    return np.asarray(labels), np.asarray(pred), np.asarray(probs)


def exact(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").strip()


def normal_body(value: object) -> str: return re.sub(r"\s+", " ", exact(value).casefold()).strip()
def normal_title(value: object) -> str: return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normal_body(value))).strip()


def load_external_after_selection() -> tuple[pd.DataFrame, dict]:
    """Open FakeNewsAMT only after ISOT validation has selected the checkpoint."""
    source = ROOT / "data" / "external_candidates" / "FakeNewsAMT" / "train-00000-of-00001.parquet"
    raw = pd.read_parquet(source).copy()
    def parse(value):
        title, separator, body = exact(value).partition("\n\n")
        return title.strip(), body.strip() if separator else ""
    pairs = raw["text"].map(parse).to_list()
    raw["title"] = [x[0] for x in pairs]; raw["body"] = [x[1] for x in pairs]
    raw["raw_label"] = raw["label"].astype(int)
    raw["label"] = 1 - raw["raw_label"]  # documented card mapping: legit=0, fake=1
    external = raw[raw["body"].ne("")].copy().reset_index(names="external_row_id")
    external["content"] = (external["title"] + " " + external["body"]).str.replace(r"\s+", " ", regex=True).str.strip()
    counts = external["label"].value_counts().to_dict()
    if len(raw) != 480 or len(external) != 430 or counts != {1: 240, 0: 190}:
        raise RuntimeError("Unexpected FakeNewsAMT source, cohort, or documented label mapping.")
    isot = pd.concat([pd.read_csv(ROOT / "News_Dataset" / "Fake.csv"), pd.read_csv(ROOT / "News_Dataset" / "True.csv")], ignore_index=True)
    checks = {
        "exact_body_overlap": len(({exact(x) for x in isot.text} - {""}) & ({exact(x) for x in external.body} - {""})),
        "normalised_body_overlap": len(({normal_body(x) for x in isot.text} - {""}) & ({normal_body(x) for x in external.body} - {""})),
        "exact_title_overlap": len(({exact(x) for x in isot.title} - {""}) & ({exact(x) for x in external.title} - {""})),
        "normalised_title_overlap": len(({normal_title(x) for x in isot.title} - {""}) & ({normal_title(x) for x in external.title} - {""})),
    }
    if any(checks.values()): raise RuntimeError(f"FakeNewsAMT independence check failed: {checks}")
    checks["source_sha256"] = sha256(source)
    return external, checks


def save_predictions(path: Path, ids, y, pred, prob):
    frame = pd.DataFrame({"true_label": y, "prediction": pred, "probability_real": prob})
    if ids is not None: frame.insert(0, "external_row_id", ids)
    frame.to_csv(path, index=False)


def figure_outputs(results: pd.DataFrame, matrices: dict, comparisons: pd.DataFrame, gaps: pd.DataFrame):
    sns.set_theme(style="whitegrid", context="talk")
    r = results.copy(); r["Evaluation"] = r["Evaluation"].map({"ISOT_test": "ISOT test", "FakeNewsAMT_external": "FakeNewsAMT"})
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=r, x="Evaluation", y="Macro_F1", hue="Evaluation", legend=False, ax=ax)
    ax.set_ylim(0, 1); ax.set_title("RoBERTa Macro-F1: ISOT test versus FakeNewsAMT"); ax.set_xlabel(""); fig.tight_layout(); fig.savefig(FIGURES / "roberta_isot_vs_fakenewsamt_macro_f1.png", dpi=220); plt.close(fig)
    for key, title, filename in [("ISOT_test", "RoBERTa confusion matrix: ISOT test", "roberta_confusion_matrix_isot.png"), ("FakeNewsAMT_external", "RoBERTa confusion matrix: FakeNewsAMT", "roberta_confusion_matrix_fakenewsamt.png")]:
        fig, ax = plt.subplots(figsize=(6, 5)); sns.heatmap(matrices[key], annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Predicted fake", "Predicted real"], yticklabels=["Actual fake", "Actual real"], ax=ax)
        ax.set_title(title); fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6)); order=["Naive Bayes", "Logistic Regression", "Random Forest", "RoBERTa"]
    sns.barplot(data=comparisons, x="Model", y="FakeNewsAMT_Macro_F1", order=order, ax=ax); ax.set_ylim(0, 1); ax.set_title("External Macro-F1 on independent FakeNewsAMT"); ax.set_ylabel("Macro-F1"); ax.set_xlabel(""); fig.tight_layout(); fig.savefig(FIGURES / "model_comparison_external_macro_f1.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=gaps, x="Model", y="Macro_F1_gap_percentage_points", order=order, color="#c44e52", ax=ax)
    ax.axhline(0, color="black", linewidth=1); ax.set_title("In-domain to external Macro-F1 generalisation gap"); ax.set_ylabel("Gap (percentage points)"); ax.set_xlabel(""); fig.tight_layout(); fig.savefig(FIGURES / "generalisation_gap_model_comparison.png", dpi=220); plt.close(fig)


def main(resume_from: str | None = None, resume_latest: bool = False):
    for d in (OUT, TABLES, FIGURES, MODEL_DIR): d.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    logger = FlushLogger(LOG_PATH); register_signals(logger)
    run_config = {"model_name": MODEL_NAME, "pretrained_source": str(PRETRAINED_SOURCE), "tokenizer": MODEL_NAME, "seed": SEED,
                  "dataset": "ISOT only until final evaluation", "max_length": MAX_LENGTH, "per_device_batch_size": PER_DEVICE_BATCH,
                  "gradient_accumulation_steps": GRADIENT_ACCUMULATION, "learning_rate": LEARNING_RATE, "epochs": EPOCHS,
                  "optimizer": "AdamW", "scheduler": "linear", "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
                  "mixed_precision": "fp16", "gradient_checkpointing": GRADIENT_CHECKPOINTING, "checkpoint_interval_steps": CHECKPOINT_INTERVAL_STEPS,
                  "selection_criterion": "ISOT validation Macro-F1", "external_opened_after_selection": True,
                  "started_utc": datetime.now(timezone.utc).isoformat()}
    config_path = unique_run_config_path(); config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    logger.write(f"PROCESS_START pid={os.getpid()} config={config_path} run_config={run_config}")
    seed_everything()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for the specified Stage 2B configuration.")
    device = torch.device("cuda:0"); torch.cuda.empty_cache()
    logger.write(f"DEVICE_READY device={device} gpu={gpu_snapshot(device)}")
    train = content_frame(ROOT / "data" / "processed" / "isot_train.csv", 31280, "train")
    valid = content_frame(ROOT / "data" / "processed" / "isot_validation.csv", 3910, "validation")
    test = content_frame(ROOT / "data" / "processed" / "isot_test.csv", 3911, "test")
    if train.label.value_counts().to_dict() != {1: 16956, 0: 14324} or valid.label.value_counts().to_dict() != {1: 2120, 0: 1790} or test.label.value_counts().to_dict() != {1: 2120, 0: 1791}:
        raise RuntimeError("Existing ISOT split labels/counts differ from validated checkpoint.")
    run_config.update({"device": str(device), "gpu": gpu_snapshot(device), "isot_train_records": len(train), "isot_validation_records": len(valid), "isot_test_records": len(test)})
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    logger.write(f"DATA_READY train={len(train)} validation={len(valid)} test={len(test)} config_updated={config_path}")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_SOURCE, use_fast=True)
    # Smoke test comes before full tokenisation/training.
    smoke = tokenizer(train.content.iloc[:2].tolist(), truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_SOURCE, num_labels=2, id2label={0:"fake",1:"real"}, label2id={"fake":0,"real":1})
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(); model.config.use_cache = False
    model.to(device); model.train()
    torch.cuda.reset_peak_memory_stats(device)
    logits = model(**{k:v.to(device) for k,v in smoke.items()}, labels=torch.tensor([0,1], device=device)).logits
    loss = logits.sum(); loss.backward(); model.zero_grad(set_to_none=True)
    smoke_info = {"forward_pass": True, "backward_pass": True, "batch_size": 2, "max_length": MAX_LENGTH,
                  "allocated_mb": round(torch.cuda.max_memory_allocated(device)/1024**2, 2), "reserved_mb": round(torch.cuda.max_memory_reserved(device)/1024**2, 2)}
    (OUT / "sanity_test.json").write_text(json.dumps(smoke_info, indent=2), encoding="utf-8")
    train_ds, valid_ds, test_ds = TextDataset(train, tokenizer), TextDataset(valid, tokenizer), TextDataset(test, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    def make_train_loader(epoch):
        generator = torch.Generator().manual_seed(SEED + epoch)
        return DataLoader(train_ds, batch_size=PER_DEVICE_BATCH, shuffle=True, generator=generator, collate_fn=collator, num_workers=0)
    eval_loader = lambda ds: DataLoader(ds, batch_size=PER_DEVICE_BATCH, shuffle=False, collate_fn=collator, num_workers=0)
    steps_per_epoch = ((len(train_ds) + PER_DEVICE_BATCH - 1) // PER_DEVICE_BATCH + GRADIENT_ACCUMULATION - 1) // GRADIENT_ACCUMULATION
    total_steps = steps_per_epoch * EPOCHS; warmup_steps = round(total_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    logs, best_f1, stagnant, global_step = [], -1.0, 0, 0
    start_epoch, resume_batch = 1, 0
    if resume_from or resume_latest:
        selected = Path(resume_from) if resume_from else find_latest_valid_checkpoint()
        if selected is None or not validate_checkpoint(selected): raise RuntimeError("No valid explicit resume checkpoint was found.")
        state = load_resume_state(selected, model, optimizer, scheduler, scaler, device)
        start_epoch, resume_batch, global_step = int(state["epoch"]), int(state["batch_index"]), int(state["global_step"])
        logger.write(f"RESUME checkpoint={selected} epoch={start_epoch} batch_index={resume_batch} global_step={global_step}")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, EPOCHS + 1):
        train_loader = make_train_loader(epoch); skip_batches = resume_batch if epoch == start_epoch else 0
        model.train(); running = 0.0; began = time.perf_counter()
        for index, batch in enumerate(train_loader, start=1):
            if index <= skip_batches: continue
            labels = batch.pop("labels").to(device); batch = {k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16): loss = model(**batch, labels=labels).loss / GRADIENT_ACCUMULATION
            scaler.scale(loss).backward(); running += loss.item() * GRADIENT_ACCUMULATION
            if index % GRADIENT_ACCUMULATION == 0 or index == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1
                logger.write(f"STEP epoch={epoch} batch_index={index} global_step={global_step} loss={loss.item()*GRADIENT_ACCUMULATION:.6f} lr={scheduler.get_last_lr()[0]:.8g} elapsed_seconds={time.perf_counter()-began:.2f} gpu={gpu_snapshot(device)}")
                if global_step % CHECKPOINT_INTERVAL_STEPS == 0:
                    save_checkpoint_atomic(model, tokenizer, optimizer, scheduler, scaler, epoch, index, global_step, logger, device)
        y, p, prob = evaluate(model, eval_loader(valid_ds), device); valid_metrics, _ = metric_bundle(y, p, prob)
        log = {"epoch": epoch, "training_loss": running / len(train_loader), "seconds": time.perf_counter()-began, "global_optimizer_step": global_step, **valid_metrics}; logs.append(log)
        logger.write(f"VALIDATION epoch={epoch} metrics={log} gpu={gpu_snapshot(device)}")
        if valid_metrics["Macro_F1"] > best_f1:
            best_f1 = valid_metrics["Macro_F1"]; stagnant = 0; model.save_pretrained(MODEL_DIR); tokenizer.save_pretrained(MODEL_DIR)
        else:
            stagnant += 1
            if stagnant >= EARLY_STOPPING_PATIENCE: break
        resume_batch = 0
    pd.DataFrame(logs).to_csv(OUT / "training_log.csv", index=False)
    logger.write(f"TRAINING_COMPLETE epochs={len(logs)} global_step={global_step}")
    # The selected state is now frozen. Only now may the independent external set be opened.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device); model.config.use_cache = False
    y_test, p_test, s_test = evaluate(model, eval_loader(test_ds), device); m_test, cm_test = metric_bundle(y_test, p_test, s_test)
    external, independence = load_external_after_selection(); external_ds = TextDataset(external, tokenizer)
    y_ext, p_ext, s_ext = evaluate(model, eval_loader(external_ds), device); m_ext, cm_ext = metric_bundle(y_ext, p_ext, s_ext)
    save_predictions(OUT / "predictions_isot_test.csv", None, y_test, p_test, s_test)
    save_predictions(OUT / "predictions_fakenewsamt.csv", external.external_row_id.to_numpy(), y_ext, p_ext, s_ext)
    rows = [{"Model":"RoBERTa", "Evaluation":"ISOT_test", "Records":len(test), **m_test}, {"Model":"RoBERTa", "Evaluation":"FakeNewsAMT_external", "Records":len(external), **m_ext}]
    result = pd.DataFrame(rows); result.to_csv(TABLES / "roberta_evaluation.csv", index=False)
    gap = m_test["Macro_F1"] - m_ext["Macro_F1"]
    gap_df = pd.DataFrame([{"Model":"RoBERTa", "ISOT_test_Macro_F1":m_test["Macro_F1"], "FakeNewsAMT_Macro_F1":m_ext["Macro_F1"], "Macro_F1_gap":gap, "Macro_F1_gap_percentage_points":100*gap, "Macro_F1_relative_change_percent":100*gap/m_test["Macro_F1"]}])
    gap_df.to_csv(TABLES / "roberta_generalisation_gap.csv", index=False)
    base = pd.read_csv(TABLES / "baseline_external_evaluation.csv"); b_i=base[base.Evaluation.eq("ISOT_test")].set_index("Model"); b_e=base[base.Evaluation.eq("FakeNewsAMT_external")].set_index("Model")
    comparison = pd.DataFrame([{"Model":name,"ISOT_test_Macro_F1":b_i.loc[name,"Macro_F1"],"FakeNewsAMT_Macro_F1":b_e.loc[name,"Macro_F1"],"Macro_F1_gap_percentage_points":100*(b_i.loc[name,"Macro_F1"]-b_e.loc[name,"Macro_F1"])} for name in b_i.index] + [{"Model":"RoBERTa","ISOT_test_Macro_F1":m_test["Macro_F1"],"FakeNewsAMT_Macro_F1":m_ext["Macro_F1"],"Macro_F1_gap_percentage_points":100*gap}])
    comparison.to_csv(TABLES / "model_comparison_external.csv", index=False)
    config = {"model_name":MODEL_NAME,"tokenizer":MODEL_NAME,"seed":SEED,"input_representation":"existing ISOT title + text content; FakeNewsAMT parsed title + body and whitespace-collapsed", "max_length":MAX_LENGTH,"per_device_batch_size":PER_DEVICE_BATCH,"gradient_accumulation_steps":GRADIENT_ACCUMULATION,"effective_batch_size":PER_DEVICE_BATCH*GRADIENT_ACCUMULATION,"learning_rate":LEARNING_RATE,"epochs_requested":EPOCHS,"epochs_completed":len(logs),"optimizer":"AdamW","weight_decay":WEIGHT_DECAY,"scheduler":"linear","warmup_ratio":WARMUP_RATIO,"warmup_steps":warmup_steps,"mixed_precision":"fp16 torch.autocast + GradScaler","gradient_checkpointing":GRADIENT_CHECKPOINTING,"selection_criterion":"highest ISOT validation Macro-F1","early_stopping_patience":EARLY_STOPPING_PATIENCE,"decision_threshold":0.5,"external_opened_after_selection":True,"device":torch.cuda.get_device_name(device),"sanity_test":smoke_info,"isot_split_hashes":{n:sha256(ROOT/'data'/'processed'/f'isot_{n}.csv') for n in ('train','validation','test')},"fakenewsamt_independence":independence,"timestamp_utc":datetime.now(timezone.utc).isoformat()}
    (OUT / "training_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "model_configuration.json").write_text(model.config.to_json_string(use_diff=False), encoding="utf-8")
    (OUT / "tokenizer_configuration.json").write_text(tokenizer.backend_tokenizer.to_str(), encoding="utf-8")
    pd.DataFrame([{"Evaluation":"ISOT_test","Actual_class":a,"Predicted_class":p,"Count":int(cm_test[i,j])} for i,a in enumerate(["fake","real"]) for j,p in enumerate(["fake","real"])] + [{"Evaluation":"FakeNewsAMT_external","Actual_class":a,"Predicted_class":p,"Count":int(cm_ext[i,j])} for i,a in enumerate(["fake","real"]) for j,p in enumerate(["fake","real"])]).to_csv(OUT / "confusion_matrices.csv", index=False)
    figure_outputs(result, {"ISOT_test":cm_test,"FakeNewsAMT_external":cm_ext}, comparison, comparison)
    def table(frame, cols):
        x=frame[cols].copy()
        for c in x.columns:
            if c not in ("Model","Evaluation","Records"): x[c]=x[c].map(lambda v:f"{v:.4f}")
        return x.to_markdown(index=False)
    report = f"""# Stage 2B — RoBERTa controlled training and external evaluation

## 1. Objective

Assess whether an ISOT-trained contextual transformer improves generalisation to the independently audited FakeNewsAMT corpus. WELFake was not opened or used.

## 2. Dataset protocol and leakage prevention

Training used only the frozen ISOT train split (31,280); validation/model selection used only frozen ISOT validation (3,910). The selected checkpoint was restored before ISOT test (3,911) and FakeNewsAMT (430) evaluation. The script does not open FakeNewsAMT until validation selection has completed. It rechecked its documented 0 overlap on exact/normalised titles and bodies before evaluation. Labels are 0=fake and 1=real throughout.

## 3. Architecture, tokenisation, and input

`roberta-base` sequence classification with two labels was used. Input is the established title-plus-body/article `content` representation; FakeNewsAMT uses the same title plus parsed non-empty body representation. Dynamic padding, a conservative 128-token maximum length, batch size 2, gradient accumulation 8, fp16, and gradient checkpointing were selected for the 4 GB GTX 1650 Ti. The pre-training forward/backward smoke test passed: peak allocated {smoke_info['allocated_mb']} MiB, reserved {smoke_info['reserved_mb']} MiB.

## 4. Training and validation

Configuration is saved in `results/roberta/training_configuration.json`. The AdamW schedule used ISOT validation Macro-F1 for checkpoint selection only; no FakeNewsAMT metric informed a decision. Epoch-level validation metrics are in `results/roberta/training_log.csv`.

## 5. Final metrics

{table(result, ['Model','Evaluation','Records','Accuracy','Precision','Recall','Macro_F1','ROC_AUC','Fake_F1','Real_F1'])}

## 6. Generalisation gap

RoBERTa's ISOT-test Macro-F1 minus FakeNewsAMT Macro-F1 is **{100*gap:.2f} percentage points** ({100*gap/m_test['Macro_F1']:.2f}% relative decrease).

## 7. Comparison with traditional ML

{table(comparison, ['Model','ISOT_test_Macro_F1','FakeNewsAMT_Macro_F1','Macro_F1_gap_percentage_points'])}

The independent external Macro-F1, rather than the in-domain score alone, is the central comparison. {'RoBERTa has the highest external Macro-F1 among these four models.' if m_ext['Macro_F1'] > comparison[comparison.Model.ne('RoBERTa')].FakeNewsAMT_Macro_F1.max() else 'RoBERTa does not have the highest external Macro-F1 among these four models.'} {'It has the smallest generalisation gap.' if 100*gap < comparison[comparison.Model.ne('RoBERTa')].Macro_F1_gap_percentage_points.min() else 'It does not have the smallest generalisation gap.'}

## 8. Limitations and reproducibility

FakeNewsAMT is a 430-item corpus of short excerpts with crowdsourced fake articles, so it is a valid independently sourced external test here but not a proxy for all organic misinformation. The 128-token limit truncates long ISOT articles; it was fixed before training for the GPU constraint and applied unchanged to both final tests. Seeds, split hashes, source hash, configurations, prediction files, model/tokenizer files, logs, and confusion matrices are stored under `results/roberta/`.

## 9. Completion decision

Stage 2B completed: GPU sanity test, ISOT-only training/validation selection, and one final evaluation on each prescribed held-out test set completed successfully. No Stage 1 or Stage 2A artefact was modified.
"""
    (ROOT / "results" / "roberta_evaluation_report.md").write_text(report, encoding="utf-8")
    logger.write(f"FINAL_EVALUATION_COMPLETE metrics={rows}"); logger.close()
    print(json.dumps({"sanity":smoke_info,"isot":m_test,"external":m_ext,"gap_pp":100*gap,"epochs_completed":len(logs)}, indent=2))


def record_top_level_exception(exc: BaseException) -> None:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    ERROR_PATH.parent.mkdir(parents=True, exist_ok=True)
    ERROR_PATH.write_text(f"{datetime.now(timezone.utc).isoformat()}\n{text}", encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} | UNHANDLED_EXCEPTION type={type(exc).__name__}\n{text}\n")
        handle.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="test logging/checkpoint/resume helpers without training")
    parser.add_argument("--resume-from", default=None, help="explicit validated checkpoint directory")
    parser.add_argument("--resume-latest", action="store_true", help="explicitly resume the latest valid checkpoint")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        try:
            main(args.resume_from, args.resume_latest)
        except BaseException as error:
            record_top_level_exception(error)
            raise
