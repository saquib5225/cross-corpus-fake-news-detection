"""Dataset loaders with explicit label mapping."""
from pathlib import Path
import pandas as pd
from .preprocessing import combine_title_text, normalise_for_tfidf

def load_isot(directory: Path) -> pd.DataFrame:
    fake = pd.read_csv(directory / "Fake.csv")
    real = pd.read_csv(directory / "True.csv")
    fake["label"] = 0
    real["label"] = 1
    frame = pd.concat([fake, real], ignore_index=True)
    frame["content"] = normalise_for_tfidf(combine_title_text(frame))
    return frame

def load_welfake(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    columns = {column.lower(): column for column in frame.columns}
    frame = frame.rename(columns={columns["title"]: "title", columns["text"]: "text", columns["label"]: "label"})
    # The released CSV has 35,028 records coded 0 and 37,106 coded 1. These
    # counts match the authors' published real/fake totals respectively, so
    # map it explicitly to this project's 0=fake, 1=real convention.
    frame["label"] = 1 - frame["label"].astype(int)
    frame["content"] = normalise_for_tfidf(combine_title_text(frame))
    return frame
