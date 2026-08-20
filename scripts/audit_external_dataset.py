"""Deterministic ISOT independence audit for locally acquired external datasets.

This script is analysis-only.  It does not load, fit, tune, or evaluate a
model.  It documents the checks used in the external-dataset selection stage.
"""
from pathlib import Path
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def exact(value: object) -> str:
    """NFKC text with normalised line endings and trimmed outer whitespace."""
    text = "" if pd.isna(value) else str(value)
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalised_body(value: object) -> str:
    return re.sub(r"\s+", " ", exact(value).casefold()).strip()


def normalised_title(value: object) -> str:
    text = re.sub(r"[^\w\s]", " ", normalised_body(value))
    return re.sub(r"\s+", " ", text).strip()


def split_embedded_title_body(value: object) -> tuple[str, str]:
    """Split FakeNewsAMT's documented title + blank-line + body encoding."""
    title, separator, body = exact(value).partition("\n\n")
    return title.strip(), body.strip() if separator else ""


def overlap_rows(name: str, isot: pd.DataFrame, candidate: pd.DataFrame,
                 candidate_title: str | None, candidate_body: str) -> list[dict]:
    checks = [("exact_body_text", "text", candidate_body, exact),
              ("normalised_body_text", "text", candidate_body, normalised_body)]
    if candidate_title:
        checks.extend([("exact_title", "title", candidate_title, exact),
                       ("normalised_title", "title", candidate_title, normalised_title)])
    rows = []
    for field, isot_field, candidate_field, transform in checks:
        reference = {transform(value) for value in isot[isot_field]} - {""}
        external = {transform(value) for value in candidate[candidate_field]} - {""}
        overlap = reference & external
        rows.append({"Dataset": name, "Check": field,
                     "Overlap_unique_records": len(overlap),
                     "Candidate_unique_records": len(external),
                     "ISOT_unique_records": len(reference),
                     "Candidate_overlap_percent": 100 * len(overlap) / len(external) if external else 0.0,
                     "Status": "PASS" if not overlap else "FAIL"})
    return rows


def main() -> None:
    isot = pd.concat([pd.read_csv(ROOT / "News_Dataset" / "Fake.csv"),
                      pd.read_csv(ROOT / "News_Dataset" / "True.csv")], ignore_index=True)
    rows: list[dict] = []

    amt = pd.read_parquet(ROOT / "data" / "external_candidates" / "FakeNewsAMT" /
                          "train-00000-of-00001.parquet")
    title_body = amt["text"].map(split_embedded_title_body).tolist()
    amt["title"] = [pair[0] for pair in title_body]
    amt["body"] = [pair[1] for pair in title_body]
    amt = amt[amt["body"].ne("")].copy()
    rows.extend(overlap_rows("FakeNewsAMT", isot, amt, "title", "body"))

    snopes = pd.read_csv(ROOT / "data" / "external_candidates" / "Misinformation_detection" /
                         "snopes_checked" / "snopes_checked_v02.csv", encoding="cp1252")
    snopes = snopes[snopes["fact_rating_phase1"].isin(["TRUE", "FALSE"])].copy()
    rows.extend(overlap_rows("MisInfoText Snopes312 strict subset", isot, snopes,
                             "article_title_phase2", "original_article_text_phase2"))

    buzzfeed = pd.read_csv(ROOT / "data" / "external_candidates" / "Misinformation_detection" /
                           "buzzfeed-v02.txt" / "buzzfeed-v02.txt", sep="\t", header=None,
                           names=["id", "url", "rating", "text", "domain", "collection"],
                           encoding="cp1252")
    # This is an audit-only provisional mapping.  It is not selected because
    # the resulting corpus is severely imbalanced and uses mostly- labels.
    buzzfeed = buzzfeed[buzzfeed["rating"].isin(["mtrue", "mfalse"])].copy()
    rows.extend(overlap_rows("MisInfoText BuzzFeed v02", isot, buzzfeed, None, "text"))

    output = ROOT / "results" / "tables" / "external_dataset_overlap.csv"
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
