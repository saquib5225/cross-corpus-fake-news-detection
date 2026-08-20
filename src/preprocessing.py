"""Leakage-safe common text preparation."""
import re
import pandas as pd

def combine_title_text(frame: pd.DataFrame, title: str = "title", text: str = "text") -> pd.Series:
    """Create a documented, equivalent title-plus-body representation."""
    return (frame[title].fillna("").astype(str).str.strip() + " " + frame[text].fillna("").astype(str).str.strip()).str.replace(r"\s+", " ", regex=True).str.strip()

def normalise_for_tfidf(text: pd.Series) -> pd.Series:
    """Minimal normalisation; TfidfVectorizer performs tokenisation and lowercasing."""
    return text.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
