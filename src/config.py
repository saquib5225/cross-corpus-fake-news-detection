from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ISOT = ROOT / "News_Dataset"
RAW_WELFAKE = ROOT / "data" / "raw" / "WELFake" / "WELFake_Dataset.csv"
RESULTS = ROOT / "results"
SEED = 42
