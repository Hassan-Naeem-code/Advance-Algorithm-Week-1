"""Utility helpers: data download and constants."""
from pathlib import Path
import requests  # type: ignore[import]
import zipfile
import io
import pandas as pd

# Default UCI Bank Marketing zip URL
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00222/bank-additional.zip"
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "bank-additional" / "bank-additional-full.csv"
OUT_DIR = ROOT / "outputs"


def ensure_data() -> pd.DataFrame:
    """Ensure dataset CSV exists locally; if not,
    download and extract from UCI.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe (sep=';')
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print("Downloading dataset from UCI... (this may take a few seconds)")
        r = requests.get(DATA_URL)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(DATA_DIR)
    # load
    df = pd.read_csv(DATA_PATH, sep=";")
    return df
