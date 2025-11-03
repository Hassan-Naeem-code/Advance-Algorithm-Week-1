"""Simple validation checks for the processed dataset.

Run this after `src/main.py` completes to ensure key expectations hold.
"""
import sys
from pathlib import Path
import pandas as pd


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "outputs" / "processed.csv"
    if not out_path.exists():
        print(f"Processed file not found: {out_path}")
        sys.exit(2)

    df = pd.read_csv(out_path)
    # Basic checks
    assert df.shape[0] > 0, "Processed dataframe is empty"

    # Ensure engineered features exist
    for col in ["age_bucket", "duration_log", "campaign_prev_ratio"]:
        assert col in df.columns, f"Missing engineered column: {col}"

    # No NaNs in numeric columns
    num_na = df.select_dtypes(include=["number"]).isnull().sum().sum()
    assert num_na == 0, f"Found {num_na} NaNs in numeric columns"

    print("Validation OK: processed dataset looks good.")


if __name__ == "__main__":
    main()
