"""Entry point for the Week 1 assignment pipeline.

Flow:
 - load dataset (downloads if missing)
 - run EDA (summary + plots)
 - clean data (missing values + outlier handling)
 - feature engineering + scaling
 - write processed dataset
"""
import sys
from pathlib import Path
import logging

# Make the project root importable so `python src/main.py` works when run from
# the repository root or from inside `src/`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_data, OUT_DIR  # noqa: E402
from src import eda, cleaning, features  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR, "plots").mkdir(parents=True, exist_ok=True)

    logging.info("Ensuring dataset is available...")
    df = ensure_data()

    logging.info("Running EDA...")
    eda.run_eda(df, out_dir=OUT_DIR)

    logging.info("Cleaning data...")
    df_clean = cleaning.clean_data(df)

    logging.info("Engineering features and scaling...")
    df_feat = features.create_features_and_scale(df_clean)

    out_file = Path(OUT_DIR) / "processed.csv"
    df_feat.to_csv(out_file, index=False)
    logging.info(f"Processed dataset written to {out_file}")


if __name__ == "__main__":
    main()
