import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.cleaning import clean_data


def test_clean_data_imputes_and_caps():
    # create a small dataframe with 'unknown', NaNs and an outlier
    df = pd.DataFrame(
        {
            "age": [25, np.nan, 80],
            "job": ["admin.", "unknown", "technician"],
            "duration": [10, 20, 10000],
            "campaign": [1, 2, 3],
            "previous": [0, 1, 0],
        }
    )

    cleaned = clean_data(df)

    # no NaNs in numeric columns
    assert not cleaned.select_dtypes(include=["number"]).isnull().any().any()

    # 'unknown' should be replaced (imputed) in job
    assert "unknown" not in cleaned["job"].values

    # duration values should be finite and numeric
    assert np.isfinite(cleaned["duration"]).all()
