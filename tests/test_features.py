import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.features import create_features_and_scale


def test_features_and_scaling():
    df = pd.DataFrame(
        {
            "age": [20, 40, 70],
            "duration": [10, 20, 30],
            "campaign": [1, 2, 3],
            "previous": [0, 1, 2],
        }
    )
    out = create_features_and_scale(df)

    # engineered columns present
    assert "age_bucket" in out.columns
    assert "duration_log" in out.columns
    assert "campaign_prev_ratio" in out.columns

    # numeric columns scaled (mean approx 0)
    num = out.select_dtypes(include=["number"]).columns
    # allow small floating point tolerance
    means = out[num].mean().abs()
    assert (means < 1e-6).all()
