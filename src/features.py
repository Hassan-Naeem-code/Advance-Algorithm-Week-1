"""Feature engineering and scaling.

Creates at least two new features and applies scaling to numeric features.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def create_features_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Age bucket (domain-informed): young, adult, senior
    bins = [0, 30, 60, 120]
    labels = ["young", "adult", "senior"]
    df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

    # 2) Duration log-transform to reduce skew and stabilize variance
    # `balance` is not present in this dataset; use `duration` (call length)
    # Clip at 0 to avoid negative values before log1p
    df["duration_log"] = np.log1p(df["duration"].clip(lower=0))

    # 3) Campaign to previous contacts ratio (small engineered signal)
    # previous is number of contacts performed before this campaign for this client
    df["campaign_prev_ratio"] = df["campaign"] / (df["previous"] + 1)

    # Prepare numeric scaling: choose numeric cols to scale
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Ensure no NaNs before scaling
    assert not df[num_cols].isnull().any().any(), "Numeric columns contain NaNs before scaling"

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

    return df_scaled
