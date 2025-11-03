"""Data cleaning: missing values, outlier detection and handling.

Approach (brief):
- Treat categorical value 'unknown' as missing.
- Impute numeric columns with median and categorical with mode.
- Detect outliers using IQR and cap (winsorize) at 1.5*IQR fences.
"""
import pandas as pd


def _replace_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    # In this dataset categorical 'unknown' denotes missing
    obj_cols = df.select_dtypes(include=[object]).columns
    for c in obj_cols:
        if df[c].isin(["unknown"]).any():
            df[c] = df[c].replace("unknown", pd.NA)
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    # Numeric: median; Categorical: mode (as string)
    for c in df.select_dtypes(include=["number"]).columns:
        if df[c].isnull().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)
    for c in df.select_dtypes(include=[object]).columns:
        if df[c].isnull().any():
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "missing"
            df[c] = df[c].fillna(fill)
    return df


def _winsorize_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return series.clip(lower=lower, upper=upper)


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["number"]).columns
    for c in num_cols:
        # apply winsorization to cap extreme values while preserving rank
        df[c] = _winsorize_iqr(df[c])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _replace_unknowns(df)
    df = _impute(df)
    df = handle_outliers(df)
    return df
