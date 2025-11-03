# Week 1 — Data Exploration & Feature Engineering (Bank Marketing)

Dataset: UCI Bank Marketing (bank-additional-full.csv)
Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

This repository contains a scripted pipeline (no notebooks) that performs:

- Exploratory data analysis (summary statistics + plots + EDA report)
- Data cleaning (missing values and outlier handling)
- Feature engineering (2+ engineered features)
- Scaling / preprocessing

How to run (macOS / zsh)

1. Using conda (recommended, resolves compiled package compatibility issues):

```bash
conda create -n week1 python=3.11 -y
conda activate week1
pip install -r requirements.txt
python src/main.py
```

2. Or using venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

What the pipeline produces

- `outputs/plots/` — PNG plots created during EDA (histograms, boxplots, scatter, correlation matrix, missingness heatmap, target balance)
- `outputs/eda_report.txt` — short textual EDA report (column info, missingness, skewness, correlation matrix)
- `outputs/processed.csv` — cleaned, feature-engineered, scaled dataset ready for modeling

Design and methods (brief)

- Dataset: UCI Bank Marketing (bank-additional-full.csv). Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

- Missing values: In this dataset some categorical fields use the string `unknown` to indicate missingness. I treat `unknown` as missing and impute:
	- Numeric columns: median imputation (robust to skew and outliers).
	- Categorical columns: mode imputation (use most frequent category).
	- Justification: median/mode are simple, reproducible, and suitable when missingness appears not to be informative for these features.

- Outliers: Numeric outliers are capped using IQR-based winsorization (clip to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]).
	- Justification: capping preserves rank information while limiting influence of extreme values on scaling and models.

- Feature engineering (examples created in `src/features.py`):
	- `age_bucket`: categorical bins for age (young/adult/senior) — helps capture nonlinear age effects.
	- `duration_log`: log1p of call duration to reduce skew and stabilize variance.
	- `campaign_prev_ratio`: ratio of `campaign` to (`previous` + 1) — an interaction-like feature indicating effort vs prior contacts.

- Scaling: numeric features are transformed using `StandardScaler` (zero mean, unit variance) so features are comparable for downstream models that are sensitive to scale.
	- Justification: standard scaling is a reasonable default for algorithms like logistic regression or SVM; if using tree-based models scaling is less critical.

Validation and checks

Run the small validation script after the pipeline to ensure expectations are met:

```bash
python tests/validate_processed.py
```

This script checks that `outputs/processed.csv` exists, contains the engineered features, has at least one row, and has no NaNs in numeric columns.