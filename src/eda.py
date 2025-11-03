"""Exploratory Data Analysis helpers.

Functions save plots to disk under `out_dir/plots` and write a short
textual EDA report to `out_dir/eda_report.txt`.
"""
# typing imports removed: type information kept inline as comment in function
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


def run_eda(df: pd.DataFrame, out_dir=None) -> None:
    # Normalize out_dir to a Path before using `/` operator
    # normalize out_dir type in code below
    if out_dir is None:
        out_path = Path("outputs")
    else:
        out_path = Path(out_dir)
    plots_dir = out_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append("EDA Report")
    report_lines.append("===========\n")

    # Basic prints
    buf = io.StringIO()
    df.info(buf=buf)
    report_lines.append("Dataframe info:")
    report_lines.append(buf.getvalue())
    report_lines.append("\nHead:\n" + df.head().to_string())

    summary = df.describe(include="all", datetime_is_numeric=True).to_string()
    report_lines.append("\nSummary statistics:\n" + summary)

    # Missingness overview (treat 'unknown' for object dtypes)
    obj_cols = df.select_dtypes(include=[object]).columns.tolist()
    miss_counts = df.isnull().sum()
    # count 'unknown' values as missing for object columns
    for c in obj_cols:
        unknowns = df[c].isin(['unknown']).sum()
        if unknowns > 0:
            miss_counts[c] = miss_counts.get(c, 0) + unknowns

    report_lines.append("\nMissing / unknown counts (cols with >0):")
    report_lines.append(miss_counts[miss_counts > 0].to_string())

    # Numeric analysis and plots
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Histograms and boxplots
    for col in num_cols[:8]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=False)
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"hist_{col}.png")
        plt.close()

        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"box_{col}.png")
        plt.close()

    # Scatter: first two numeric
    if len(num_cols) >= 2:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            x=df[num_cols[0]],
            y=df[num_cols[1]],
            alpha=0.3,
        )
        plt.title(f"Scatter: {num_cols[0]} vs {num_cols[1]}")
    plt.tight_layout()
    out_name = f"scatter_{num_cols[0]}_{num_cols[1]}.png"
    plt.savefig(plots_dir / out_name)
    plt.close()

    # Correlation heatmap for numeric features
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=False,
            cmap='coolwarm',
            center=0,
        )
        plt.title("Correlation matrix (numeric features)")
        plt.tight_layout()
        plt.savefig(plots_dir / "corr_matrix.png")
        plt.close()
        report_lines.append("\nCorrelation matrix (numeric features):")
        report_lines.append(corr.to_string())

    # Missingness heatmap (visual)
    try:
        miss_mask = df.isnull()
        # also mark 'unknown' as missing for object cols
        for c in obj_cols:
            miss_mask[c] = miss_mask[c] | df[c].isin(['unknown'])
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            miss_mask.iloc[:500].T,
            cbar=False,
        )  # show first 500 rows for legibility
        plt.title("Missingness heatmap (first 500 rows)")
        plt.tight_layout()
        plt.savefig(plots_dir / "missingness_heatmap.png")
        plt.close()
    except Exception:
        # don't fail EDA if heatmap cannot be produced
        report_lines.append("Could not produce missingness heatmap")

    # Target class balance (if 'y' exists)
    if 'y' in df.columns:
        vc = df['y'].value_counts()
        plt.figure(figsize=(5, 4))
        sns.barplot(x=vc.index.astype(str), y=vc.values)
        plt.title('Target class balance (y)')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(plots_dir / 'target_balance.png')
        plt.close()
        report_lines.append('\nTarget class balance:')
        report_lines.append(vc.to_string())

    # Skewness for numeric columns
    skewness = df[num_cols].skew().sort_values(ascending=False)
    report_lines.append('\nSkewness (numeric cols):')
    report_lines.append(skewness.to_string())

    # Write textual report
    report_path = out_path / "eda_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(
        "EDA complete — plots saved to",
        plots_dir,
        "and report to",
        report_path,
    )
