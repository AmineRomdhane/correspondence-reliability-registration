#!/usr/bin/env python3
"""
Analyze feature correlations for the correspondence reliability dataset.

Inputs:
- results/learning_data/all_correspondences.csv

Outputs:
- results/tables/feature_correlation_pearson.csv
- results/tables/feature_correlation_spearman.csv
- results/tables/feature_target_correlation.csv
- results/tables/highly_correlated_feature_pairs.csv
- results/tables/feature_target_correlation_by_scenario.csv
- results/figures/feature_correlation_pearson.png
- results/figures/feature_correlation_spearman.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = BASE_DIR / "results" / "learning_data" / "all_correspondences.csv"

TABLE_DIR = BASE_DIR / "results" / "tables"
FIGURE_DIR = BASE_DIR / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "distance_T0",
    "point_to_plane_residual",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]

TARGET = "target_weight"


def plot_correlation_matrix(corr, title, output_path):
    labels = list(corr.columns)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1.0, vmax=1.0)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    # Write correlation values inside cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def find_highly_correlated_pairs(corr, threshold=0.85):
    rows = []
    columns = list(corr.columns)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            f1 = columns[i]
            f2 = columns[j]
            value = corr.loc[f1, f2]

            if abs(value) >= threshold:
                rows.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": value,
                    "abs_correlation": abs(value),
                })

    return pd.DataFrame(rows).sort_values(
        by="abs_correlation",
        ascending=False,
    ) if rows else pd.DataFrame(
        columns=["feature_1", "feature_2", "correlation", "abs_correlation"]
    )


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = FEATURES + [TARGET, "scenario", "sample_id"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise RuntimeError(f"Missing columns in dataset: {missing}")

    print(f"Loaded dataset: {INPUT_CSV}")
    print(f"Rows: {len(df)}")
    print(f"Features: {FEATURES}")

    # Keep only useful columns and remove invalid values
    analysis_cols = FEATURES + [TARGET]
    data = df[analysis_cols].replace([np.inf, -np.inf], np.nan).dropna()

    print(f"Rows after removing NaN/inf: {len(data)}")

    # Pearson correlation: linear correlation
    pearson_corr = data.corr(method="pearson")

    # Spearman correlation: monotonic/rank correlation
    spearman_corr = data.corr(method="spearman")

    pearson_path = TABLE_DIR / "feature_correlation_pearson.csv"
    spearman_path = TABLE_DIR / "feature_correlation_spearman.csv"

    pearson_corr.to_csv(pearson_path)
    spearman_corr.to_csv(spearman_path)

    print(f"Saved Pearson correlation:  {pearson_path}")
    print(f"Saved Spearman correlation: {spearman_path}")

    # Correlation with target
    target_rows = []
    for feature in FEATURES:
        target_rows.append({
            "feature": feature,
            "pearson_corr_with_target": pearson_corr.loc[feature, TARGET],
            "spearman_corr_with_target": spearman_corr.loc[feature, TARGET],
            "abs_pearson": abs(pearson_corr.loc[feature, TARGET]),
            "abs_spearman": abs(spearman_corr.loc[feature, TARGET]),
        })

    target_corr = pd.DataFrame(target_rows)
    target_corr = target_corr.sort_values(by="abs_spearman", ascending=False)

    target_corr_path = TABLE_DIR / "feature_target_correlation.csv"
    target_corr.to_csv(target_corr_path, index=False)

    print(f"Saved target correlation: {target_corr_path}")

    # Highly correlated feature pairs, excluding target_weight
    pearson_feature_corr = data[FEATURES].corr(method="pearson")
    high_pairs = find_highly_correlated_pairs(
        pearson_feature_corr,
        threshold=0.85,
    )

    high_pairs_path = TABLE_DIR / "highly_correlated_feature_pairs.csv"
    high_pairs.to_csv(high_pairs_path, index=False)

    print(f"Saved highly correlated pairs: {high_pairs_path}")

    # Correlation with target by scenario
    scenario_rows = []

    for scenario, group in df.groupby("scenario"):
        group_data = group[analysis_cols].replace([np.inf, -np.inf], np.nan).dropna()

        if len(group_data) < 10:
            continue

        group_corr = group_data.corr(method="spearman")

        for feature in FEATURES:
            scenario_rows.append({
                "scenario": scenario,
                "feature": feature,
                "spearman_corr_with_target": group_corr.loc[feature, TARGET],
                "num_rows": len(group_data),
            })

    scenario_corr = pd.DataFrame(scenario_rows)
    scenario_corr = scenario_corr.sort_values(
        by=["scenario", "spearman_corr_with_target"],
        ascending=[True, False],
    )

    scenario_corr_path = TABLE_DIR / "feature_target_correlation_by_scenario.csv"
    scenario_corr.to_csv(scenario_corr_path, index=False)

    print(f"Saved scenario target correlation: {scenario_corr_path}")

    # Figures
    plot_correlation_matrix(
        pearson_corr,
        "Pearson correlation between features and target",
        FIGURE_DIR / "feature_correlation_pearson.png",
    )

    plot_correlation_matrix(
        spearman_corr,
        "Spearman correlation between features and target",
        FIGURE_DIR / "feature_correlation_spearman.png",
    )

    print(f"Saved Pearson figure:  {FIGURE_DIR / 'feature_correlation_pearson.png'}")
    print(f"Saved Spearman figure: {FIGURE_DIR / 'feature_correlation_spearman.png'}")

    print("\nCorrelation with target_weight:")
    print(target_corr.to_string(index=False))

    print("\nHighly correlated feature pairs |Pearson| >= 0.85:")
    if len(high_pairs) == 0:
        print("None")
    else:
        print(high_pairs.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()

