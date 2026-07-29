#!/usr/bin/env python3
"""
Create a curated synthetic + real correspondence dataset.

Input:
- configs/dataset_samples_synthetic_plus_real.csv
- results/learning_data_synthetic_plus_real/
- results/learning_data_synthetic_plus_real/dataset_summary.csv

Output:
- configs/dataset_samples_synthetic_plus_real_curated.csv
- results/learning_data_synthetic_plus_real_curated/
- results/tables/curated_real_samples_kept.csv
- results/tables/curated_real_samples_rejected.csv

Rule:
- Keep all synthetic samples.
- Keep real samples only if their extracted labels are informative.
"""

from pathlib import Path
import shutil
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_CONFIG = BASE_DIR / "configs" / "dataset_samples_synthetic_plus_real.csv"
INPUT_DATA_DIR = BASE_DIR / "results" / "learning_data_synthetic_plus_real"
INPUT_SUMMARY = INPUT_DATA_DIR / "dataset_summary.csv"

OUTPUT_CONFIG = BASE_DIR / "configs" / "dataset_samples_synthetic_plus_real_curated.csv"
OUTPUT_DATA_DIR = BASE_DIR / "results" / "learning_data_synthetic_plus_real_curated"

KEPT_REAL_TABLE = BASE_DIR / "results" / "tables" / "curated_real_samples_kept.csv"
REJECTED_REAL_TABLE = BASE_DIR / "results" / "tables" / "curated_real_samples_rejected.csv"

MIN_POSITIVE_RATE = 0.05
MAX_POSITIVE_RATE = 0.75
MIN_POSITIVE = 30
MIN_NEGATIVE = 30


def copy_if_exists(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    config_df = pd.read_csv(INPUT_CONFIG)
    summary_df = pd.read_csv(INPUT_SUMMARY)

    real_mask = summary_df["sample_id"].astype(str).str.startswith("real_")
    real_summary = summary_df[real_mask].copy()
    synthetic_summary = summary_df[~real_mask].copy()

    keep_real = (
        (real_summary["positive_rate"] >= MIN_POSITIVE_RATE)
        & (real_summary["positive_rate"] <= MAX_POSITIVE_RATE)
        & (real_summary["num_positive"] >= MIN_POSITIVE)
        & (real_summary["num_negative"] >= MIN_NEGATIVE)
    )

    kept_real = real_summary[keep_real].copy()
    rejected_real = real_summary[~keep_real].copy()

    kept_sample_ids = set(synthetic_summary["sample_id"]) | set(kept_real["sample_id"])

    curated_config = config_df[
        config_df["sample_id"].isin(kept_sample_ids)
    ].copy()

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "results" / "tables").mkdir(parents=True, exist_ok=True)

    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    curated_config.to_csv(OUTPUT_CONFIG, index=False)

    kept_real.to_csv(KEPT_REAL_TABLE, index=False)
    rejected_real.to_csv(REJECTED_REAL_TABLE, index=False)

    # Copy feature_columns.txt if present
    copy_if_exists(
        INPUT_DATA_DIR / "feature_columns.txt",
        OUTPUT_DATA_DIR / "feature_columns.txt",
    )

    # Copy selected per-sample CSVs
    copied = 0

    for sample_id in curated_config["sample_id"]:
        src = INPUT_DATA_DIR / f"{sample_id}_correspondences.csv"
        dst = OUTPUT_DATA_DIR / f"{sample_id}_correspondences.csv"

        copy_if_exists(src, dst)

        if dst.exists():
            copied += 1

    # Rebuild all_correspondences.csv and dataset_summary.csv
    dfs = []

    for sample_id in curated_config["sample_id"]:
        f = OUTPUT_DATA_DIR / f"{sample_id}_correspondences.csv"
        if f.exists():
            dfs.append(pd.read_csv(f))

    if not dfs:
        raise RuntimeError("No correspondence CSV files were copied.")

    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    all_df.to_csv(OUTPUT_DATA_DIR / "all_correspondences.csv", index=False)

    curated_summary = summary_df[
        summary_df["sample_id"].isin(kept_sample_ids)
    ].copy()

    curated_summary.to_csv(OUTPUT_DATA_DIR / "dataset_summary.csv", index=False)

    print("=" * 80)
    print("Curated synthetic + real dataset created")
    print("=" * 80)
    print(f"Synthetic samples kept: {len(synthetic_summary)}")
    print(f"Real samples before:    {len(real_summary)}")
    print(f"Real samples kept:      {len(kept_real)}")
    print(f"Real samples rejected:  {len(rejected_real)}")
    print(f"Total samples kept:     {len(curated_config)}")
    print(f"Copied CSV files:       {copied}")
    print(f"Total rows:             {len(all_df)}")
    print(f"Positive rate:          {all_df['target_weight'].mean():.4f}")
    print("-" * 80)
    print(f"Config:                 {OUTPUT_CONFIG}")
    print(f"Dataset folder:         {OUTPUT_DATA_DIR}")
    print(f"Kept real table:        {KEPT_REAL_TABLE}")
    print(f"Rejected real table:    {REJECTED_REAL_TABLE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
