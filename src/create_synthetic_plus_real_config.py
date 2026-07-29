#!/usr/bin/env python3
"""
Create a new dataset config that keeps the original synthetic config unchanged
and appends validated real pseudo-ground-truth registrations.

Input:
- configs/dataset_samples.csv
- real_results/tables/real_registration_results.csv

Output:
- configs/dataset_samples_synthetic_plus_real.csv
- results/tables/real_rows_selected_for_learning.csv
"""

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

SYNTHETIC_CONFIG = BASE_DIR / "configs" / "dataset_samples.csv"
REAL_TABLE = BASE_DIR / "real_results" / "tables" / "real_registration_results.csv"

OUTPUT_CONFIG = BASE_DIR / "configs" / "dataset_samples_synthetic_plus_real.csv"
SELECTED_REAL_TABLE = BASE_DIR / "results" / "tables" / "real_rows_selected_for_learning.csv"

MIN_FITNESS = 0.30
MAX_RMSE = 0.06

VOXEL_SIZE = 0.05
NORMAL_RADIUS = 0.15
FPFH_RADIUS = 0.25
DENSITY_RADIUS = 0.15
LABEL_THRESHOLD_M = 0.10
MAX_SOURCE_POINTS = 2000

DIFFICULTIES = [
    {
        "suffix": "easy",
        "perturb_rot_deg": 3.0,
        "perturb_trans_m": 0.05,
        "max_corr_distance_m": 0.40,
    },
    {
        "suffix": "medium",
        "perturb_rot_deg": 5.0,
        "perturb_trans_m": 0.08,
        "max_corr_distance_m": 0.50,
    },
    {
        "suffix": "hard",
        "perturb_rot_deg": 8.0,
        "perturb_trans_m": 0.12,
        "max_corr_distance_m": 0.60,
    },
]

CONFIG_COLUMNS = [
    "sample_id",
    "scenario",
    "reference_path",
    "observation_path",
    "label_transform_path",
    "label_transform_direction",
    "T0_mode",
    "T0_path",
    "perturb_rot_deg",
    "perturb_trans_m",
    "voxel_size",
    "normal_radius",
    "fpfh_radius",
    "density_radius",
    "max_corr_distance_m",
    "label_threshold_m",
    "max_source_points",
]


def clean_name(text):
    text = str(text).lower()
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")

    out = "".join(cleaned)

    while "__" in out:
        out = out.replace("__", "_")

    return out.strip("_")


def exists_path(path_value):
    if pd.isna(path_value):
        return False

    p = Path(str(path_value))

    if p.is_absolute():
        return p.exists()

    return (BASE_DIR / p).exists()


def to_relative_string(path_value):
    p = Path(str(path_value))

    if p.is_absolute():
        try:
            return str(p.relative_to(BASE_DIR))
        except ValueError:
            return str(p)

    return str(p)


def main():
    synthetic_df = pd.read_csv(SYNTHETIC_CONFIG)
    real_df = pd.read_csv(REAL_TABLE)

    for col in CONFIG_COLUMNS:
        if col not in synthetic_df.columns:
            raise RuntimeError(f"Synthetic config missing column: {col}")

    required_real_cols = [
        "run_name",
        "cad_file",
        "target_file",
        "matrix_file",
        "final_eval_fitness",
        "final_eval_rmse",
    ]

    missing_real = [c for c in required_real_cols if c not in real_df.columns]
    if missing_real:
        raise RuntimeError(f"Real table missing columns: {missing_real}")

    real_df["final_eval_fitness"] = pd.to_numeric(
        real_df["final_eval_fitness"],
        errors="coerce",
    )

    real_df["final_eval_rmse"] = pd.to_numeric(
        real_df["final_eval_rmse"],
        errors="coerce",
    )

    selected = []

    for _, row in real_df.iterrows():
        fitness = row["final_eval_fitness"]
        rmse = row["final_eval_rmse"]

        if pd.isna(fitness) or pd.isna(rmse):
            continue

        if fitness < MIN_FITNESS:
            continue

        if rmse > MAX_RMSE:
            continue

        if not exists_path(row["cad_file"]):
            continue

        if not exists_path(row["target_file"]):
            continue

        if not exists_path(row["matrix_file"]):
            continue

        selected.append(row)

    selected_df = pd.DataFrame(selected)

    if selected_df.empty:
        raise RuntimeError(
            "No real rows selected. Check file paths or relax MIN_FITNESS/MAX_RMSE."
        )

    real_config_rows = []

    for _, row in selected_df.iterrows():
        run_name = clean_name(row["run_name"])

        for difficulty in DIFFICULTIES:
            sample_id = f"real_{run_name}_{difficulty['suffix']}"

            real_config_rows.append({
                "sample_id": sample_id,
                "scenario": "real_previous_registration",
                "reference_path": to_relative_string(row["target_file"]),
                "observation_path": to_relative_string(row["cad_file"]),
                "label_transform_path": to_relative_string(row["matrix_file"]),
                "label_transform_direction": "obs_to_ref",
                "T0_mode": "perturb_label",
                "T0_path": "",
                "perturb_rot_deg": difficulty["perturb_rot_deg"],
                "perturb_trans_m": difficulty["perturb_trans_m"],
                "voxel_size": VOXEL_SIZE,
                "normal_radius": NORMAL_RADIUS,
                "fpfh_radius": FPFH_RADIUS,
                "density_radius": DENSITY_RADIUS,
                "max_corr_distance_m": difficulty["max_corr_distance_m"],
                "label_threshold_m": LABEL_THRESHOLD_M,
                "max_source_points": MAX_SOURCE_POINTS,
            })

    real_config_df = pd.DataFrame(real_config_rows, columns=CONFIG_COLUMNS)

    combined_df = pd.concat(
        [
            synthetic_df[CONFIG_COLUMNS],
            real_config_df[CONFIG_COLUMNS],
        ],
        axis=0,
        ignore_index=True,
    )

    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_REAL_TABLE.parent.mkdir(parents=True, exist_ok=True)

    combined_df.to_csv(OUTPUT_CONFIG, index=False)
    selected_df.to_csv(SELECTED_REAL_TABLE, index=False)

    print("=" * 80)
    print("Created synthetic + real dataset config")
    print("=" * 80)
    print(f"Synthetic rows:       {len(synthetic_df)}")
    print(f"Selected real pairs:  {len(selected_df)}")
    print(f"Added real rows:      {len(real_config_df)}")
    print(f"Total config rows:    {len(combined_df)}")
    print("-" * 80)
    print(f"New config:           {OUTPUT_CONFIG}")
    print(f"Selected real table:  {SELECTED_REAL_TABLE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
