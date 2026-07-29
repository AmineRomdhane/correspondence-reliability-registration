#!/usr/bin/env python3
"""
Compare one unseen real v2 case using the final all-v3 MLP model.

Unseen definition:
- model trained on synthetic + real clean v3
- test case selected from real clean v2 but absent from v3

Methods:
1. initial_T0
2. one_shot_unweighted_svd
3. mlp_weighted_svd
4. pseudo_gt_label

No normal ICP is used here, because the fair comparison is:
one-shot unweighted SVD vs one-shot MLP-weighted SVD.

Outputs:
results/by_test/unseen_real_v2_registration_comparison/<sample_id>_<init_tag>/
    registration_comparison_metrics.csv
    candidate_features_with_mlp_weights.csv
    figures/
"""

from pathlib import Path
import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd
import open3d as o3d
import torch

from run_unseen_synthetic_registration_comparison import (
    FEATURES,
    MODEL_PATH,
    SCALER_PATH,
    make_transform,
    inverse_transform,
    save_transform,
    preprocess_pcd,
    build_candidate_features,
    load_model_and_scaler,
    predict_weights,
    weighted_svd_correction,
    evaluate_transform,
    make_registered_cloud,
    plot_alignment,
    plot_bar,
    plot_weight_histogram,
    translation_error_m,
    rotation_error_deg,
)


BASE_DIR = Path(__file__).resolve().parents[1]

V2_CONFIG = BASE_DIR / "configs" / "dataset_samples_synthetic_plus_real_curated_clean_v2.csv"
V3_CONFIG = BASE_DIR / "configs" / "dataset_samples_synthetic_plus_real_curated_clean_v3.csv"

TEST_NAME = "unseen_real_v2_registration_comparison"
OUT_ROOT = BASE_DIR / "results" / "by_test" / TEST_NAME


def resolve_path(path_str):
    p = Path(str(path_str))

    if p.is_absolute():
        return p

    return BASE_DIR / p


def load_transform(path):
    T = np.loadtxt(path)

    if T.shape != (4, 4):
        raise RuntimeError(f"Transform is not 4x4: {path}")

    return T


def get_column(row, candidates, default=None):
    for c in candidates:
        if c in row and not pd.isna(row[c]):
            return row[c]

    return default


def get_real_v2_only_rows():
    if not V2_CONFIG.exists():
        raise FileNotFoundError(f"Missing v2 config: {V2_CONFIG}")

    if not V3_CONFIG.exists():
        raise FileNotFoundError(f"Missing v3 config: {V3_CONFIG}")

    v2 = pd.read_csv(V2_CONFIG)
    v3 = pd.read_csv(V3_CONFIG)

    v2["sample_id"] = v2["sample_id"].astype(str)
    v3["sample_id"] = v3["sample_id"].astype(str)

    v2_real = v2[v2["sample_id"].str.startswith("real_")].copy()
    v3_real_ids = set(v3[v3["sample_id"].str.startswith("real_")]["sample_id"].unique())

    unseen = v2_real[~v2_real["sample_id"].isin(v3_real_ids)].copy()

    return unseen


def list_unseen_samples():
    unseen = get_real_v2_only_rows()

    if len(unseen) == 0:
        print("No v2-only real samples found.")
        return

    print("=" * 100)
    print("Real v2 samples absent from v3")
    print("=" * 100)

    cols = ["sample_id"]

    for extra in ["case_id", "run_name", "difficulty", "positive_rate"]:
        if extra in unseen.columns:
            cols.append(extra)

    print(unseen[cols].drop_duplicates().to_string(index=False))
    print("=" * 100)


def choose_sample_row(sample_id=None):
    unseen = get_real_v2_only_rows()

    if len(unseen) == 0:
        raise RuntimeError("No real v2-only samples found.")

    if sample_id is None:
        medium = unseen[unseen["sample_id"].str.endswith("_medium")]

        if len(medium) > 0:
            row = medium.iloc[0]
        else:
            row = unseen.iloc[0]

        print("No --sample-id given. Auto-selected:")
        print(row["sample_id"])
        return row.to_dict()

    matches = unseen[unseen["sample_id"] == sample_id]

    if len(matches) == 0:
        print("Requested sample_id was not found among v2-only real samples.")
        print()
        list_unseen_samples()
        raise RuntimeError(f"Invalid unseen v2 sample_id: {sample_id}")

    return matches.iloc[0].to_dict()


def make_output_dir(sample_id, args):
    init_tag = (
        f"rx{args.init_rx_deg:+.1f}"
        f"_ry{args.init_ry_deg:+.1f}"
        f"_rz{args.init_rz_deg:+.1f}"
        f"_tx{args.init_tx:+.2f}"
        f"_ty{args.init_ty:+.2f}"
        f"_tz{args.init_tz:+.2f}"
    )
    init_tag = init_tag.replace("+", "p").replace("-", "m").replace(".", "p")

    safe_id = sample_id.replace("/", "_").replace(" ", "_")

    out_dir = OUT_ROOT / f"{safe_id}_{init_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--list-unseen", action="store_true")
    parser.add_argument("--sample-id", type=str, default=None)

    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--normal-radius", type=float, default=None)
    parser.add_argument("--fpfh-radius", type=float, default=None)
    parser.add_argument("--density-radius", type=float, default=None)
    parser.add_argument("--max-corr-distance", type=float, default=None)
    parser.add_argument("--weight-threshold", type=float, default=0.0)

    parser.add_argument("--init-rx-deg", type=float, default=3.0)
    parser.add_argument("--init-ry-deg", type=float, default=-2.0)
    parser.add_argument("--init-rz-deg", type=float, default=12.0)
    parser.add_argument("--init-tx", type=float, default=0.10)
    parser.add_argument("--init-ty", type=float, default=-0.07)
    parser.add_argument("--init-tz", type=float, default=0.04)

    args = parser.parse_args()

    if args.list_unseen:
        list_unseen_samples()
        return

    row = choose_sample_row(args.sample_id)
    sample_id = str(row["sample_id"])

    out_dir = make_output_dir(sample_id, args)

    reference_path = resolve_path(
        get_column(row, ["reference_path", "target_file", "target_path"])
    )
    observation_path = resolve_path(
        get_column(row, ["observation_path", "cad_file", "source_path"])
    )
    label_transform_path = resolve_path(
        get_column(row, ["label_transform_path", "matrix_file"])
    )

    voxel_size = float(args.voxel_size if args.voxel_size is not None else get_column(row, ["voxel_size", "voxel"], 0.05))
    normal_radius = float(args.normal_radius if args.normal_radius is not None else get_column(row, ["normal_radius"], 0.15))
    fpfh_radius = float(args.fpfh_radius if args.fpfh_radius is not None else get_column(row, ["fpfh_radius"], 0.25))
    density_radius = float(args.density_radius if args.density_radius is not None else get_column(row, ["density_radius"], 0.15))
    max_corr_distance = float(args.max_corr_distance if args.max_corr_distance is not None else get_column(row, ["max_corr_distance"], 0.50))

    label_direction = str(get_column(row, ["label_transform_direction"], "obs_to_ref"))

    T_label = load_transform(label_transform_path)

    if label_direction == "ref_to_obs":
        T_label = inverse_transform(T_label)

    # Controlled initial error around pseudo-GT.
    # T_label maps observation/source -> reference/target.
    T_init_error = make_transform(
        rx_deg=args.init_rx_deg,
        ry_deg=args.init_ry_deg,
        rz_deg=args.init_rz_deg,
        tx=args.init_tx,
        ty=args.init_ty,
        tz=args.init_tz,
    )

    T0 = T_init_error @ T_label

    print("=" * 100)
    print("Unseen real v2 registration comparison")
    print("=" * 100)
    print("sample_id:", sample_id)
    print("reference:", reference_path)
    print("observation:", observation_path)
    print("pseudo-GT transform:", label_transform_path)
    print("label direction:", label_direction)
    print("output:", out_dir)
    print("-" * 100)
    print("voxel_size:", voxel_size)
    print("normal_radius:", normal_radius)
    print("fpfh_radius:", fpfh_radius)
    print("density_radius:", density_radius)
    print("max_corr_distance:", max_corr_distance)
    print("weight_threshold:", args.weight_threshold)
    print("-" * 100)
    print("initial perturbation:")
    print("rx ry rz:", args.init_rx_deg, args.init_ry_deg, args.init_rz_deg)
    print("tx ty tz:", args.init_tx, args.init_ty, args.init_tz)
    print("-" * 100)
    print("MLP model:", MODEL_PATH)
    print("MLP scaler:", SCALER_PATH)
    print("=" * 100)

    source = o3d.io.read_point_cloud(str(observation_path))
    target = o3d.io.read_point_cloud(str(reference_path))

    if source.is_empty():
        raise RuntimeError(f"Observation/source cloud is empty: {observation_path}")

    if target.is_empty():
        raise RuntimeError(f"Reference/target cloud is empty: {reference_path}")

    target_down, target_fpfh = preprocess_pcd(
        target,
        voxel_size=voxel_size,
        normal_radius=normal_radius,
        fpfh_radius=fpfh_radius,
    )

    source_down, source_fpfh = preprocess_pcd(
        source,
        voxel_size=voxel_size,
        normal_radius=normal_radius,
        fpfh_radius=fpfh_radius,
    )

    print("Downsampled reference points:", len(target_down.points))
    print("Downsampled observation points:", len(source_down.points))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler = load_model_and_scaler(device)

    # ------------------------------------------------------------------
    # Candidate features + MLP weights
    # ------------------------------------------------------------------

    t_start = time.perf_counter()

    features_df = build_candidate_features(
        source_down=source_down,
        target_down=target_down,
        source_fpfh=source_fpfh,
        target_fpfh=target_fpfh,
        T0=T0,
        max_corr_distance=max_corr_distance,
        density_radius=density_radius,
    )

    weights = predict_weights(model, scaler, features_df, device)
    features_df["mlp_weight"] = weights

    t_features_and_weights = time.perf_counter() - t_start

    print("Candidate correspondences:", len(features_df))
    print("Weight mean:", float(np.mean(weights)))
    print("Weight median:", float(np.median(weights)))
    print("Weight min/max:", float(np.min(weights)), float(np.max(weights)))

    # ------------------------------------------------------------------
    # One-shot unweighted SVD baseline
    # ------------------------------------------------------------------

    t_start = time.perf_counter()

    uniform_weights = np.ones(len(features_df), dtype=np.float64)

    T_unweighted_svd, T_unweighted_delta, unweighted_stats = weighted_svd_correction(
        source_down=source_down,
        target_down=target_down,
        features_df=features_df,
        weights=uniform_weights,
        T0=T0,
        min_weight=0.0,
    )

    t_unweighted_svd = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # MLP-weighted SVD
    # ------------------------------------------------------------------

    t_start = time.perf_counter()

    T_mlp, T_mlp_delta, mlp_stats = weighted_svd_correction(
        source_down=source_down,
        target_down=target_down,
        features_df=features_df,
        weights=weights,
        T0=T0,
        min_weight=args.weight_threshold,
    )

    t_mlp_svd = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Evaluation against pseudo-GT
    # ------------------------------------------------------------------

    eval_initial = evaluate_transform(source_down, target_down, T0, max_corr_distance)
    eval_unweighted = evaluate_transform(source_down, target_down, T_unweighted_svd, max_corr_distance)
    eval_mlp = evaluate_transform(source_down, target_down, T_mlp, max_corr_distance)
    eval_label = evaluate_transform(source_down, target_down, T_label, max_corr_distance)

    method_data = [
        ("initial_T0", T0, eval_initial, 0.0),
        ("one_shot_unweighted_svd", T_unweighted_svd, eval_unweighted, t_unweighted_svd),
        ("mlp_weighted_svd", T_mlp, eval_mlp, t_features_and_weights + t_mlp_svd),
        ("pseudo_gt_label", T_label, eval_label, 0.0),
    ]

    rows = []

    for method, T, ev, runtime in method_data:
        rows.append({
            "sample_id": sample_id,
            "method": method,
            "fitness": ev["fitness"],
            "rmse": ev["rmse"],
            "num_correspondences": ev["num_correspondences"],
            "translation_error_vs_pseudo_gt_m": translation_error_m(T, T_label),
            "rotation_error_vs_pseudo_gt_deg": rotation_error_deg(T, T_label),
            "time_s": runtime,
            "max_corr_distance": max_corr_distance,
            "voxel_size": voxel_size,
        })

    metrics_df = pd.DataFrame(rows)

    metrics_path = out_dir / "registration_comparison_metrics.csv"
    features_path = out_dir / "candidate_features_with_mlp_weights.csv"

    metrics_df.to_csv(metrics_path, index=False)
    features_df.to_csv(features_path, index=False)

    print()
    print("Comparison metrics:")
    print(metrics_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save transforms and clouds
    # ------------------------------------------------------------------

    save_transform(out_dir / "T_initial_T0.txt", T0)
    save_transform(out_dir / "T_one_shot_unweighted_svd.txt", T_unweighted_svd)
    save_transform(out_dir / "T_one_shot_unweighted_delta.txt", T_unweighted_delta)
    save_transform(out_dir / "T_mlp_weighted_svd.txt", T_mlp)
    save_transform(out_dir / "T_mlp_delta.txt", T_mlp_delta)
    save_transform(out_dir / "T_pseudo_gt_label.txt", T_label)

    o3d.io.write_point_cloud(str(out_dir / "reference_downsampled.ply"), target_down)
    o3d.io.write_point_cloud(str(out_dir / "observation_downsampled.ply"), source_down)

    registered = {
        "initial_T0": make_registered_cloud(source_down, T0),
        "one_shot_unweighted_svd": make_registered_cloud(source_down, T_unweighted_svd),
        "mlp_weighted_svd": make_registered_cloud(source_down, T_mlp),
        "pseudo_gt_label": make_registered_cloud(source_down, T_label),
    }

    for name, cloud in registered.items():
        o3d.io.write_point_cloud(str(out_dir / f"observation_registered_{name}.ply"), cloud)

    with (out_dir / "unweighted_svd_stats.json").open("w") as f:
        json.dump(unweighted_stats, f, indent=2)

    with (out_dir / "mlp_weight_stats.json").open("w") as f:
        json.dump(mlp_stats, f, indent=2)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_alignment(
        target_down=target_down,
        source_down=source_down,
        T=T0,
        title="Initial alignment T0",
        out_path=figures_dir / "alignment_initial_T0.png",
    )

    plot_alignment(
        target_down=target_down,
        source_down=source_down,
        T=T_unweighted_svd,
        title="One-shot unweighted SVD",
        out_path=figures_dir / "alignment_one_shot_unweighted_svd.png",
    )

    plot_alignment(
        target_down=target_down,
        source_down=source_down,
        T=T_mlp,
        title="MLP-weighted SVD",
        out_path=figures_dir / "alignment_mlp_weighted_svd.png",
    )

    plot_alignment(
        target_down=target_down,
        source_down=source_down,
        T=T_label,
        title="Pseudo-GT alignment",
        out_path=figures_dir / "alignment_pseudo_gt_label.png",
    )

    plot_bar(
        metrics_df=metrics_df,
        metric="rmse",
        title="RMSE comparison on unseen real v2 case",
        ylabel="RMSE",
        out_path=figures_dir / "bar_rmse_comparison.png",
    )

    plot_bar(
        metrics_df=metrics_df,
        metric="translation_error_vs_pseudo_gt_m",
        title="Translation error vs pseudo-GT",
        ylabel="Translation error (m)",
        out_path=figures_dir / "bar_translation_error_comparison.png",
    )

    plot_bar(
        metrics_df=metrics_df,
        metric="rotation_error_vs_pseudo_gt_deg",
        title="Rotation error vs pseudo-GT",
        ylabel="Rotation error (deg)",
        out_path=figures_dir / "bar_rotation_error_comparison.png",
    )

    plot_weight_histogram(
        weights=weights,
        out_path=figures_dir / "mlp_weight_histogram.png",
    )

    summary = {
        "test_name": TEST_NAME,
        "sample_id": sample_id,
        "output_dir": str(out_dir),
        "v2_config": str(V2_CONFIG),
        "v3_config": str(V3_CONFIG),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "reference_path": str(reference_path),
        "observation_path": str(observation_path),
        "label_transform_path": str(label_transform_path),
        "label_direction": label_direction,
        "voxel_size": voxel_size,
        "normal_radius": normal_radius,
        "fpfh_radius": fpfh_radius,
        "density_radius": density_radius,
        "max_corr_distance": max_corr_distance,
        "weight_threshold": args.weight_threshold,
        "init_rx_deg": args.init_rx_deg,
        "init_ry_deg": args.init_ry_deg,
        "init_rz_deg": args.init_rz_deg,
        "init_tx": args.init_tx,
        "init_ty": args.init_ty,
        "init_tz": args.init_tz,
        "num_reference_downsampled": int(len(target_down.points)),
        "num_observation_downsampled": int(len(source_down.points)),
        "num_candidate_correspondences": int(len(features_df)),
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Saved:")
    print(metrics_path)
    print(features_path)
    print(figures_dir)
    print(out_dir / "summary.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
