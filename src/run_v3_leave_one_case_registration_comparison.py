#!/usr/bin/env python3
"""
Leave-one-real-case-out registration comparison on clean v3.

For each real v3 case:
- hold out the entire real case from training
- train MLP on synthetic + all other real v3 cases
- test registration on the held-out real case
- compare:
    1. initial_T0
    2. one_shot_unweighted_svd
    3. mlp_weighted_svd
    4. pseudo_gt_label

No normal ICP is used, because the fair comparison is:
one-shot unweighted SVD vs one-shot MLP-weighted SVD.

Outputs:
results/by_test/v3_leave_one_case_registration_comparison/
    registration_metrics_by_case.csv
    registration_metrics_average.csv
    registration_improvement_vs_unweighted.csv
    figures/
    folds/<case_id>/
"""

from pathlib import Path
import argparse
import copy
import json
import pickle
import time
import re
import numpy as np
import pandas as pd

import open3d as o3d

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_unseen_synthetic_registration_comparison import (
    FEATURES,
    CorrespondenceMLP,
    make_transform,
    inverse_transform,
    save_transform,
    preprocess_pcd,
    build_candidate_features,
    predict_weights,
    weighted_svd_correction,
    evaluate_transform,
    make_registered_cloud,
    plot_alignment,
    plot_weight_histogram,
    translation_error_m,
    rotation_error_deg,
    transform_points,
)


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = BASE_DIR / "configs" / "dataset_samples_synthetic_plus_real_curated_clean_v3.csv"
CORRESPONDENCE_CSV = (
    BASE_DIR
    / "results"
    / "learning_data_synthetic_plus_real_curated_clean_v3"
    / "all_correspondences.csv"
)

TEST_NAME = "v3_leave_one_case_registration_comparison"
OUT_ROOT = BASE_DIR / "results" / "by_test" / TEST_NAME

TARGET = "target_weight"

RANDOM_STATE = 42
VAL_FRACTION_BY_SAMPLE_ID = 0.20

LR = 1e-3
DROPOUT = 0.10
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1024
EPOCHS = 120
PATIENCE = 12


def safe_name(s):
    s = str(s)
    s = s.replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def strip_difficulty(sample_id):
    sample_id = str(sample_id)

    for suffix in ["_easy", "_medium", "_hard"]:
        if sample_id.endswith(suffix):
            return sample_id[:-len(suffix)]

    return sample_id


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


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, score)


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def predict_score(model, X, device):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    scores = []

    with torch.no_grad():
        for start in range(0, len(X_tensor), 8192):
            xb = X_tensor[start:start + 8192]
            logits = model(xb)
            prob = torch.sigmoid(logits)
            scores.append(prob.cpu().numpy())

    return np.concatenate(scores, axis=0)


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    df = pd.read_csv(CONFIG_PATH)
    df["sample_id"] = df["sample_id"].astype(str)

    if "case_id" in df.columns:
        df["case_id_auto"] = df["case_id"].astype(str)
        bad = df["case_id_auto"].isin(["nan", "None", ""])
        df.loc[bad, "case_id_auto"] = df.loc[bad, "sample_id"].apply(strip_difficulty)
    else:
        df["case_id_auto"] = df["sample_id"].apply(strip_difficulty)

    return df


def load_correspondences(config_df):
    if not CORRESPONDENCE_CSV.exists():
        raise FileNotFoundError(f"Missing correspondence CSV: {CORRESPONDENCE_CSV}")

    df = pd.read_csv(CORRESPONDENCE_CSV)
    df["sample_id"] = df["sample_id"].astype(str)

    required = ["sample_id", TARGET] + FEATURES
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing columns in correspondence CSV: {missing}")

    sample_to_case = dict(zip(config_df["sample_id"], config_df["case_id_auto"]))

    df["case_id_auto"] = df["sample_id"].map(sample_to_case)
    df["case_id_auto"] = df["case_id_auto"].fillna(df["sample_id"].apply(strip_difficulty))
    df["is_real"] = df["sample_id"].str.startswith("real_")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).copy()
    df[TARGET] = df[TARGET].astype(int)

    return df


def choose_test_row(config_df, case_id, test_difficulty):
    case_rows = config_df[
        (config_df["sample_id"].str.startswith("real_"))
        & (config_df["case_id_auto"] == case_id)
    ].copy()

    if len(case_rows) == 0:
        raise RuntimeError(f"No config rows found for case_id={case_id}")

    if test_difficulty != "any":
        wanted_suffix = "_" + test_difficulty
        preferred = case_rows[case_rows["sample_id"].str.endswith(wanted_suffix)]

        if len(preferred) > 0:
            return preferred.iloc[0].to_dict()

    medium = case_rows[case_rows["sample_id"].str.endswith("_medium")]

    if len(medium) > 0:
        return medium.iloc[0].to_dict()

    return case_rows.iloc[0].to_dict()


def split_train_val_by_sample_id(train_df, seed):
    sample_ids = np.array(sorted(train_df["sample_id"].astype(str).unique()))

    rng = np.random.default_rng(seed)
    rng.shuffle(sample_ids)

    n_val = max(1, int(round(VAL_FRACTION_BY_SAMPLE_ID * len(sample_ids))))
    val_ids = set(sample_ids[:n_val].tolist())

    val_df = train_df[train_df["sample_id"].isin(val_ids)].copy()
    tr_df = train_df[~train_df["sample_id"].isin(val_ids)].copy()

    return tr_df, val_df, sorted(val_ids)


def train_fold_model(all_corr_df, heldout_case_id, out_dir, seed):
    """
    Train on:
    - all synthetic samples
    - all real v3 samples except heldout_case_id
    """

    fold_train_df = all_corr_df[
        (~all_corr_df["is_real"])
        | (all_corr_df["case_id_auto"] != heldout_case_id)
    ].copy()

    fold_test_like = all_corr_df[
        (all_corr_df["is_real"])
        & (all_corr_df["case_id_auto"] == heldout_case_id)
    ].copy()

    if len(fold_train_df) == 0:
        raise RuntimeError("Empty training dataframe.")

    if len(fold_test_like) == 0:
        raise RuntimeError(f"No held-out correspondence rows for case: {heldout_case_id}")

    tr_df, val_df, val_sample_ids = split_train_val_by_sample_id(fold_train_df, seed)

    scaler = StandardScaler()
    scaler.fit(tr_df[FEATURES].values)

    X_train = scaler.transform(tr_df[FEATURES].values)
    y_train = tr_df[TARGET].values.astype(np.float32)

    X_val = scaler.transform(val_df[FEATURES].values)
    y_val = val_df[TARGET].values.astype(np.float32)

    num_pos = float(np.sum(y_train == 1))
    num_neg = float(np.sum(y_train == 0))

    pos_weight_value = num_neg / num_pos if num_pos > 0 else 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CorrespondenceMLP(input_dim=len(FEATURES), dropout=DROPOUT).to(device)

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    patience_count = 0
    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        val_score = predict_score(model, X_val, device)
        val_auc = safe_auc(y_val.astype(int), val_score)

        improved = val_loss < best_val_loss - 1e-6

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "best_epoch_so_far": best_epoch,
            "best_val_loss_so_far": best_val_loss,
            "patience_count": patience_count,
            "pos_weight": pos_weight_value,
            "train_positive_rate": float(np.mean(y_train)),
            "val_positive_rate": float(np.mean(y_val)),
            "heldout_case_id": heldout_case_id,
        })

        if patience_count >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    with (out_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": FEATURES,
            "dropout": DROPOUT,
            "input_dim": len(FEATURES),
        },
        out_dir / "model.pt",
    )

    metadata = {
        "heldout_case_id": heldout_case_id,
        "train_rows": int(len(tr_df)),
        "val_rows": int(len(val_df)),
        "heldout_corr_rows_not_used_for_training": int(len(fold_test_like)),
        "train_positive_rate": float(np.mean(y_train)),
        "val_positive_rate": float(np.mean(y_val)),
        "pos_weight": float(pos_weight_value),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs_run": int(history_rows[-1]["epoch"]),
        "val_sample_ids": val_sample_ids,
        "features": FEATURES,
    }

    with (out_dir / "training_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    return model, scaler, metadata, device


def centered_initial_transform(T_label, center, args):
    """
    Apply the initial perturbation around the object/reference centroid.

    This avoids the unrealistic effect where a small rotation about the world origin
    creates a huge translation displacement when the object is far from the origin.
    """

    E = make_transform(
        rx_deg=args.init_rx_deg,
        ry_deg=args.init_ry_deg,
        rz_deg=args.init_rz_deg,
        tx=args.init_tx,
        ty=args.init_ty,
        tz=args.init_tz,
    )

    T_center = np.eye(4)
    T_center[:3, 3] = center

    T_uncenter = np.eye(4)
    T_uncenter[:3, 3] = -center

    return T_center @ E @ T_uncenter @ T_label


def point_action_error_stats(source_down, T_est, T_ref):
    pts = np.asarray(source_down.points)

    A = transform_points(pts, T_est)
    B = transform_points(pts, T_ref)

    d = np.linalg.norm(A - B, axis=1)

    return {
        "point_action_mean_error_vs_pseudo_gt_m": float(np.mean(d)),
        "point_action_median_error_vs_pseudo_gt_m": float(np.median(d)),
        "point_action_rmse_vs_pseudo_gt_m": float(np.sqrt(np.mean(d ** 2))),
    }


def run_registration_for_case(model, scaler, device, test_row, heldout_case_id, out_dir, args):
    sample_id = str(test_row["sample_id"])

    reference_path = resolve_path(
        get_column(test_row, ["reference_path", "target_file", "target_path"])
    )
    observation_path = resolve_path(
        get_column(test_row, ["observation_path", "cad_file", "source_path"])
    )
    label_transform_path = resolve_path(
        get_column(test_row, ["label_transform_path", "matrix_file"])
    )

    voxel_size = float(args.voxel_size if args.voxel_size is not None else get_column(test_row, ["voxel_size", "voxel"], 0.05))
    normal_radius = float(args.normal_radius if args.normal_radius is not None else get_column(test_row, ["normal_radius"], 0.15))
    fpfh_radius = float(args.fpfh_radius if args.fpfh_radius is not None else get_column(test_row, ["fpfh_radius"], 0.25))
    density_radius = float(args.density_radius if args.density_radius is not None else get_column(test_row, ["density_radius"], 0.15))
    max_corr_distance = float(args.max_corr_distance if args.max_corr_distance is not None else get_column(test_row, ["max_corr_distance"], 0.50))

    label_direction = str(get_column(test_row, ["label_transform_direction"], "obs_to_ref"))

    T_label = load_transform(label_transform_path)

    if label_direction == "ref_to_obs":
        T_label = inverse_transform(T_label)

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

    target_center = np.mean(np.asarray(target_down.points), axis=0)
    T0 = centered_initial_transform(T_label, target_center, args)

    # ------------------------------------------------------------------
    # Features and weights
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

    # ------------------------------------------------------------------
    # One-shot unweighted SVD
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
    # Evaluation
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
        row = {
            "heldout_case_id": heldout_case_id,
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
            "num_candidate_correspondences": int(len(features_df)),
            "weight_mean": float(np.mean(weights)),
            "weight_median": float(np.median(weights)),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
        }

        row.update(point_action_error_stats(source_down, T, T_label))
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Save per-case outputs
    # ------------------------------------------------------------------

    metrics_df.to_csv(out_dir / "registration_comparison_metrics.csv", index=False)
    features_df.to_csv(out_dir / "candidate_features_with_mlp_weights.csv", index=False)

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

    metadata = {
        "heldout_case_id": heldout_case_id,
        "sample_id": sample_id,
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

    with (out_dir / "registration_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    if args.save_case_figures:
        figures_dir = out_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        plot_alignment(
            target_down=target_down,
            source_down=source_down,
            T=T0,
            title=f"Initial T0 - {heldout_case_id}",
            out_path=figures_dir / "alignment_initial_T0.png",
        )

        plot_alignment(
            target_down=target_down,
            source_down=source_down,
            T=T_unweighted_svd,
            title=f"One-shot unweighted SVD - {heldout_case_id}",
            out_path=figures_dir / "alignment_one_shot_unweighted_svd.png",
        )

        plot_alignment(
            target_down=target_down,
            source_down=source_down,
            T=T_mlp,
            title=f"MLP-weighted SVD - {heldout_case_id}",
            out_path=figures_dir / "alignment_mlp_weighted_svd.png",
        )

        plot_alignment(
            target_down=target_down,
            source_down=source_down,
            T=T_label,
            title=f"Pseudo-GT - {heldout_case_id}",
            out_path=figures_dir / "alignment_pseudo_gt_label.png",
        )

        plot_weight_histogram(
            weights=weights,
            out_path=figures_dir / "mlp_weight_histogram.png",
        )

    return metrics_df


def flatten_columns(df):
    df.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in df.columns
    ]

    return df


def compute_average_tables(all_metrics_df, out_root):
    metric_cols = [
        "fitness",
        "rmse",
        "translation_error_vs_pseudo_gt_m",
        "rotation_error_vs_pseudo_gt_deg",
        "point_action_mean_error_vs_pseudo_gt_m",
        "point_action_median_error_vs_pseudo_gt_m",
        "point_action_rmse_vs_pseudo_gt_m",
        "time_s",
        "num_candidate_correspondences",
    ]

    avg = (
        all_metrics_df
        .groupby("method")[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    avg = flatten_columns(avg)

    method_order = {
        "initial_T0": 0,
        "one_shot_unweighted_svd": 1,
        "mlp_weighted_svd": 2,
        "pseudo_gt_label": 3,
    }

    avg["method_order"] = avg["method"].map(method_order)
    avg = avg.sort_values("method_order").drop(columns=["method_order"])

    avg_path = out_root / "registration_metrics_average.csv"
    avg.to_csv(avg_path, index=False)

    # Improvement table: positive values mean MLP is better than unweighted SVD.
    rows = []

    for case_id, g in all_metrics_df.groupby("heldout_case_id"):
        g = g.set_index("method")

        if "one_shot_unweighted_svd" not in g.index or "mlp_weighted_svd" not in g.index:
            continue

        u = g.loc["one_shot_unweighted_svd"]
        m = g.loc["mlp_weighted_svd"]

        rows.append({
            "heldout_case_id": case_id,
            "sample_id": m["sample_id"],
            "rmse_improvement_mlp_vs_unweighted": u["rmse"] - m["rmse"],
            "translation_error_improvement_mlp_vs_unweighted_m": (
                u["translation_error_vs_pseudo_gt_m"]
                - m["translation_error_vs_pseudo_gt_m"]
            ),
            "rotation_error_improvement_mlp_vs_unweighted_deg": (
                u["rotation_error_vs_pseudo_gt_deg"]
                - m["rotation_error_vs_pseudo_gt_deg"]
            ),
            "point_action_rmse_improvement_mlp_vs_unweighted_m": (
                u["point_action_rmse_vs_pseudo_gt_m"]
                - m["point_action_rmse_vs_pseudo_gt_m"]
            ),
            "mlp_better_rmse": bool(m["rmse"] < u["rmse"]),
            "mlp_better_translation_error": bool(
                m["translation_error_vs_pseudo_gt_m"]
                < u["translation_error_vs_pseudo_gt_m"]
            ),
            "mlp_better_rotation_error": bool(
                m["rotation_error_vs_pseudo_gt_deg"]
                < u["rotation_error_vs_pseudo_gt_deg"]
            ),
            "mlp_better_point_action_rmse": bool(
                m["point_action_rmse_vs_pseudo_gt_m"]
                < u["point_action_rmse_vs_pseudo_gt_m"]
            ),
        })

    improvement_df = pd.DataFrame(rows)
    improvement_path = out_root / "registration_improvement_vs_unweighted.csv"
    improvement_df.to_csv(improvement_path, index=False)

    improvement_avg = pd.DataFrame([{
        "num_cases": int(len(improvement_df)),
        "mean_rmse_improvement_mlp_vs_unweighted": float(improvement_df["rmse_improvement_mlp_vs_unweighted"].mean()),
        "mean_translation_error_improvement_mlp_vs_unweighted_m": float(improvement_df["translation_error_improvement_mlp_vs_unweighted_m"].mean()),
        "mean_rotation_error_improvement_mlp_vs_unweighted_deg": float(improvement_df["rotation_error_improvement_mlp_vs_unweighted_deg"].mean()),
        "mean_point_action_rmse_improvement_mlp_vs_unweighted_m": float(improvement_df["point_action_rmse_improvement_mlp_vs_unweighted_m"].mean()),
        "win_rate_rmse": float(improvement_df["mlp_better_rmse"].mean()),
        "win_rate_translation_error": float(improvement_df["mlp_better_translation_error"].mean()),
        "win_rate_rotation_error": float(improvement_df["mlp_better_rotation_error"].mean()),
        "win_rate_point_action_rmse": float(improvement_df["mlp_better_point_action_rmse"].mean()),
    }])

    improvement_avg_path = out_root / "registration_improvement_average_vs_unweighted.csv"
    improvement_avg.to_csv(improvement_avg_path, index=False)

    return avg, improvement_df, improvement_avg


def plot_average_metric(avg_df, metric_mean_col, title, ylabel, out_path):
    methods = avg_df["method"].tolist()
    values = avg_df[metric_mean_col].values

    plt.figure(figsize=(9, 4.8))
    plt.bar(methods, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_improvement_by_case(improvement_df, col, title, ylabel, out_path):
    df = improvement_df.copy()
    df = df.sort_values(col)

    labels = [safe_name(x).replace("real_", "")[:35] for x in df["heldout_case_id"]]

    plt.figure(figsize=(11, 5.5))
    plt.bar(labels, df[col].values)
    plt.axhline(0.0, linestyle="--")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_global_figures(avg_df, improvement_df, out_root):
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_average_metric(
        avg_df,
        "rmse_mean",
        "Average RMSE comparison - leave-one-real-case-out v3",
        "Average RMSE",
        figures_dir / "average_rmse_comparison.png",
    )

    plot_average_metric(
        avg_df,
        "translation_error_vs_pseudo_gt_m_mean",
        "Average translation error vs pseudo-GT",
        "Average translation error (m)",
        figures_dir / "average_translation_error_comparison.png",
    )

    plot_average_metric(
        avg_df,
        "rotation_error_vs_pseudo_gt_deg_mean",
        "Average rotation error vs pseudo-GT",
        "Average rotation error (deg)",
        figures_dir / "average_rotation_error_comparison.png",
    )

    plot_average_metric(
        avg_df,
        "point_action_rmse_vs_pseudo_gt_m_mean",
        "Average point-action RMSE vs pseudo-GT",
        "Average point-action RMSE (m)",
        figures_dir / "average_point_action_rmse_comparison.png",
    )

    plot_improvement_by_case(
        improvement_df,
        "rmse_improvement_mlp_vs_unweighted",
        "MLP improvement over unweighted SVD by case - RMSE",
        "RMSE improvement, positive = MLP better",
        figures_dir / "improvement_by_case_rmse.png",
    )

    plot_improvement_by_case(
        improvement_df,
        "translation_error_improvement_mlp_vs_unweighted_m",
        "MLP improvement over unweighted SVD by case - translation",
        "Translation error improvement (m), positive = MLP better",
        figures_dir / "improvement_by_case_translation.png",
    )

    plot_improvement_by_case(
        improvement_df,
        "rotation_error_improvement_mlp_vs_unweighted_deg",
        "MLP improvement over unweighted SVD by case - rotation",
        "Rotation error improvement (deg), positive = MLP better",
        figures_dir / "improvement_by_case_rotation.png",
    )

    plot_improvement_by_case(
        improvement_df,
        "point_action_rmse_improvement_mlp_vs_unweighted_m",
        "MLP improvement over unweighted SVD by case - point-action RMSE",
        "Point-action RMSE improvement (m), positive = MLP better",
        figures_dir / "improvement_by_case_point_action_rmse.png",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-difficulty", type=str, default="medium", choices=["easy", "medium", "hard", "any"])
    parser.add_argument("--only-case", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=0, help="0 means all folds.")

    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--normal-radius", type=float, default=None)
    parser.add_argument("--fpfh-radius", type=float, default=None)
    parser.add_argument("--density-radius", type=float, default=None)
    parser.add_argument("--max-corr-distance", type=float, default=None)
    parser.add_argument("--weight-threshold", type=float, default=0.0)

    parser.add_argument("--init-rx-deg", type=float, default=1.0)
    parser.add_argument("--init-ry-deg", type=float, default=-1.0)
    parser.add_argument("--init-rz-deg", type=float, default=5.0)
    parser.add_argument("--init-tx", type=float, default=0.03)
    parser.add_argument("--init-ty", type=float, default=-0.03)
    parser.add_argument("--init-tz", type=float, default=0.02)

    parser.add_argument("--save-case-figures", action="store_true")

    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    config_df = load_config()
    corr_df = load_correspondences(config_df)

    real_cases = sorted(
        config_df[config_df["sample_id"].str.startswith("real_")]["case_id_auto"].unique()
    )

    if args.only_case is not None:
        real_cases = [c for c in real_cases if c == args.only_case]

        if len(real_cases) == 0:
            raise RuntimeError(f"Requested --only-case not found: {args.only_case}")

    if args.max_folds and args.max_folds > 0:
        real_cases = real_cases[:args.max_folds]

    print("=" * 100)
    print(TEST_NAME)
    print("=" * 100)
    print("Config:", CONFIG_PATH)
    print("Correspondences:", CORRESPONDENCE_CSV)
    print("Number of real held-out cases:", len(real_cases))
    print("Test difficulty:", args.test_difficulty)
    print("Output:", OUT_ROOT)
    print("-" * 100)
    print("Initial perturbation around object centroid:")
    print("rx ry rz:", args.init_rx_deg, args.init_ry_deg, args.init_rz_deg)
    print("tx ty tz:", args.init_tx, args.init_ty, args.init_tz)
    print("=" * 100)

    all_case_metrics = []

    for fold_idx, heldout_case_id in enumerate(real_cases, start=1):
        case_safe = safe_name(heldout_case_id)
        fold_dir = OUT_ROOT / "folds" / case_safe
        fold_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 100)
        print(f"Fold {fold_idx}/{len(real_cases)}")
        print("Held-out case:", heldout_case_id)
        print("=" * 100)

        test_row = choose_test_row(config_df, heldout_case_id, args.test_difficulty)

        print("Test sample:", test_row["sample_id"])

        model, scaler, train_metadata, device = train_fold_model(
            all_corr_df=corr_df,
            heldout_case_id=heldout_case_id,
            out_dir=fold_dir,
            seed=RANDOM_STATE + fold_idx,
        )

        print("Training rows:", train_metadata["train_rows"])
        print("Validation rows:", train_metadata["val_rows"])
        print("Best epoch:", train_metadata["best_epoch"])
        print("Best val loss:", train_metadata["best_val_loss"])

        metrics_df = run_registration_for_case(
            model=model,
            scaler=scaler,
            device=device,
            test_row=test_row,
            heldout_case_id=heldout_case_id,
            out_dir=fold_dir,
            args=args,
        )

        print()
        print(metrics_df[[
            "method",
            "rmse",
            "translation_error_vs_pseudo_gt_m",
            "rotation_error_vs_pseudo_gt_deg",
            "point_action_rmse_vs_pseudo_gt_m",
        ]].to_string(index=False))

        all_case_metrics.append(metrics_df)

    all_metrics_df = pd.concat(all_case_metrics, ignore_index=True)

    all_metrics_path = OUT_ROOT / "registration_metrics_by_case.csv"
    all_metrics_df.to_csv(all_metrics_path, index=False)

    avg_df, improvement_df, improvement_avg_df = compute_average_tables(all_metrics_df, OUT_ROOT)
    make_global_figures(avg_df, improvement_df, OUT_ROOT)

    summary = {
        "test_name": TEST_NAME,
        "num_cases": int(len(real_cases)),
        "test_difficulty": args.test_difficulty,
        "config_path": str(CONFIG_PATH),
        "correspondence_csv": str(CORRESPONDENCE_CSV),
        "output_root": str(OUT_ROOT),
        "init_rx_deg": args.init_rx_deg,
        "init_ry_deg": args.init_ry_deg,
        "init_rz_deg": args.init_rz_deg,
        "init_tx": args.init_tx,
        "init_ty": args.init_ty,
        "init_tz": args.init_tz,
        "weight_threshold": args.weight_threshold,
        "features": FEATURES,
    }

    with (OUT_ROOT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 100)
    print("FINAL AVERAGE TABLE")
    print("=" * 100)
    print(avg_df.to_string(index=False))

    print()
    print("=" * 100)
    print("MLP IMPROVEMENT VS UNWEIGHTED SVD")
    print("Positive values mean MLP-weighted SVD is better.")
    print("=" * 100)
    print(improvement_avg_df.to_string(index=False))

    print()
    print("Saved:")
    print(all_metrics_path)
    print(OUT_ROOT / "registration_metrics_average.csv")
    print(OUT_ROOT / "registration_improvement_vs_unweighted.csv")
    print(OUT_ROOT / "registration_improvement_average_vs_unweighted.csv")
    print(OUT_ROOT / "figures")
    print("=" * 100)


if __name__ == "__main__":
    main()
