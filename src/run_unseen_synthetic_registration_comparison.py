#!/usr/bin/env python3
"""
Unseen synthetic registration comparison.

Methods compared:
1. initial_T0
2. mlp_weighted_svd
3. normal_icp
4. ground_truth

This script:
- generates a new synthetic asymmetric 3D scene not taken from the training CSV
- creates an observation by applying a known transform + noise + partial overlap + outliers
- defines ground truth registration transform as observation -> reference
- creates an imperfect initial transform T0
- runs one-shot MLP-weighted SVD registration
- runs normal point-to-plane ICP from the same T0
- saves comparison metrics and figures

Required trained model:
results/by_test/mlp_synthetic_plus_all_real_v3_final/
    mlp_synthetic_plus_all_real_v3_final_model.pt
    mlp_synthetic_plus_all_real_v3_final_scaler.pkl
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
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_NAME = "mlp_synthetic_plus_all_real_v3_final"
MODEL_DIR = BASE_DIR / "results" / "by_test" / MODEL_NAME

MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_model.pt"
SCALER_PATH = MODEL_DIR / f"{MODEL_NAME}_scaler.pkl"

TEST_NAME = "unseen_synthetic_registration_comparison"

FEATURES = [
    "distance_T0",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]


class CorrespondenceMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_transform(rx_deg=0.0, ry_deg=0.0, rz_deg=0.0, tx=0.0, ty=0.0, tz=0.0):
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ], dtype=np.float64)

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=np.float64)

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)

    return T


def transform_points(points, T):
    points_h = np.hstack([points, np.ones((len(points), 1))])
    return (T @ points_h.T).T[:, :3]


def transform_normals(normals, T):
    R = T[:3, :3]
    out = (R @ normals.T).T

    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0

    return out / n


def inverse_transform(T):
    return np.linalg.inv(T)


def save_transform(path, T):
    np.savetxt(path, T, fmt="%.10f")


def rotation_error_deg(T_est, T_ref):
    R_est = T_est[:3, :3]
    R_ref = T_ref[:3, :3]

    R_delta = R_ref.T @ R_est

    value = (np.trace(R_delta) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)

    return float(np.degrees(np.arccos(value)))


def translation_error_m(T_est, T_ref):
    return float(np.linalg.norm(T_est[:3, 3] - T_ref[:3, 3]))


def mesh_to_points(mesh, n_points):
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return pcd


def create_box(center, size):
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=size[0],
        height=size[1],
        depth=size[2],
    )
    mesh.translate(np.array(center) - np.array(size) / 2.0)
    return mesh


def create_cylinder(center, radius, depth, rotation_deg=(0, 0, 0)):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=depth,
        resolution=48,
        split=4,
    )
    T = make_transform(
        rx_deg=rotation_deg[0],
        ry_deg=rotation_deg[1],
        rz_deg=rotation_deg[2],
        tx=center[0],
        ty=center[1],
        tz=center[2],
    )
    mesh.transform(T)
    return mesh


def create_sphere(center, radius):
    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius,
        resolution=32,
    )
    mesh.translate(center)
    return mesh


def generate_unseen_reference(seed=123, n_points=9000):
    """
    Create an asymmetric synthetic reference scene.

    It is intentionally not just one of the previous simple shapes.
    It combines:
    - stepped blocks
    - a vertical panel
    - a tilted cylinder
    - a small sphere marker
    - a side beam

    This helps avoid perfect symmetry.
    """

    rng = np.random.default_rng(seed)

    parts = []

    # Stairs / stacked blocks
    parts.append((create_box(center=(-0.6, -0.25, 0.10), size=(0.8, 0.35, 0.20)), 1500))
    parts.append((create_box(center=(-0.25, -0.20, 0.32), size=(0.7, 0.35, 0.25)), 1500))
    parts.append((create_box(center=(0.10, -0.15, 0.58), size=(0.65, 0.35, 0.30)), 1500))

    # Vertical panel
    parts.append((create_box(center=(0.55, 0.20, 0.55), size=(0.12, 0.75, 0.90)), 1400))

    # Horizontal side beam
    parts.append((create_box(center=(0.05, 0.55, 0.35), size=(1.10, 0.12, 0.18)), 1100))

    # Tilted cylinder
    parts.append((create_cylinder(center=(-0.35, 0.35, 0.55), radius=0.13, depth=0.9, rotation_deg=(72, 18, 10)), 1200))

    # Small sphere marker
    parts.append((create_sphere(center=(0.65, -0.42, 0.38), radius=0.16), 800))

    pcds = []

    for mesh, count in parts:
        pcds.append(mesh_to_points(mesh, count))

    ref = o3d.geometry.PointCloud()

    for p in pcds:
        ref += p

    ref = ref.voxel_down_sample(voxel_size=0.01)

    points = np.asarray(ref.points)

    if len(points) > n_points:
        idx = rng.choice(len(points), size=n_points, replace=False)
        ref.points = o3d.utility.Vector3dVector(points[idx])

    ref.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30)
    )
    ref.normalize_normals()

    return ref


def create_observation_from_reference(
    reference,
    seed,
    true_transform_ref_to_obs,
    noise_sigma=0.008,
    keep_ratio=0.78,
    outlier_count=700,
):
    rng = np.random.default_rng(seed)

    ref_points = np.asarray(reference.points)

    obs_points = transform_points(ref_points, true_transform_ref_to_obs)

    # Partial overlap: keep a subset according to spatial condition + random sampling
    x = obs_points[:, 0]
    y = obs_points[:, 1]

    spatial_mask = ~((x > np.quantile(x, 0.78)) & (y < np.quantile(y, 0.35)))
    obs_points = obs_points[spatial_mask]

    n_keep = int(keep_ratio * len(obs_points))
    n_keep = max(1000, min(n_keep, len(obs_points)))

    keep_idx = rng.choice(len(obs_points), size=n_keep, replace=False)
    obs_points = obs_points[keep_idx]

    # Noise
    obs_points = obs_points + rng.normal(0.0, noise_sigma, size=obs_points.shape)

    # Add clutter/outliers
    mins = obs_points.min(axis=0)
    maxs = obs_points.max(axis=0)
    span = maxs - mins

    outliers = rng.uniform(
        low=mins - 0.25 * span,
        high=maxs + 0.25 * span,
        size=(outlier_count, 3),
    )

    obs_points = np.vstack([obs_points, outliers])

    obs = o3d.geometry.PointCloud()
    obs.points = o3d.utility.Vector3dVector(obs_points)

    obs.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30)
    )
    obs.normalize_normals()

    return obs


def preprocess_pcd(pcd, voxel_size, normal_radius, fpfh_radius):
    if voxel_size > 0:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = o3d.geometry.PointCloud(pcd)

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=30,
        )
    )
    pcd_down.normalize_normals()

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius,
            max_nn=100,
        )
    )

    return pcd_down, np.asarray(fpfh.data).T


def local_density(points, radius):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    tree = o3d.geometry.KDTreeFlann(pcd)

    density = np.zeros(len(points), dtype=np.float64)

    for i, p in enumerate(points):
        k, _, _ = tree.search_radius_vector_3d(p, radius)
        density[i] = max(k - 1, 0)

    return density


def build_candidate_features(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    T0,
    max_corr_distance,
    density_radius,
):
    source_pts = np.asarray(source_down.points)
    target_pts = np.asarray(target_down.points)

    source_normals = np.asarray(source_down.normals)
    target_normals = np.asarray(target_down.normals)

    source_pts_t0 = transform_points(source_pts, T0)
    source_normals_t0 = transform_normals(source_normals, T0)

    target_tree = o3d.geometry.KDTreeFlann(target_down)

    source_indices = []
    target_indices = []
    distances = []

    for i, p in enumerate(source_pts_t0):
        k, idx, dist2 = target_tree.search_knn_vector_3d(p, 1)

        if k <= 0:
            continue

        j = int(idx[0])
        d = float(np.sqrt(dist2[0]))

        if d <= max_corr_distance:
            source_indices.append(i)
            target_indices.append(j)
            distances.append(d)

    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)

    if len(source_indices) == 0:
        raise RuntimeError("No candidate correspondences found. Increase max_corr_distance.")

    # Mutual nearest neighbor
    source_t0_pcd = o3d.geometry.PointCloud()
    source_t0_pcd.points = o3d.utility.Vector3dVector(source_pts_t0)
    source_t0_tree = o3d.geometry.KDTreeFlann(source_t0_pcd)

    is_mutual = np.zeros(len(source_indices), dtype=np.float64)

    for k_idx, (i, j) in enumerate(zip(source_indices, target_indices)):
        kt, idx_back, _ = source_t0_tree.search_knn_vector_3d(target_pts[j], 1)

        if kt > 0 and int(idx_back[0]) == int(i):
            is_mutual[k_idx] = 1.0

    # Density ratio
    src_density = local_density(source_pts_t0, density_radius)
    tgt_density = local_density(target_pts, density_radius)

    med_src = np.median(src_density[src_density > 0]) if np.any(src_density > 0) else 1.0
    med_tgt = np.median(tgt_density[tgt_density > 0]) if np.any(tgt_density > 0) else 1.0

    eps = 1e-6

    normal_dot_abs = np.abs(
        np.sum(source_normals_t0[source_indices] * target_normals[target_indices], axis=1)
    )

    fpfh_distance = np.linalg.norm(
        source_fpfh[source_indices] - target_fpfh[target_indices],
        axis=1,
    )

    log_density_ratio = np.log(
        ((src_density[source_indices] / med_src) + eps)
        /
        ((tgt_density[target_indices] / med_tgt) + eps)
    )

    features_df = pd.DataFrame({
        "source_index": source_indices,
        "target_index": target_indices,
        "distance_T0": distances,
        "normal_dot_abs": normal_dot_abs,
        "fpfh_distance": fpfh_distance,
        "log_normalized_density_ratio": log_density_ratio,
        "is_mutual_nn": is_mutual,
    })

    return features_df


def load_model_and_scaler(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            f"Train it first with: python3 src/train_final_mlp_synthetic_plus_all_real_v3.py"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}\n"
            f"Train it first with: python3 src/train_final_mlp_synthetic_plus_all_real_v3.py"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    dropout = float(checkpoint.get("dropout", 0.10))
    input_dim = int(checkpoint.get("input_dim", len(FEATURES)))

    model = CorrespondenceMLP(input_dim=input_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with SCALER_PATH.open("rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_weights(model, scaler, features_df, device):
    X = features_df[FEATURES].values
    Xs = scaler.transform(X)

    xt = torch.tensor(Xs, dtype=torch.float32).to(device)

    probs = []

    with torch.no_grad():
        for start in range(0, len(xt), 8192):
            xb = xt[start:start + 8192]
            logits = model(xb)
            prob = torch.sigmoid(logits)
            probs.append(prob.cpu().numpy())

    return np.concatenate(probs, axis=0)


def weighted_svd_correction(source_down, target_down, features_df, weights, T0, min_weight=0.0):
    source_pts = np.asarray(source_down.points)
    target_pts = np.asarray(target_down.points)

    src_idx = features_df["source_index"].values.astype(int)
    tgt_idx = features_df["target_index"].values.astype(int)

    A = transform_points(source_pts[src_idx], T0)
    B = target_pts[tgt_idx]

    w = np.asarray(weights, dtype=np.float64)

    keep = w > min_weight

    A = A[keep]
    B = B[keep]
    w = w[keep]

    if len(A) < 3:
        raise RuntimeError("Not enough weighted correspondences for SVD registration.")

    raw_w = w.copy()

    w = np.clip(w, 1e-6, None)
    w = w / np.sum(w)

    centroid_A = np.sum(A * w[:, None], axis=0)
    centroid_B = np.sum(B * w[:, None], axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ (B_centered * w[:, None])

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    delta_T = np.eye(4)
    delta_T[:3, :3] = R
    delta_T[:3, 3] = t

    final_T = delta_T @ T0

    stats = {
        "num_candidates_total": int(len(features_df)),
        "num_candidates_used": int(len(A)),
        "weight_threshold": float(min_weight),
        "min_raw_weight_used": float(np.min(raw_w)),
        "max_raw_weight_used": float(np.max(raw_w)),
        "mean_raw_weight_used": float(np.mean(raw_w)),
        "median_raw_weight_used": float(np.median(raw_w)),
        "svd_singular_1": float(S[0]),
        "svd_singular_2": float(S[1]),
        "svd_singular_3": float(S[2]),
    }

    return final_T, delta_T, stats


def run_icp(source_down, target_down, T0, max_corr_distance, max_iteration):
    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_corr_distance,
        T0,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )

    return result.transformation, result


def evaluate_transform(source_down, target_down, T, max_corr_distance):
    result = o3d.pipelines.registration.evaluate_registration(
        source_down,
        target_down,
        max_corr_distance,
        T,
    )

    return {
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "num_correspondences": int(len(result.correspondence_set)),
    }


def make_registered_cloud(source_down, T):
    p = o3d.geometry.PointCloud(source_down)
    p.transform(T)
    return p


def plot_alignment(target_down, source_down, T, title, out_path, max_points=2500):
    target_pts = np.asarray(target_down.points)
    source_pts = transform_points(np.asarray(source_down.points), T)

    rng = np.random.default_rng(123)

    if len(target_pts) > max_points:
        idx = rng.choice(len(target_pts), size=max_points, replace=False)
        target_pts = target_pts[idx]

    if len(source_pts) > max_points:
        idx = rng.choice(len(source_pts), size=max_points, replace=False)
        source_pts = source_pts[idx]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], s=1, label="reference")
    ax.scatter(source_pts[:, 0], source_pts[:, 1], source_pts[:, 2], s=1, label="registered observation")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    all_pts = np.vstack([target_pts, source_pts])
    center = all_pts.mean(axis=0)
    span = np.max(np.ptp(all_pts, axis=0))
    span = max(span, 1e-6)

    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bar(metrics_df, metric, title, ylabel, out_path):
    df = metrics_df.copy()

    methods = df["method"].tolist()
    values = df[metric].values

    plt.figure(figsize=(8, 4.5))
    plt.bar(methods, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_weight_histogram(weights, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.hist(weights, bins=40)
    plt.xlabel("MLP reliability weight")
    plt.ylabel("Number of correspondences")
    plt.title("Distribution of MLP correspondence weights")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--normal-radius", type=float, default=0.15)
    parser.add_argument("--fpfh-radius", type=float, default=0.25)
    parser.add_argument("--density-radius", type=float, default=0.15)
    parser.add_argument("--max-corr-distance", type=float, default=0.25)
    parser.add_argument("--icp-max-iteration", type=int, default=50)
    parser.add_argument("--weight-threshold", type=float, default=0.0)

    # Initial registration error applied on top of the ground-truth inverse transform.
    # This controls how wrong T0 is.
    parser.add_argument("--init-rx-deg", type=float, default=1.5)
    parser.add_argument("--init-ry-deg", type=float, default=-1.2)
    parser.add_argument("--init-rz-deg", type=float, default=6.0)
    parser.add_argument("--init-tx", type=float, default=0.08)
    parser.add_argument("--init-ty", type=float, default=-0.05)
    parser.add_argument("--init-tz", type=float, default=0.035)

    args = parser.parse_args()

    init_tag = (
        f"seed_{args.seed}"
        f"_rx{args.init_rx_deg:+.1f}"
        f"_ry{args.init_ry_deg:+.1f}"
        f"_rz{args.init_rz_deg:+.1f}"
        f"_tx{args.init_tx:+.2f}"
        f"_ty{args.init_ty:+.2f}"
        f"_tz{args.init_tz:+.2f}"
    )
    init_tag = init_tag.replace("+", "p").replace("-", "m").replace(".", "p")

    out_dir = BASE_DIR / "results" / "by_test" / TEST_NAME / init_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Unseen synthetic registration comparison")
    print("=" * 100)
    print(f"Output dir: {out_dir}")
    print(f"Seed: {args.seed}")
    print(f"Model: {MODEL_PATH}")
    print(f"Scaler: {SCALER_PATH}")
    print("=" * 100)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler = load_model_and_scaler(device)

    # ------------------------------------------------------------------
    # Generate unseen synthetic case
    # ------------------------------------------------------------------

    reference = generate_unseen_reference(seed=args.seed, n_points=9000)

    T_true_ref_to_obs = make_transform(
        rx_deg=3.0,
        ry_deg=-4.0,
        rz_deg=18.0,
        tx=0.32,
        ty=-0.18,
        tz=0.09,
    )

    T_gt_obs_to_ref = inverse_transform(T_true_ref_to_obs)

    observation = create_observation_from_reference(
        reference=reference,
        seed=args.seed + 1,
        true_transform_ref_to_obs=T_true_ref_to_obs,
        noise_sigma=0.008,
        keep_ratio=0.78,
        outlier_count=700,
    )

    # Initial transform is a controlled perturbation of the true registration transform
    T_init_error = make_transform(
        rx_deg=args.init_rx_deg,
        ry_deg=args.init_ry_deg,
        rz_deg=args.init_rz_deg,
        tx=args.init_tx,
        ty=args.init_ty,
        tz=args.init_tz,
    )

    T0 = T_init_error @ T_gt_obs_to_ref

    o3d.io.write_point_cloud(str(out_dir / "reference_original.ply"), reference)
    o3d.io.write_point_cloud(str(out_dir / "observation_original.ply"), observation)

    save_transform(out_dir / "T_true_ref_to_obs.txt", T_true_ref_to_obs)
    save_transform(out_dir / "T_ground_truth_obs_to_ref.txt", T_gt_obs_to_ref)
    save_transform(out_dir / "T_initial_T0.txt", T0)

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    target_down, target_fpfh = preprocess_pcd(
        reference,
        voxel_size=args.voxel_size,
        normal_radius=args.normal_radius,
        fpfh_radius=args.fpfh_radius,
    )

    source_down, source_fpfh = preprocess_pcd(
        observation,
        voxel_size=args.voxel_size,
        normal_radius=args.normal_radius,
        fpfh_radius=args.fpfh_radius,
    )

    o3d.io.write_point_cloud(str(out_dir / "reference_downsampled.ply"), target_down)
    o3d.io.write_point_cloud(str(out_dir / "observation_downsampled.ply"), source_down)

    print(f"Reference downsampled points:   {len(target_down.points)}")
    print(f"Observation downsampled points: {len(source_down.points)}")

    # ------------------------------------------------------------------
    # Build features + MLP weights
    # ------------------------------------------------------------------

    t_start = time.perf_counter()

    features_df = build_candidate_features(
        source_down=source_down,
        target_down=target_down,
        source_fpfh=source_fpfh,
        target_fpfh=target_fpfh,
        T0=T0,
        max_corr_distance=args.max_corr_distance,
        density_radius=args.density_radius,
    )

    weights = predict_weights(model, scaler, features_df, device)
    features_df["mlp_weight"] = weights

    t_features_and_weights = time.perf_counter() - t_start

    features_df.to_csv(out_dir / "candidate_features_with_mlp_weights.csv", index=False)

    print(f"Candidate correspondences: {len(features_df)}")
    print(f"MLP weight mean:           {np.mean(weights):.4f}")
    print(f"MLP weight median:         {np.median(weights):.4f}")
    print(f"MLP weight min/max:        {np.min(weights):.4f} / {np.max(weights):.4f}")

    # ------------------------------------------------------------------
    # MLP weighted SVD
    # ------------------------------------------------------------------

    t_start = time.perf_counter()

    T_mlp, T_delta, mlp_stats = weighted_svd_correction(
        source_down=source_down,
        target_down=target_down,
        features_df=features_df,
        weights=weights,
        T0=T0,
        min_weight=args.weight_threshold,
    )

    t_mlp_svd = time.perf_counter() - t_start

    save_transform(out_dir / "T_mlp_weighted_svd.txt", T_mlp)
    save_transform(out_dir / "T_mlp_delta.txt", T_delta)

    with (out_dir / "mlp_weight_stats.json").open("w") as f:
        json.dump(mlp_stats, f, indent=2)

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

    save_transform(out_dir / "T_one_shot_unweighted_svd.txt", T_unweighted_svd)
    save_transform(out_dir / "T_one_shot_unweighted_delta.txt", T_unweighted_delta)

    with (out_dir / "unweighted_svd_stats.json").open("w") as f:
        json.dump(unweighted_stats, f, indent=2)

    # ------------------------------------------------------------------
    # Normal ICP
    # ------------------------------------------------------------------

    t_start = time.perf_counter()

    T_icp, icp_result = run_icp(
        source_down=source_down,
        target_down=target_down,
        T0=T0,
        max_corr_distance=args.max_corr_distance,
        max_iteration=args.icp_max_iteration,
    )

    t_icp = time.perf_counter() - t_start

    save_transform(out_dir / "T_normal_icp.txt", T_icp)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    eval_initial = evaluate_transform(source_down, target_down, T0, args.max_corr_distance)
    eval_unweighted_svd = evaluate_transform(source_down, target_down, T_unweighted_svd, args.max_corr_distance)
    eval_mlp = evaluate_transform(source_down, target_down, T_mlp, args.max_corr_distance)
    eval_gt = evaluate_transform(source_down, target_down, T_gt_obs_to_ref, args.max_corr_distance)

    rows = []

    method_data = [
        ("initial_T0", T0, eval_initial, 0.0),
        ("one_shot_unweighted_svd", T_unweighted_svd, eval_unweighted_svd, t_unweighted_svd),
        ("mlp_weighted_svd", T_mlp, eval_mlp, t_features_and_weights + t_mlp_svd),
        ("ground_truth", T_gt_obs_to_ref, eval_gt, 0.0),
    ]

    for method, T, ev, runtime in method_data:
        rows.append({
            "method": method,
            "fitness": ev["fitness"],
            "rmse": ev["rmse"],
            "num_correspondences": ev["num_correspondences"],
            "translation_error_m": translation_error_m(T, T_gt_obs_to_ref),
            "rotation_error_deg": rotation_error_deg(T, T_gt_obs_to_ref),
            "time_s": runtime,
            "max_corr_distance": args.max_corr_distance,
            "voxel_size": args.voxel_size,
        })

    metrics_df = pd.DataFrame(rows)

    metrics_path = out_dir / "registration_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\nComparison metrics:")
    print(metrics_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save registered point clouds
    # ------------------------------------------------------------------

    registered = {
        "initial_T0": make_registered_cloud(source_down, T0),
        "one_shot_unweighted_svd": make_registered_cloud(source_down, T_unweighted_svd),
        "mlp_weighted_svd": make_registered_cloud(source_down, T_mlp),
        "ground_truth": make_registered_cloud(source_down, T_gt_obs_to_ref),
    }

    for name, cloud in registered.items():
        o3d.io.write_point_cloud(str(out_dir / f"observation_registered_{name}.ply"), cloud)

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
        title="One-shot unweighted SVD registration",
        out_path=figures_dir / "alignment_one_shot_unweighted_svd.png",
    )

    plot_alignment(
        target_down=target_down,
        source_down=source_down,
        T=T_mlp,
        title="MLP-weighted SVD registration",
        out_path=figures_dir / "alignment_mlp_weighted_svd.png",
    )

    plot_alignment(
        target_down=target_down,
        source_down=source_down,
        T=T_icp,
        title="Normal ICP registration",
        out_path=figures_dir / "alignment_normal_icp.png",
    )

    plot_alignment(
        target_down=target_down,
        source_down=source_down,
        T=T_gt_obs_to_ref,
        title="Ground-truth alignment",
        out_path=figures_dir / "alignment_ground_truth.png",
    )

    plot_bar(
        metrics_df=metrics_df,
        metric="rmse",
        title="RMSE comparison",
        ylabel="RMSE",
        out_path=figures_dir / "bar_rmse_comparison.png",
    )

    plot_bar(
        metrics_df=metrics_df,
        metric="translation_error_m",
        title="Translation error vs ground truth",
        ylabel="Translation error (m)",
        out_path=figures_dir / "bar_translation_error_comparison.png",
    )

    plot_bar(
        metrics_df=metrics_df,
        metric="rotation_error_deg",
        title="Rotation error vs ground truth",
        ylabel="Rotation error (deg)",
        out_path=figures_dir / "bar_rotation_error_comparison.png",
    )

    plot_bar(
        metrics_df=metrics_df,
        metric="time_s",
        title="Runtime comparison",
        ylabel="Time (s)",
        out_path=figures_dir / "bar_runtime_comparison.png",
    )

    plot_weight_histogram(
        weights=weights,
        out_path=figures_dir / "mlp_weight_histogram.png",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = {
        "test_name": TEST_NAME,
        "seed": args.seed,
        "output_dir": str(out_dir),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "voxel_size": args.voxel_size,
        "normal_radius": args.normal_radius,
        "fpfh_radius": args.fpfh_radius,
        "density_radius": args.density_radius,
        "max_corr_distance": args.max_corr_distance,
        "icp_max_iteration": args.icp_max_iteration,
        "weight_threshold": args.weight_threshold,
        "init_rx_deg": args.init_rx_deg,
        "init_ry_deg": args.init_ry_deg,
        "init_rz_deg": args.init_rz_deg,
        "init_tx": args.init_tx,
        "init_ty": args.init_ty,
        "init_tz": args.init_tz,
        "num_reference_downsampled": len(target_down.points),
        "num_observation_downsampled": len(source_down.points),
        "num_candidate_correspondences": int(len(features_df)),
        "mlp_weight_stats": mlp_stats,
        "metrics_csv": str(metrics_path),
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs:")
    print(metrics_path)
    print(figures_dir)
    print(out_dir / "summary.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
