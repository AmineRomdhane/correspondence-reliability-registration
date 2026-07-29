#!/usr/bin/env python3
"""
Extract correspondence-level training data for learned correspondence weighting.

For each candidate correspondence q_i <-> p_j, the script extracts:

Input features:
- distance_T0
- point_to_plane_residual
- normal_dot_abs
- fpfh_distance
- log_normalized_density_ratio
- is_mutual_nn

Training labels / debug:
- label_distance
- target_weight

Important convention:
- reference cloud P is the fixed target cloud.
- observation cloud Q is the moving source cloud.
- T_label must map observation -> reference.
- T0 must also map observation -> reference, but it can be imperfect.

For synthetic data where T_true maps reference -> observation,
set label_transform_direction = ref_to_obs in the CSV config.
"""

import csv
import math
import time
from pathlib import Path

import numpy as np
import open3d as o3d


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "dataset_samples_synthetic_plus_real.csv"
OUT_DIR = BASE_DIR / "results" / "learning_data_synthetic_plus_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "distance_T0",
    "point_to_plane_residual",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]

OUTPUT_COLUMNS = [
    "sample_id",
    "scenario",
    "source_index",
    "target_index",
    "distance_T0",
    "point_to_plane_residual",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
    "label_distance",
    "target_weight",
]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def read_transform(path: Path) -> np.ndarray:
    T = np.loadtxt(path)
    if T.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4, got {T.shape}: {path}")
    return T.astype(float)


def parse_float(row, key, default):
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def parse_int(row, key, default):
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


def make_random_small_transform(rot_deg: float, trans_m: float, rng: np.random.Generator) -> np.ndarray:
    """
    Create a small random SE(3) perturbation.
    """
    T = np.eye(4)

    if rot_deg > 0:
        angle = np.deg2rad(rot_deg) * rng.uniform(-1.0, 1.0)
        axis = rng.normal(size=3)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-12:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / axis_norm

        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])

        R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
        T[:3, :3] = R

    if trans_m > 0:
        T[:3, 3] = rng.uniform(-trans_m, trans_m, size=3)

    return T


def load_and_downsample(path: Path, voxel_size: float) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Empty point cloud: {path}")

    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)

    return cloud


def estimate_normals(cloud: o3d.geometry.PointCloud, radius: float, max_nn: int = 30):
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )
    cloud.normalize_normals()


def compute_fpfh(cloud: o3d.geometry.PointCloud, radius: float, max_nn: int = 100) -> np.ndarray:
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        ),
    )
    return np.asarray(fpfh.data).T  # shape: (N, 33)


def make_cloud_from_points(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def compute_density_counts(points: np.ndarray, radius: float) -> np.ndarray:
    """
    Density = number of neighbors inside a radius.
    Includes the point itself.
    """
    cloud = make_cloud_from_points(points)
    tree = o3d.geometry.KDTreeFlann(cloud)

    counts = np.zeros(len(points), dtype=float)

    for i, p in enumerate(points):
        k, _, _ = tree.search_radius_vector_3d(p, radius)
        counts[i] = float(k)

    return counts


def build_reverse_nearest_indices(target_points: np.ndarray, source_points_T0: np.ndarray) -> np.ndarray:
    """
    For every target point p_j, find nearest transformed source point T0 q_i.
    Returns reverse_nn[j] = i.
    """
    source_T0_cloud = make_cloud_from_points(source_points_T0)
    source_T0_tree = o3d.geometry.KDTreeFlann(source_T0_cloud)

    reverse_nn = np.full(len(target_points), -1, dtype=int)

    for j, p in enumerate(target_points):
        k, idx, _ = source_T0_tree.search_knn_vector_3d(p, 1)
        if k > 0:
            reverse_nn[j] = int(idx[0])

    return reverse_nn


def get_label_transform(row) -> np.ndarray:
    label_path = resolve_path(row["label_transform_path"])
    T_raw = read_transform(label_path)

    direction = row.get("label_transform_direction", "obs_to_ref").strip()

    if direction == "obs_to_ref":
        return T_raw

    if direction == "ref_to_obs":
        return np.linalg.inv(T_raw)

    raise ValueError(
        "label_transform_direction must be either 'obs_to_ref' or 'ref_to_obs'"
    )


def get_initial_transform(row, T_label: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mode = row.get("T0_mode", "identity").strip()

    if mode == "identity":
        return np.eye(4)

    if mode == "label":
        print("WARNING: T0_mode=label creates very easy correspondences. Use only for debugging.")
        return T_label.copy()

    if mode == "perturb_label":
        rot_deg = parse_float(row, "perturb_rot_deg", 5.0)
        trans_m = parse_float(row, "perturb_trans_m", 0.10)
        T_delta = make_random_small_transform(rot_deg, trans_m, rng)
        return T_delta @ T_label

    if mode == "file":
        T0_path_str = row.get("T0_path", "").strip()
        if not T0_path_str:
            raise ValueError("T0_mode=file requires T0_path")
        return read_transform(resolve_path(T0_path_str))

    raise ValueError("T0_mode must be one of: identity, label, perturb_label, file")


def extract_one_sample(row, combined_writer):
    start_time = time.time()

    sample_id = row["sample_id"].strip()
    scenario = row["scenario"].strip()

    reference_path = resolve_path(row["reference_path"])
    observation_path = resolve_path(row["observation_path"])

    voxel_size = parse_float(row, "voxel_size", 0.05)
    normal_radius = parse_float(row, "normal_radius", 0.15)
    fpfh_radius = parse_float(row, "fpfh_radius", 0.25)
    density_radius = parse_float(row, "density_radius", 0.15)
    max_corr_distance = parse_float(row, "max_corr_distance_m", 0.40)
    label_threshold = parse_float(row, "label_threshold_m", 0.05)
    max_source_points = parse_int(row, "max_source_points", 0)

    rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))

    print("\n" + "=" * 80)
    print(f"Sample: {sample_id}")
    print(f"Scenario: {scenario}")
    print(f"Reference: {reference_path}")
    print(f"Observation: {observation_path}")

    T_label = get_label_transform(row)
    T0 = get_initial_transform(row, T_label, rng)

    reference = load_and_downsample(reference_path, voxel_size)
    observation = load_and_downsample(observation_path, voxel_size)

    print(f"Reference points after voxel: {len(reference.points)}")
    print(f"Observation points after voxel: {len(observation.points)}")

    print("Estimating normals...")
    estimate_normals(reference, normal_radius)
    estimate_normals(observation, normal_radius)

    print("Computing FPFH descriptors...")
    reference_fpfh = compute_fpfh(reference, fpfh_radius)
    observation_fpfh = compute_fpfh(observation, fpfh_radius)

    reference_points = np.asarray(reference.points)
    observation_points = np.asarray(observation.points)

    reference_normals = np.asarray(reference.normals)
    observation_normals = np.asarray(observation.normals)

    observation_points_T0 = transform_points(observation_points, T0)
    observation_points_label = transform_points(observation_points, T_label)

    R0 = T0[:3, :3]
    observation_normals_T0 = (R0 @ observation_normals.T).T

    print("Computing density counts...")
    source_density = compute_density_counts(observation_points, density_radius)
    target_density = compute_density_counts(reference_points, density_radius)

    median_source_density = np.median(source_density[source_density > 0])
    median_target_density = np.median(target_density[target_density > 0])

    if not np.isfinite(median_source_density) or median_source_density <= 0:
        median_source_density = 1.0
    if not np.isfinite(median_target_density) or median_target_density <= 0:
        median_target_density = 1.0

    source_density_norm = source_density / median_source_density
    target_density_norm = target_density / median_target_density

    eps = 1e-6

    print("Building KD-trees...")
    reference_tree = o3d.geometry.KDTreeFlann(reference)
    reverse_nn = build_reverse_nearest_indices(reference_points, observation_points_T0)

    source_indices = np.arange(len(observation_points), dtype=int)

    if max_source_points > 0 and max_source_points < len(source_indices):
        source_indices = rng.choice(source_indices, size=max_source_points, replace=False)
        source_indices = np.sort(source_indices)

    per_sample_path = OUT_DIR / f"{sample_id}_correspondences.csv"

    num_candidates = 0
    num_positive = 0
    num_negative = 0

    with open(per_sample_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for i in source_indices:
            q_T0 = observation_points_T0[i]

            k, idx, dist2 = reference_tree.search_knn_vector_3d(q_T0, 1)
            if k == 0:
                continue

            j = int(idx[0])
            distance_T0 = float(math.sqrt(dist2[0]))

            if distance_T0 > max_corr_distance:
                continue

            p = reference_points[j]
            n_p = reference_normals[j]
            n_q_T0 = observation_normals_T0[i]

            diff_T0 = q_T0 - p

            point_to_plane_residual = float(abs(np.dot(n_p, diff_T0)))

            normal_dot = float(abs(np.dot(n_q_T0, n_p)))
            normal_dot = max(0.0, min(1.0, normal_dot))

            fpfh_distance = float(np.linalg.norm(observation_fpfh[i] - reference_fpfh[j]))

            log_normalized_density_ratio = float(
                math.log((source_density_norm[i] + eps) / (target_density_norm[j] + eps))
            )

            is_mutual_nn = 1 if reverse_nn[j] == i else 0

            label_distance = float(np.linalg.norm(observation_points_label[i] - p))
            target_weight = 1 if label_distance < label_threshold else 0

            row_out = {
                "sample_id": sample_id,
                "scenario": scenario,
                "source_index": i,
                "target_index": j,
                "distance_T0": distance_T0,
                "point_to_plane_residual": point_to_plane_residual,
                "normal_dot_abs": normal_dot,
                "fpfh_distance": fpfh_distance,
                "log_normalized_density_ratio": log_normalized_density_ratio,
                "is_mutual_nn": is_mutual_nn,
                "label_distance": label_distance,
                "target_weight": target_weight,
            }

            writer.writerow(row_out)
            combined_writer.writerow(row_out)

            num_candidates += 1
            if target_weight == 1:
                num_positive += 1
            else:
                num_negative += 1

    elapsed = time.time() - start_time

    positive_rate = num_positive / max(num_candidates, 1)

    print(f"Saved: {per_sample_path}")
    print(f"Candidates kept: {num_candidates}")
    print(f"Positive reliable correspondences: {num_positive}")
    print(f"Negative unreliable correspondences: {num_negative}")
    print(f"Positive rate: {positive_rate:.3f}")
    print(f"Extraction time: {elapsed:.2f} s")

    return {
        "sample_id": sample_id,
        "scenario": scenario,
        "num_candidates": num_candidates,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "positive_rate": positive_rate,
        "time_s": elapsed,
    }


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    combined_path = OUT_DIR / "all_correspondences.csv"
    summary_path = OUT_DIR / "dataset_summary.csv"
    feature_path = OUT_DIR / "feature_columns.txt"

    with open(feature_path, "w") as f:
        for col in FEATURE_COLUMNS:
            f.write(col + "\n")

    summaries = []

    with open(CONFIG_PATH, newline="") as config_file, open(combined_path, "w", newline="") as combined_file:
        reader = csv.DictReader(config_file)
        combined_writer = csv.DictWriter(combined_file, fieldnames=OUTPUT_COLUMNS)
        combined_writer.writeheader()

        for row in reader:
            if not row.get("sample_id", "").strip():
                continue
            summary = extract_one_sample(row, combined_writer)
            summaries.append(summary)

    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "sample_id",
            "scenario",
            "num_candidates",
            "num_positive",
            "num_negative",
            "positive_rate",
            "time_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print("\n" + "=" * 80)
    print("Dataset extraction finished.")
    print(f"Combined dataset: {combined_path}")
    print(f"Summary: {summary_path}")
    print(f"Feature columns: {feature_path}")


if __name__ == "__main__":
    main()

