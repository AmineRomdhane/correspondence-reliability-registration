#!/usr/bin/env python3
"""
Create multi-shape synthetic cases for correspondence-weight learning.

This script generates several different reference geometries, then creates
different observation scenarios from each geometry.

Generated base shapes:
1. box_sphere
2. l_shape
3. boxes_cylinder
4. wall_column

Generated scenarios for each shape:
1. static_noise
2. partial_overlap
3. added_outliers
4. strong_noise
5. scene_change

For each case, the script saves:
- reference.ply
- observation.ply
- T_true.txt

Convention:
T_true maps reference -> observation.

The extraction script must use:
label_transform_direction = ref_to_obs

The script also generates:
configs/dataset_samples.csv
"""

from pathlib import Path
import csv
import numpy as np
import open3d as o3d


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_BASE = BASE_DIR / "data" / "learning_synthetic_cases_multishape"
CONFIG_PATH = BASE_DIR / "configs" / "dataset_samples.csv"

OUT_BASE.mkdir(parents=True, exist_ok=True)
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def create_box(width, height, depth, center):
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=width,
        height=height,
        depth=depth,
    )
    mesh.translate([
        center[0] - width / 2.0,
        center[1] - height / 2.0,
        center[2] - depth / 2.0,
    ])
    return mesh


def create_sphere(radius, center):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=32)
    mesh.translate(center)
    return mesh


def create_cylinder(radius, depth, center):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=depth,
        resolution=48,
    )
    mesh.translate(center)
    return mesh


def mesh_to_cloud(mesh, n_points=6000):
    mesh.compute_vertex_normals()
    cloud = mesh.sample_points_uniformly(number_of_points=n_points)

    pts = np.asarray(cloud.points)
    centroid = pts.mean(axis=0)
    pts = pts - centroid

    centered = o3d.geometry.PointCloud()
    centered.points = o3d.utility.Vector3dVector(pts)
    return centered


# ---------------------------------------------------------------------
# Base shapes
# ---------------------------------------------------------------------

def make_box_sphere():
    box = create_box(2.0, 1.0, 1.0, center=[0.0, 0.0, 0.0])
    sphere = create_sphere(0.4, center=[0.8, 0.0, 0.35])
    mesh = box + sphere
    return mesh_to_cloud(mesh, n_points=6000)


def make_l_shape():
    box1 = create_box(2.0, 0.45, 0.7, center=[0.0, 0.0, 0.0])
    box2 = create_box(0.45, 1.6, 0.7, center=[-0.75, 0.55, 0.0])
    small_box = create_box(0.45, 0.45, 0.45, center=[0.75, 0.55, 0.35])
    mesh = box1 + box2 + small_box
    return mesh_to_cloud(mesh, n_points=6000)


def make_boxes_cylinder():
    base = create_box(1.8, 0.8, 0.4, center=[0.0, 0.0, -0.2])
    vertical_box = create_box(0.5, 0.5, 1.3, center=[-0.55, 0.0, 0.35])
    cylinder = create_cylinder(0.28, 1.2, center=[0.55, 0.0, 0.25])
    sphere = create_sphere(0.25, center=[0.55, 0.0, 0.95])
    mesh = base + vertical_box + cylinder + sphere
    return mesh_to_cloud(mesh, n_points=6000)


def make_wall_column():
    floor = create_box(2.4, 1.8, 0.12, center=[0.0, 0.0, -0.55])
    wall = create_box(2.4, 0.12, 1.4, center=[0.0, 0.85, 0.1])
    column = create_cylinder(0.22, 1.4, center=[-0.65, 0.25, 0.1])
    block = create_box(0.55, 0.45, 0.6, center=[0.65, -0.25, -0.25])
    mesh = floor + wall + column + block
    return mesh_to_cloud(mesh, n_points=6000)


# ---------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------

def rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)],
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1],
    ])

    return Rz @ Ry @ Rx


def make_transform(shape_index):
    """
    Returns T_true mapping reference -> observation.
    Each shape gets a slightly different known transform.
    """
    T = np.eye(4)

    rz = 12.0 + 3.0 * shape_index
    ry = 2.0 * shape_index
    rx = -1.5 * shape_index

    T[:3, :3] = rotation_matrix_xyz(rx, ry, rz)
    T[:3, 3] = np.array([
        0.45 + 0.08 * shape_index,
        -0.20 + 0.05 * shape_index,
        0.08,
    ])

    return T


def transform_cloud(cloud, T):
    out = o3d.geometry.PointCloud(cloud)
    out.transform(T)
    return out


# ---------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------

def add_gaussian_noise(cloud, sigma, rng):
    pts = np.asarray(cloud.points)
    noisy = pts + rng.normal(0.0, sigma, size=pts.shape)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(noisy)
    return out


def partial_overlap(cloud, keep_ratio=0.65):
    pts = np.asarray(cloud.points)

    # Keep only a spatial part of the cloud.
    x_threshold = np.quantile(pts[:, 0], keep_ratio)
    kept = pts[pts[:, 0] <= x_threshold]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(kept)
    return out


def added_outliers(cloud, outlier_ratio, rng):
    pts = np.asarray(cloud.points)

    n_outliers = int(len(pts) * outlier_ratio)

    min_bound = pts.min(axis=0)
    max_bound = pts.max(axis=0)

    margin = 0.6
    low = min_bound - margin
    high = max_bound + margin

    outliers = rng.uniform(low=low, high=high, size=(n_outliers, 3))
    combined = np.vstack([pts, outliers])

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(combined)
    return out


def remove_region(cloud, remove_ratio=0.25):
    pts = np.asarray(cloud.points)

    # Remove one spatial region to simulate missing structure.
    y_threshold = np.quantile(pts[:, 1], 1.0 - remove_ratio)
    kept = pts[pts[:, 1] <= y_threshold]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(kept)
    return out


def add_changed_object(cloud, rng):
    pts = np.asarray(cloud.points)

    min_bound = pts.min(axis=0)
    max_bound = pts.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    extent = max_bound - min_bound

    extra = create_box(
        0.45,
        0.45,
        0.55,
        center=[
            center[0] + 0.35 * extent[0],
            center[1] - 0.30 * extent[1],
            center[2],
        ],
    )

    extra_cloud = extra.sample_points_uniformly(number_of_points=800)
    extra_pts = np.asarray(extra_cloud.points)
    extra_pts = extra_pts + rng.normal(0.0, 0.005, size=extra_pts.shape)

    combined = np.vstack([pts, extra_pts])

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(combined)
    return out


def make_observation_for_scenario(base_observation, scenario, rng):
    if scenario == "static_noise":
        return add_gaussian_noise(base_observation, sigma=0.01, rng=rng)

    if scenario == "partial_overlap":
        obs = add_gaussian_noise(base_observation, sigma=0.01, rng=rng)
        return partial_overlap(obs, keep_ratio=0.65)

    if scenario == "added_outliers":
        obs = add_gaussian_noise(base_observation, sigma=0.01, rng=rng)
        return added_outliers(obs, outlier_ratio=0.20, rng=rng)

    if scenario == "strong_noise":
        return add_gaussian_noise(base_observation, sigma=0.03, rng=rng)

    if scenario == "scene_change":
        obs = add_gaussian_noise(base_observation, sigma=0.01, rng=rng)
        obs = remove_region(obs, remove_ratio=0.25)
        obs = add_changed_object(obs, rng=rng)
        return obs

    raise ValueError(f"Unknown scenario: {scenario}")


# ---------------------------------------------------------------------
# Saving and config generation
# ---------------------------------------------------------------------

def save_case(shape_name, scenario, reference, observation, T_true):
    case_dir = OUT_BASE / shape_name / scenario
    case_dir.mkdir(parents=True, exist_ok=True)

    ref_path = case_dir / "reference.ply"
    obs_path = case_dir / "observation.ply"
    T_path = case_dir / "T_true.txt"

    o3d.io.write_point_cloud(str(ref_path), reference)
    o3d.io.write_point_cloud(str(obs_path), observation)
    np.savetxt(T_path, T_true)

    print(f"Saved {shape_name}/{scenario}")
    print(f"  reference points:   {len(reference.points)}")
    print(f"  observation points: {len(observation.points)}")

    return ref_path, obs_path, T_path


def rel(path):
    return str(path.relative_to(BASE_DIR))


def generate_config(saved_cases):
    """
    Write configs/dataset_samples.csv.

    Three T0 difficulty levels are used for every generated case.
    Feature extraction parameters are kept fixed for consistency.
    """

    fieldnames = [
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

    difficulties = [
        {
            "name": "medium_2",
            "perturb_rot_deg": 5.5,
            "perturb_trans_m": 0.09,
            "max_corr_distance_m": 0.45,
        },
        {
            "name": "medium_3",
            "perturb_rot_deg": 6.0,
            "perturb_trans_m": 0.10,
            "max_corr_distance_m": 0.50,
        },
        {
            "name": "hard_mixed",
            "perturb_rot_deg": 6.5,
            "perturb_trans_m": 0.11,
            "max_corr_distance_m": 0.55,
        },
    ]

    rows = []

    for case in saved_cases:
        shape_name = case["shape_name"]
        scenario = case["scenario"]

        for diff in difficulties:
            sample_id = f"{shape_name}_{scenario}_{diff['name']}"

            rows.append({
                "sample_id": sample_id,
                "scenario": scenario,
                "reference_path": rel(case["reference_path"]),
                "observation_path": rel(case["observation_path"]),
                "label_transform_path": rel(case["T_path"]),
                "label_transform_direction": "ref_to_obs",
                "T0_mode": "perturb_label",
                "T0_path": "",
                "perturb_rot_deg": diff["perturb_rot_deg"],
                "perturb_trans_m": diff["perturb_trans_m"],
                "voxel_size": 0.05,
                "normal_radius": 0.15,
                "fpfh_radius": 0.25,
                "density_radius": 0.15,
                "max_corr_distance_m": diff["max_corr_distance_m"],
                "label_threshold_m": 0.08,
                "max_source_points": 1500,
            })

    with open(CONFIG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nGenerated config: {CONFIG_PATH}")
    print(f"Number of dataset samples: {len(rows)}")


def main():
    rng = np.random.default_rng(42)

    shape_builders = {
        "box_sphere": make_box_sphere,
        "l_shape": make_l_shape,
        "boxes_cylinder": make_boxes_cylinder,
        "wall_column": make_wall_column,
    }

    scenarios = [
        "static_noise",
        "partial_overlap",
        "added_outliers",
        "strong_noise",
        "scene_change",
    ]

    saved_cases = []

    for shape_index, (shape_name, builder) in enumerate(shape_builders.items()):
        print("\n" + "=" * 80)
        print(f"Generating shape: {shape_name}")

        reference = builder()
        T_true = make_transform(shape_index)
        base_observation = transform_cloud(reference, T_true)

        for scenario in scenarios:
            observation = make_observation_for_scenario(base_observation, scenario, rng)

            ref_path, obs_path, T_path = save_case(
                shape_name=shape_name,
                scenario=scenario,
                reference=reference,
                observation=observation,
                T_true=T_true,
            )

            saved_cases.append({
                "shape_name": shape_name,
                "scenario": scenario,
                "reference_path": ref_path,
                "observation_path": obs_path,
                "T_path": T_path,
            })

    generate_config(saved_cases)

    print("\nDone.")
    print("Next run:")
    print("  rm -f results/learning_data/*.csv")
    print("  python src/extract_correspondence_dataset.py")


if __name__ == "__main__":
    main()
