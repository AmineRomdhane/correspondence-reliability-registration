#!/usr/bin/env python3
"""
Train Random Forest baselines for correspondence reliability prediction.

Input:
- results/learning_data/all_correspondences.csv

Outputs:
- results/tables/random_forest_results_by_shape.csv
- results/tables/random_forest_results_average.csv
- results/tables/random_forest_feature_importance_full.csv
- results/tables/random_forest_feature_importance_reduced.csv
- results/tables/random_forest_test_metrics_by_scenario_full.csv
- results/tables/random_forest_test_metrics_by_scenario_reduced.csv
- results/figures/random_forest_confusion_matrix_<feature_set>_<test_shape>.png

Evaluation:
Leave-one-shape-out:
- Train on 3 shapes
- Test on 1 unseen shape
- Repeat for all shapes
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = BASE_DIR / "results" / "learning_data" / "all_correspondences.csv"

TABLE_DIR = BASE_DIR / "results" / "tables"
FIGURE_DIR = BASE_DIR / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "target_weight"

FULL_FEATURES = [
    "distance_T0",
    "point_to_plane_residual",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]

REDUCED_FEATURES = [
    "distance_T0",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]

KNOWN_SHAPES = [
    "box_sphere",
    "l_shape",
    "boxes_cylinder",
    "wall_column",
]

RANDOM_STATE = 42


def infer_shape_name(sample_id: str) -> str:
    for shape in KNOWN_SHAPES:
        if sample_id.startswith(shape + "_"):
            return shape

    raise ValueError(f"Could not infer shape name from sample_id: {sample_id}")


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_auc(y_true, y_proba),
    }


def plot_confusion_matrix(cm, title, output_path):
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Bad", "Reliable"])
    ax.set_yticklabels(["Bad", "Reliable"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_random_forest():
    """
    Conservative Random Forest baseline.

    n_estimators:
        Number of trees.

    max_depth:
        Limits tree complexity. Prevents each tree from memorizing the dataset.

    min_samples_leaf:
        A leaf must contain at least this many samples.
        This gives smoother decisions and reduces overfitting.

    class_weight="balanced":
        Handles the fact that reliable/bad correspondences are not perfectly balanced.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def evaluate_one_split(df, feature_set_name, features, test_shape):
    train_df = df[df["shape_name"] != test_shape].copy()
    test_df = df[df["shape_name"] == test_shape].copy()

    X_train = train_df[features].values
    y_train = train_df[TARGET].astype(int).values

    X_test = test_df[features].values
    y_test = test_df[TARGET].astype(int).values

    clf = make_random_forest()
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    cm_path = FIGURE_DIR / f"random_forest_confusion_matrix_{feature_set_name}_{test_shape}.png"
    plot_confusion_matrix(
        cm,
        f"Random Forest confusion matrix ({feature_set_name}, test={test_shape})",
        cm_path,
    )

    # Metrics by scenario for this test shape
    scenario_rows = []

    test_df = test_df.copy()
    test_df["pred"] = y_test_pred
    test_df["proba_reliable"] = y_test_proba

    for scenario, group in test_df.groupby("scenario"):
        y_s = group[TARGET].astype(int).values
        pred_s = group["pred"].astype(int).values
        proba_s = group["proba_reliable"].values

        metrics_s = compute_metrics(y_s, pred_s, proba_s)

        row = {
            "feature_set": feature_set_name,
            "test_shape": test_shape,
            "scenario": scenario,
            "num_rows": len(group),
            "positive_rate": float(np.mean(y_s)),
        }
        row.update(metrics_s)
        scenario_rows.append(row)

    # Feature importance for this split
    importance_rows = []
    for feature, importance in zip(features, clf.feature_importances_):
        importance_rows.append({
            "feature_set": feature_set_name,
            "test_shape": test_shape,
            "feature": feature,
            "importance": importance,
        })

    result_row = {
        "model": "RandomForest",
        "feature_set": feature_set_name,
        "test_shape": test_shape,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
    }

    for k, v in train_metrics.items():
        result_row[f"train_{k}"] = v

    for k, v in test_metrics.items():
        result_row[f"test_{k}"] = v

    print("\n" + "-" * 80)
    print(f"Feature set: {feature_set_name} | Test shape: {test_shape}")
    print("-" * 80)
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Train positive rate: {np.mean(y_train):.4f}")
    print(f"Test positive rate:  {np.mean(y_test):.4f}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nSaved confusion matrix: {cm_path}")

    return result_row, scenario_rows, importance_rows


def summarize_results(results_df):
    metric_cols = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    ]

    summary_rows = []

    for feature_set, group in results_df.groupby("feature_set"):
        row = {
            "model": "RandomForest",
            "feature_set": feature_set,
            "num_folds": len(group),
        }

        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def summarize_feature_importance(importance_df):
    rows = []

    for (feature_set, feature), group in importance_df.groupby(["feature_set", "feature"]):
        rows.append({
            "feature_set": feature_set,
            "feature": feature,
            "importance_mean": group["importance"].mean(),
            "importance_std": group["importance"].std(),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["feature_set", "importance_mean"],
        ascending=[True, False],
    )

    return out


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["sample_id", "scenario", TARGET] + FULL_FEATURES
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)
    df[TARGET] = df[TARGET].astype(int)

    df["shape_name"] = df["sample_id"].apply(infer_shape_name)

    print(f"Loaded dataset: {INPUT_CSV}")
    print(f"Rows: {len(df)}")

    print("\nRows by shape:")
    print(df["shape_name"].value_counts().to_string())

    print("\nPositive rate by shape:")
    print(df.groupby("shape_name")[TARGET].mean().to_string())

    feature_sets = {
        "full": FULL_FEATURES,
        "reduced": REDUCED_FEATURES,
    }

    all_result_rows = []
    all_scenario_rows = []
    all_importance_rows = []

    for feature_set_name, features in feature_sets.items():
        for test_shape in KNOWN_SHAPES:
            result_row, scenario_rows, importance_rows = evaluate_one_split(
                df=df,
                feature_set_name=feature_set_name,
                features=features,
                test_shape=test_shape,
            )

            all_result_rows.append(result_row)
            all_scenario_rows.extend(scenario_rows)
            all_importance_rows.extend(importance_rows)

    results_df = pd.DataFrame(all_result_rows)
    scenario_df = pd.DataFrame(all_scenario_rows)
    importance_df = pd.DataFrame(all_importance_rows)

    results_path = TABLE_DIR / "random_forest_results_by_shape.csv"
    results_df.to_csv(results_path, index=False)

    average_df = summarize_results(results_df)
    average_path = TABLE_DIR / "random_forest_results_average.csv"
    average_df.to_csv(average_path, index=False)

    importance_summary = summarize_feature_importance(importance_df)

    for feature_set_name in feature_sets.keys():
        subset = importance_summary[importance_summary["feature_set"] == feature_set_name]
        path = TABLE_DIR / f"random_forest_feature_importance_{feature_set_name}.csv"
        subset.to_csv(path, index=False)

    for feature_set_name in feature_sets.keys():
        subset = scenario_df[scenario_df["feature_set"] == feature_set_name]
        path = TABLE_DIR / f"random_forest_test_metrics_by_scenario_{feature_set_name}.csv"
        subset.to_csv(path, index=False)

    print("\n" + "=" * 80)
    print("Random Forest results by shape:")
    print(results_df.to_string(index=False))

    print("\nAverage results:")
    print(average_df.to_string(index=False))

    print("\nAverage feature importance:")
    print(importance_summary.to_string(index=False))

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {average_path}")
    print("  results/tables/random_forest_feature_importance_full.csv")
    print("  results/tables/random_forest_feature_importance_reduced.csv")
    print("  results/tables/random_forest_test_metrics_by_scenario_full.csv")
    print("  results/tables/random_forest_test_metrics_by_scenario_reduced.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
