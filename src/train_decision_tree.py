#!/usr/bin/env python3
"""
Train an interpretable Decision Tree for correspondence reliability prediction.

Input:
- results/learning_data/all_correspondences.csv

Outputs:
- results/tables/decision_tree_results.csv
- results/tables/decision_tree_feature_importance_*.csv
- results/tables/decision_tree_test_metrics_by_scenario_*.csv
- results/tables/decision_tree_rules_*.txt
- results/figures/decision_tree_*.png
- results/figures/decision_tree_confusion_matrix_*.png

Important:
The split is done by SHAPE, not by random rows.

Default:
train shapes = box_sphere, l_shape, boxes_cylinder
test shape   = wall_column
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
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

TEST_SHAPE = "wall_column"

MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 100
RANDOM_STATE = 42


def infer_shape_name(sample_id: str) -> str:
    """
    Infer the base shape from the sample_id.
    Example:
    wall_column_static_noise_medium_2 -> wall_column
    boxes_cylinder_scene_change_hard_mixed -> boxes_cylinder
    """
    for shape in KNOWN_SHAPES:
        if sample_id.startswith(shape + "_"):
            return shape

    raise ValueError(f"Could not infer shape name from sample_id: {sample_id}")


def safe_auc(y_true, y_score):
    """
    ROC-AUC is undefined if y_true contains only one class.
    """
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


def train_and_evaluate(df, feature_set_name, features):
    print("\n" + "=" * 80)
    print(f"Training Decision Tree: {feature_set_name}")
    print("=" * 80)

    train_df = df[df["shape_name"] != TEST_SHAPE].copy()
    test_df = df[df["shape_name"] == TEST_SHAPE].copy()

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Train shapes: {sorted(train_df['shape_name'].unique())}")
    print(f"Test shapes:  {sorted(test_df['shape_name'].unique())}")

    X_train = train_df[features].values
    y_train = train_df[TARGET].astype(int).values

    X_test = test_df[features].values
    y_test = test_df[TARGET].astype(int).values

    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    print("\nTrain metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save feature importance
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": clf.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    importance_path = TABLE_DIR / f"decision_tree_feature_importance_{feature_set_name}.csv"
    importance_df.to_csv(importance_path, index=False)

    print(f"\nSaved feature importance: {importance_path}")
    print(importance_df.to_string(index=False))

    # Save rules
    rules = export_text(
        clf,
        feature_names=features,
        decimals=4,
        spacing=3,
    )

    rules_path = TABLE_DIR / f"decision_tree_rules_{feature_set_name}.txt"
    rules_path.write_text(rules)

    print(f"\nSaved tree rules: {rules_path}")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    cm_path = FIGURE_DIR / f"decision_tree_confusion_matrix_{feature_set_name}.png"
    plot_confusion_matrix(
        cm,
        f"Decision Tree confusion matrix ({feature_set_name})",
        cm_path,
    )

    print(f"Saved confusion matrix: {cm_path}")

    # Save tree figure
    tree_fig_path = FIGURE_DIR / f"decision_tree_{feature_set_name}.png"

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        clf,
        feature_names=features,
        class_names=["bad", "reliable"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(tree_fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved tree figure: {tree_fig_path}")

    # Metrics by scenario on test shape
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
            "test_shape": TEST_SHAPE,
            "scenario": scenario,
            "num_rows": len(group),
            "positive_rate": float(np.mean(y_s)),
        }
        row.update(metrics_s)
        scenario_rows.append(row)

    scenario_metrics_df = pd.DataFrame(scenario_rows)
    scenario_metrics_path = TABLE_DIR / f"decision_tree_test_metrics_by_scenario_{feature_set_name}.csv"
    scenario_metrics_df.to_csv(scenario_metrics_path, index=False)

    print(f"Saved test metrics by scenario: {scenario_metrics_path}")

    result_row = {
        "model": "DecisionTree",
        "feature_set": feature_set_name,
        "test_shape": TEST_SHAPE,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
    }

    for k, v in train_metrics.items():
        result_row[f"train_{k}"] = v

    for k, v in test_metrics.items():
        result_row[f"test_{k}"] = v

    return result_row


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["sample_id", "scenario", TARGET] + FULL_FEATURES
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # Remove invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)

    # Make sure target is binary integer
    df[TARGET] = df[TARGET].astype(int)

    # Infer shape name
    df["shape_name"] = df["sample_id"].apply(infer_shape_name)

    print(f"Loaded dataset: {INPUT_CSV}")
    print(f"Rows: {len(df)}")
    print("\nRows by shape:")
    print(df["shape_name"].value_counts().to_string())

    print("\nPositive rate by shape:")
    print(df.groupby("shape_name")[TARGET].mean().to_string())

    results = []

    results.append(
        train_and_evaluate(
            df=df,
            feature_set_name="full",
            features=FULL_FEATURES,
        )
    )

    results.append(
        train_and_evaluate(
            df=df,
            feature_set_name="reduced",
            features=REDUCED_FEATURES,
        )
    )

    results_df = pd.DataFrame(results)

    results_path = TABLE_DIR / "decision_tree_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 80)
    print("Final comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSaved results: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
