#!/usr/bin/env python3
"""
Train a PyTorch MLP for correspondence reliability prediction.

Input:
- results/learning_data/all_correspondences.csv

Outputs:
- results/tables/mlp_results_by_shape.csv
- results/tables/mlp_results_average.csv
- results/tables/mlp_training_history_<feature_set>_<test_shape>.csv
- results/figures/mlp_confusion_matrix_<feature_set>_<test_shape>.png
- results/figures/mlp_loss_curve_<feature_set>_<test_shape>.png

Evaluation:
Leave-one-shape-out:
- Train on 3 shapes
- Test on 1 unseen shape
- Repeat for all 4 shapes

Feature sets:
- full
- reduced
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


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

EPOCHS = 120
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class CorrespondenceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_loader(X, y, batch_size, shuffle):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if train:
                loss.backward()
                optimizer.step()

        batch_count = len(y_batch)
        total_loss += loss.item() * batch_count
        total_count += batch_count

    return total_loss / max(total_count, 1)


def predict_proba(model, X, device, batch_size=4096):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    all_probs = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


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


def plot_loss_curve(history_df, title, output_path):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def split_train_val_by_sample_id(train_df, val_fraction=0.20):
    """
    Validation split is done by sample_id, not by random rows.
    This avoids having the exact same sample_id in train and validation.
    """
    sample_ids = sorted(train_df["sample_id"].unique())

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(sample_ids)

    n_val = max(1, int(round(len(sample_ids) * val_fraction)))

    val_ids = set(sample_ids[:n_val])
    train_ids = set(sample_ids[n_val:])

    inner_train_df = train_df[train_df["sample_id"].isin(train_ids)].copy()
    val_df = train_df[train_df["sample_id"].isin(val_ids)].copy()

    return inner_train_df, val_df


def train_and_evaluate_one_split(df, feature_set_name, features, test_shape, device):
    print("\n" + "=" * 80)
    print(f"MLP | feature_set={feature_set_name} | test_shape={test_shape}")
    print("=" * 80)

    trainval_df = df[df["shape_name"] != test_shape].copy()
    test_df = df[df["shape_name"] == test_shape].copy()

    inner_train_df, val_df = split_train_val_by_sample_id(trainval_df)

    print(f"Train rows: {len(inner_train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")

    print(f"Train shapes: {sorted(inner_train_df['shape_name'].unique())}")
    print(f"Val sample_ids: {len(val_df['sample_id'].unique())}")
    print(f"Test shape: {test_shape}")

    X_train_raw = inner_train_df[features].values.astype(np.float32)
    y_train = inner_train_df[TARGET].astype(int).values.astype(np.float32)

    X_val_raw = val_df[features].values.astype(np.float32)
    y_val = val_df[TARGET].astype(int).values.astype(np.float32)

    X_test_raw = test_df[features].values.astype(np.float32)
    y_test = test_df[TARGET].astype(int).values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))

    pos_weight_value = n_neg / max(n_pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    print(f"Train positive rate: {np.mean(y_train):.4f}")
    print(f"Val positive rate:   {np.mean(y_val):.4f}")
    print(f"Test positive rate:  {np.mean(y_test):.4f}")
    print(f"pos_weight:          {pos_weight_value:.4f}")

    model = CorrespondenceMLP(input_dim=len(features)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = np.inf
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"best_epoch={best_epoch}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history_rows)

    history_path = TABLE_DIR / f"mlp_training_history_{feature_set_name}_{test_shape}.csv"
    history_df.to_csv(history_path, index=False)

    loss_curve_path = FIGURE_DIR / f"mlp_loss_curve_{feature_set_name}_{test_shape}.png"
    plot_loss_curve(
        history_df,
        f"MLP loss curve ({feature_set_name}, test={test_shape})",
        loss_curve_path,
    )

    y_trainval = trainval_df[TARGET].astype(int).values.astype(np.float32)
    X_trainval_raw = trainval_df[features].values.astype(np.float32)
    X_trainval = scaler.transform(X_trainval_raw).astype(np.float32)

    y_trainval_proba = predict_proba(model, X_trainval, device)
    y_test_proba = predict_proba(model, X_test, device)

    y_trainval_pred = (y_trainval_proba >= 0.5).astype(int)
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    trainval_metrics = compute_metrics(
        y_trainval.astype(int),
        y_trainval_pred,
        y_trainval_proba,
    )

    test_metrics = compute_metrics(
        y_test.astype(int),
        y_test_pred,
        y_test_proba,
    )

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    cm = confusion_matrix(y_test.astype(int), y_test_pred, labels=[0, 1])

    cm_path = FIGURE_DIR / f"mlp_confusion_matrix_{feature_set_name}_{test_shape}.png"
    plot_confusion_matrix(
        cm,
        f"MLP confusion matrix ({feature_set_name}, test={test_shape})",
        cm_path,
    )

    print(f"Saved history:          {history_path}")
    print(f"Saved loss curve:       {loss_curve_path}")
    print(f"Saved confusion matrix: {cm_path}")

    result_row = {
        "model": "MLP",
        "feature_set": feature_set_name,
        "test_shape": test_shape,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_rows": len(inner_train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_positive_rate": float(np.mean(y_train)),
        "val_positive_rate": float(np.mean(y_val)),
        "test_positive_rate": float(np.mean(y_test)),
    }

    for k, v in trainval_metrics.items():
        result_row[f"trainval_{k}"] = v

    for k, v in test_metrics.items():
        result_row[f"test_{k}"] = v

    return result_row


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
            "model": "MLP",
            "feature_set": feature_set,
            "num_folds": len(group),
        }

        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main():
    set_seed(RANDOM_STATE)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

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

    result_rows = []

    for feature_set_name, features in feature_sets.items():
        for test_shape in KNOWN_SHAPES:
            row = train_and_evaluate_one_split(
                df=df,
                feature_set_name=feature_set_name,
                features=features,
                test_shape=test_shape,
                device=device,
            )
            result_rows.append(row)

    results_df = pd.DataFrame(result_rows)

    results_path = TABLE_DIR / "mlp_results_by_shape.csv"
    results_df.to_csv(results_path, index=False)

    average_df = summarize_results(results_df)

    average_path = TABLE_DIR / "mlp_results_average.csv"
    average_df.to_csv(average_path, index=False)

    print("\n" + "=" * 80)
    print("MLP results by shape:")
    print(results_df.to_string(index=False))

    print("\nAverage results:")
    print(average_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {average_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
