#!/usr/bin/env python3
"""
Train final MLP on synthetic + all real v3 correspondence data.

This creates a reusable model for weighted registration.

Output:
results/by_test/mlp_synthetic_plus_all_real_v3_final/
    mlp_synthetic_plus_all_real_v3_final_model.pt
    mlp_synthetic_plus_all_real_v3_final_scaler.pkl
    mlp_synthetic_plus_all_real_v3_final_training_history.csv
    mlp_synthetic_plus_all_real_v3_final_metadata.json
"""

from pathlib import Path
import copy
import json
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    BASE_DIR
    / "results"
    / "learning_data_synthetic_plus_real_curated_clean_v3"
    / "all_correspondences.csv"
)

TEST_NAME = "mlp_synthetic_plus_all_real_v3_final"
OUT_DIR = BASE_DIR / "results" / "by_test" / TEST_NAME

FEATURES = [
    "distance_T0",
    "normal_dot_abs",
    "fpfh_distance",
    "log_normalized_density_ratio",
    "is_mutual_nn",
]

TARGET = "target_weight"

RANDOM_STATE = 42
VAL_FRACTION_BY_SAMPLE_ID = 0.20

LR = 1e-3
DROPOUT = 0.10
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1024
EPOCHS = 150
PATIENCE = 15


class CorrespondenceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


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


def split_by_sample_id(df):
    sample_ids = np.array(sorted(df["sample_id"].astype(str).unique()))

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(sample_ids)

    n_val = max(1, int(round(VAL_FRACTION_BY_SAMPLE_ID * len(sample_ids))))
    val_ids = set(sample_ids[:n_val].tolist())

    val_df = df[df["sample_id"].isin(val_ids)].copy()
    train_df = df[~df["sample_id"].isin(val_ids)].copy()

    return train_df, val_df, sorted(val_ids)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    print("=" * 100)
    print(TEST_NAME)
    print("=" * 100)
    print("Input:", INPUT_CSV)
    print("Output:", OUT_DIR)
    print("Features:", FEATURES)
    print("=" * 100)

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["sample_id", TARGET] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols).copy()
    df[TARGET] = df[TARGET].astype(int)

    train_df, val_df, val_sample_ids = split_by_sample_id(df)

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES].values)

    X_train = scaler.transform(train_df[FEATURES].values)
    y_train = train_df[TARGET].values.astype(np.float32)

    X_val = scaler.transform(val_df[FEATURES].values)
    y_val = val_df[TARGET].values.astype(np.float32)

    num_pos = float(np.sum(y_train == 1))
    num_neg = float(np.sum(y_train == 0))

    pos_weight_value = num_neg / num_pos if num_pos > 0 else 1.0

    print(f"Rows total: {len(df)}")
    print(f"Rows train: {len(train_df)}")
    print(f"Rows val:   {len(val_df)}")
    print(f"Train positive rate: {np.mean(y_train):.4f}")
    print(f"Val positive rate:   {np.mean(y_val):.4f}")
    print(f"pos_weight:          {pos_weight_value:.4f}")
    print("-" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("-" * 100)

    model = CorrespondenceMLP(input_dim=len(FEATURES)).to(device)

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
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"val_auc={val_auc:.4f} | "
                f"best_epoch={best_epoch}"
            )

        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = OUT_DIR / f"{TEST_NAME}_model.pt"
    scaler_path = OUT_DIR / f"{TEST_NAME}_scaler.pkl"
    history_path = OUT_DIR / f"{TEST_NAME}_training_history.csv"
    metadata_path = OUT_DIR / f"{TEST_NAME}_metadata.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": FEATURES,
            "dropout": DROPOUT,
            "input_dim": len(FEATURES),
        },
        model_path,
    )

    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)

    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    metadata = {
        "test_name": TEST_NAME,
        "input_csv": str(INPUT_CSV),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "features": FEATURES,
        "lr": LR,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs_run": int(history_rows[-1]["epoch"]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "pos_weight": float(pos_weight_value),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_positive_rate": float(np.mean(y_train)),
        "val_positive_rate": float(np.mean(y_val)),
        "val_sample_ids": val_sample_ids,
    }

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print(model_path)
    print(scaler_path)
    print(history_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
