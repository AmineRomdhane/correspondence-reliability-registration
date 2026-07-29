#!/usr/bin/env python3
"""
Evaluate the reduced MLP on real data using leave-one-real-case-out testing.

Dataset:
- results/learning_data_synthetic_plus_real_curated_clean_v3/all_correspondences.csv

Protocol:
- Test set = one complete real registration case
- Training set = all synthetic samples + all other real registration cases
- Validation set = group split from training data
- Real easy/medium/hard variants of the same registration stay together

Outputs:
- results/tables/mlp_real_holdout_clean_v3_results_by_case.csv
- results/tables/mlp_real_holdout_clean_v3_results_average.csv
- results/tables/mlp_real_holdout_clean_v3_threshold_results_by_case.csv
- results/tables/mlp_real_holdout_clean_v3_threshold_results_average.csv
"""

from pathlib import Path
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from train_mlp_pytorch import (
    BASE_DIR,
    TABLE_DIR,
    TARGET,
    REDUCED_FEATURES,
    RANDOM_STATE,
    EPOCHS,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    PATIENCE,
    set_seed,
    CorrespondenceMLP,
    make_loader,
    run_epoch,
    predict_proba,
)


INPUT_CSV = (
    BASE_DIR
    / "results"
    / "learning_data_synthetic_plus_real_curated_clean_v3"
    / "all_correspondences.csv"
)

MODEL_NAME = "MLP_real_holdout_clean_v3"
FEATURE_SET_NAME = "reduced"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAIN_THRESHOLD = 0.5
VAL_FRACTION = 0.20


def get_real_case_id(sample_id):
    """
    Convert:
    real_20260703_131528_xxx_easy
    into:
    20260703_131528_xxx
    """
    sample_id = str(sample_id)

    if not sample_id.startswith("real_"):
        return ""

    case_id = sample_id[len("real_"):]

    for suffix in ["_easy", "_medium", "_hard"]:
        if case_id.endswith(suffix):
            case_id = case_id[: -len(suffix)]
            break

    return case_id


def get_group_id(sample_id):
    """
    Used for validation splitting.

    Synthetic samples are grouped by sample_id.
    Real samples are grouped by real_case_id so easy/medium/hard variants
    of the same real case do not get split apart.
    """
    sample_id = str(sample_id)

    if sample_id.startswith("real_"):
        return "real_case__" + get_real_case_id(sample_id)

    return "synthetic_sample__" + sample_id


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_auc(y_true, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "num_predicted_reliable": int(np.sum(y_pred == 1)),
        "acceptance_rate": float(np.mean(y_pred == 1)),
    }


def split_train_val_by_group(trainval_df, random_state):
    groups = np.array(sorted(trainval_df["group_id"].unique()))

    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)

    n_val = max(1, int(round(len(groups) * VAL_FRACTION)))

    val_groups = set(groups[:n_val])
    train_groups = set(groups[n_val:])

    train_df = trainval_df[trainval_df["group_id"].isin(train_groups)].copy()
    val_df = trainval_df[trainval_df["group_id"].isin(val_groups)].copy()

    if train_df.empty or val_df.empty:
        raise RuntimeError("Empty train or validation split.")

    return train_df, val_df


def train_one_real_holdout(df, heldout_case_id, device, fold_index):
    test_mask = (df["is_real"]) & (df["real_case_id"] == heldout_case_id)

    test_df = df[test_mask].copy()
    trainval_df = df[~test_mask].copy()

    if test_df.empty:
        raise RuntimeError(f"Empty test set for real case: {heldout_case_id}")

    train_df, val_df = split_train_val_by_group(
        trainval_df=trainval_df,
        random_state=RANDOM_STATE + fold_index,
    )

    X_train_raw = train_df[REDUCED_FEATURES].values.astype(np.float32)
    y_train = train_df[TARGET].astype(int).values.astype(np.float32)

    X_val_raw = val_df[REDUCED_FEATURES].values.astype(np.float32)
    y_val = val_df[TARGET].astype(int).values.astype(np.float32)

    X_test_raw = test_df[REDUCED_FEATURES].values.astype(np.float32)
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

    model = CorrespondenceMLP(input_dim=len(REDUCED_FEATURES)).to(device)

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

    print("\n" + "=" * 100)
    print(f"Held-out real case: {heldout_case_id}")
    print("=" * 100)
    print(f"Train rows:      {len(train_df)}")
    print(f"Val rows:        {len(val_df)}")
    print(f"Test rows:       {len(test_df)}")
    print(f"Train pos rate:  {float(np.mean(y_train)):.4f}")
    print(f"Val pos rate:    {float(np.mean(y_val)):.4f}")
    print(f"Test pos rate:   {float(np.mean(y_test)):.4f}")
    print(f"pos_weight:      {pos_weight_value:.4f}")

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
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

    y_test_int = y_test.astype(int)
    y_test_proba = predict_proba(model, X_test, device)

    main_metrics = compute_metrics(
        y_true=y_test_int,
        y_proba=y_test_proba,
        threshold=MAIN_THRESHOLD,
    )

    main_row = {
        "model": MODEL_NAME,
        "feature_set": FEATURE_SET_NAME,
        "heldout_real_case": heldout_case_id,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_positive_rate": float(np.mean(y_train)),
        "val_positive_rate": float(np.mean(y_val)),
        "test_positive_rate": float(np.mean(y_test)),
    }

    main_row.update(main_metrics)

    threshold_rows = []

    for threshold in THRESHOLDS:
        row = {
            "model": MODEL_NAME,
            "feature_set": FEATURE_SET_NAME,
            "heldout_real_case": heldout_case_id,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_rows": len(test_df),
            "test_positive_rate": float(np.mean(y_test)),
        }

        row.update(
            compute_metrics(
                y_true=y_test_int,
                y_proba=y_test_proba,
                threshold=threshold,
            )
        )

        threshold_rows.append(row)

    return main_row, threshold_rows


def summarize_average(df, group_cols=None):
    if group_cols is None:
        group_cols = []

    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "fp",
        "fn",
        "tp",
        "tn",
        "num_predicted_reliable",
        "acceptance_rate",
    ]

    rows = []

    if group_cols:
        iterator = df.groupby(group_cols)
    else:
        iterator = [((), df)]

    for key, group in iterator:
        row = {
            "model": MODEL_NAME,
            "feature_set": FEATURE_SET_NAME,
            "num_folds": len(group),
        }

        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            for col, value in zip(group_cols, key):
                row[col] = value

        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    set_seed(RANDOM_STATE)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["sample_id", "scenario", TARGET] + REDUCED_FEATURES
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols).copy()
    df[TARGET] = df[TARGET].astype(int)

    df["sample_id"] = df["sample_id"].astype(str)
    df["is_real"] = df["sample_id"].str.startswith("real_")
    df["real_case_id"] = df["sample_id"].apply(get_real_case_id)
    df["group_id"] = df["sample_id"].apply(get_group_id)

    real_case_ids = sorted(
        case_id for case_id in df.loc[df["is_real"], "real_case_id"].unique()
        if case_id
    )

    if not real_case_ids:
        raise RuntimeError("No real cases found in dataset.")

    print(f"Number of real cases: {len(real_case_ids)}")
    print("Real cases:")
    for case_id in real_case_ids:
        n_rows = len(df[(df["is_real"]) & (df["real_case_id"] == case_id)])
        pos_rate = df[(df["is_real"]) & (df["real_case_id"] == case_id)][TARGET].mean()
        print(f"  {case_id} | rows={n_rows} | pos_rate={pos_rate:.4f}")

    result_rows = []
    threshold_rows_all = []

    for fold_index, heldout_case_id in enumerate(real_case_ids):
        main_row, threshold_rows = train_one_real_holdout(
            df=df,
            heldout_case_id=heldout_case_id,
            device=device,
            fold_index=fold_index,
        )

        result_rows.append(main_row)
        threshold_rows_all.extend(threshold_rows)

    results_df = pd.DataFrame(result_rows)
    threshold_df = pd.DataFrame(threshold_rows_all)

    by_case_path = TABLE_DIR / "mlp_real_holdout_clean_v3_results_by_case.csv"
    avg_path = TABLE_DIR / "mlp_real_holdout_clean_v3_results_average.csv"

    threshold_by_case_path = TABLE_DIR / "mlp_real_holdout_clean_v3_threshold_results_by_case.csv"
    threshold_avg_path = TABLE_DIR / "mlp_real_holdout_clean_v3_threshold_results_average.csv"

    results_df.to_csv(by_case_path, index=False)
    threshold_df.to_csv(threshold_by_case_path, index=False)

    avg_df = summarize_average(results_df)
    avg_df.to_csv(avg_path, index=False)

    threshold_avg_df = summarize_average(
        threshold_df,
        group_cols=["threshold"],
    ).sort_values(by="threshold")

    threshold_avg_df.to_csv(threshold_avg_path, index=False)

    print("\n" + "=" * 100)
    print("Average real-holdout result at threshold 0.5:")
    print(avg_df.to_string(index=False))

    print("\nAverage threshold results:")
    print(threshold_avg_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {by_case_path}")
    print(f"  {avg_path}")
    print(f"  {threshold_by_case_path}")
    print(f"  {threshold_avg_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
