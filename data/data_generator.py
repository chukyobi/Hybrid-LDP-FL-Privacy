"""
data/data_generator.py
======================
Generates a synthetic digital footprint dataset, labels each feature
as HIGH or LOW sensitivity, and splits it into per-client shards
for federated learning simulation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    NUM_SAMPLES, NUM_FEATURES, NUM_CLASSES, TEST_SPLIT, RANDOM_SEED,
    HIGH_SENSITIVITY_FEATURES, LOW_SENSITIVITY_FEATURES, NUM_CLIENTS
)

np.random.seed(RANDOM_SEED)

# ── All feature names (ordered) ───────────────────────────────────────────────
ALL_FEATURES = HIGH_SENSITIVITY_FEATURES + LOW_SENSITIVITY_FEATURES


def generate_dataset(n_samples: int = NUM_SAMPLES) -> pd.DataFrame:
    """
    Generate a synthetic digital-footprint dataset.

    Each row represents one user's aggregated metadata snapshot.
    HIGH-sensitivity features have heavier inter-user variance
    (simulating informativeness) to make privacy more critical.

    Returns
    -------
    pd.DataFrame with columns = ALL_FEATURES + ['label', 'sensitivity_class']
    """
    data = {}

    # High-sensitivity features: wider range → more identifying
    for feat in HIGH_SENSITIVITY_FEATURES:
        data[feat] = np.random.normal(loc=0.5, scale=0.4, size=n_samples).clip(0, 1)

    # Low-sensitivity features: wider range to preserve utility under noise
    for feat in LOW_SENSITIVITY_FEATURES:
        data[feat] = np.random.normal(loc=0.5, scale=0.35, size=n_samples).clip(0, 1)

    df = pd.DataFrame(data)

    # Binary label: privacy-risk level (0 = low risk, 1 = high risk)
    # Derived from a combination of high-sensitivity and low-sensitivity features
    high_score = df[HIGH_SENSITIVITY_FEATURES].mean(axis=1)
    low_score = df[LOW_SENSITIVITY_FEATURES].mean(axis=1)
    risk_score = 0.3 * high_score + 0.7 * low_score + np.random.normal(0, 0.02, n_samples)
    df["label"] = (risk_score > risk_score.median()).astype(int)

    return df


def label_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'sensitivity_class' column per-row.
    A record is HIGH if any high-sensitivity feature exceeds threshold 0.7.
    Otherwise it's LOW.
    """
    high_mask = (df[HIGH_SENSITIVITY_FEATURES] > 0.7).any(axis=1)
    df["sensitivity_class"] = np.where(high_mask, "HIGH", "LOW")
    print(f"[DataGen] Sensitivity distribution:\n{df['sensitivity_class'].value_counts()}")
    return df


def split_for_clients(df: pd.DataFrame, n_clients: int = NUM_CLIENTS):
    """
    Shard the dataset across n_clients (non-IID simulation:
    clients with higher IDs receive more HIGH-sensitivity records).

    Returns
    -------
    list of DataFrames, one per client
    """
    # Sort: HIGH sensitivity first, then LOW
    df_sorted = df.sort_values("sensitivity_class", ascending=False).reset_index(drop=True)
    shards = np.array_split(df_sorted, n_clients)
    return shards


def save_datasets(output_dir: str = "data/processed"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    # Full dataset
    df = generate_dataset()
    df = label_sensitivity(df)
    df.to_csv("data/raw/footprint_raw.csv", index=False)
    print("[DataGen] Raw dataset saved → data/raw/footprint_raw.csv")

    # Train / test split
    test_n = int(len(df) * TEST_SPLIT)
    df_test  = df.sample(n=test_n, random_state=RANDOM_SEED)
    df_train = df.drop(df_test.index)
    df_test.to_csv(f"{output_dir}/test.csv",  index=False)
    df_train.to_csv(f"{output_dir}/train.csv", index=False)

    # Per-client shards
    client_shards = split_for_clients(df_train)
    for i, shard in enumerate(client_shards):
        path = f"{output_dir}/client_{i}.csv"
        shard.to_csv(path, index=False)
        print(f"[DataGen] Client {i:2d} shard → {path}  ({len(shard)} records, "
              f"HIGH={( shard['sensitivity_class']=='HIGH').sum()})")

    print(f"\n[DataGen] ✓ Train={len(df_train)}, Test={len(df_test)} records ready.")
    return df_train, df_test, client_shards


if __name__ == "__main__":
    save_datasets()
