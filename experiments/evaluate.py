"""
experiments/evaluate.py
=======================
Evaluation functions for Chapter 4 metrics:
  1. Model Accuracy (utility)
  2. Re-identification Rate (privacy — simulated via distance-based attack)
  3. Training Speed (collected per round in server.py)
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import HIGH_SENSITIVITY_FEATURES, LOW_SENSITIVITY_FEATURES

FEATURE_COLS = HIGH_SENSITIVITY_FEATURES + LOW_SENSITIVITY_FEATURES


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """
    Evaluate global model on held-out test set.

    Returns
    -------
    dict with keys: accuracy, f1, re_id_rate
    """
    model.eval()
    X = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
    y_true = test_df["label"].values

    with torch.no_grad():
        logits  = model(X)
        y_pred  = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Simulated re-identification rate:
    # Percentage of test records whose privatized features are "close enough"
    # to the original to be reconstructed (proxy attack using L2 distance threshold).
    re_id_rate = _simulated_reid_rate(test_df)

    return {
        "accuracy":   round(acc,      4),
        "f1_score":   round(f1,       4),
        "re_id_rate": round(re_id_rate, 4),
    }


def _simulated_reid_rate(df: pd.DataFrame,
                          threshold: float = 0.15) -> float:
    """
    Simulates a gradient inversion / membership inference attack.

    Logic:
      - Original data variance across features.
      - If privatized values are still within `threshold` of the global
        feature mean (too clustered), attacker can guess original value.

    This is a proxy metric — replace with actual DLG attack from attacks/
    for final thesis results.
    """
    X = df[FEATURE_COLS].values.astype(np.float32)
    global_mean = X.mean(axis=0)
    distances   = np.linalg.norm(X - global_mean, axis=1)
    vulnerable  = (distances < threshold).sum()
    return vulnerable / len(df)


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print a metrics dictionary."""
    tag = f"[{label}]" if label else "[Eval]"
    print(f"\n{tag} ─── Results ─────────────────────────")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.4f}")
        else:
            print(f"  {k:<18}: {v}")
    print("─" * 42)
