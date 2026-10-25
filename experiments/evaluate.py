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


def evaluate(model, test_df: pd.DataFrame, privacy_mode: str = "none", epsilon_static: float = 1.0) -> dict:
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
    # Percentage of test records whose privatized features can be successfully
    # linked back to the original records via a nearest-neighbor attack.
    re_id_rate = _simulated_reid_rate(test_df, privacy_mode, epsilon_static)

    return {
        "accuracy":   round(acc,      4),
        "f1_score":   round(f1,       4),
        "re_id_rate": round(re_id_rate, 4),
    }


def _simulated_reid_rate(df: pd.DataFrame,
                          privacy_mode: str,
                          epsilon_static: float) -> float:
    """
    Simulates a linkage attack (re-identification attack).
    For each user record, we apply the LDP mechanism to get a noisy representation.
    We then try to find the closest match in the original database.
    If the closest match is the correct original record, the user is re-identified.
    """
    X = df[FEATURE_COLS].values.astype(np.float32)
    
    if privacy_mode == "none":
        X_noisy = X
    elif privacy_mode == "static":
        from ldp.context_aware_ldp import apply_static_ldp
        # Disable print statement inside apply_static_ldp for cleaner logs during evaluation
        import sys, os
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            df_noisy = apply_static_ldp(df.copy(), epsilon=epsilon_static)
        finally:
            sys.stdout = old_stdout
        X_noisy = df_noisy[FEATURE_COLS].values.astype(np.float32)
    elif privacy_mode == "hybrid":
        from ldp.context_aware_ldp import apply_context_aware_ldp
        # Disable print statement inside apply_context_aware_ldp for cleaner logs during evaluation
        import sys, os
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            df_noisy = apply_context_aware_ldp(df.copy(), FEATURE_COLS)
        finally:
            sys.stdout = old_stdout
        X_noisy = df_noisy[FEATURE_COLS].values.astype(np.float32)
    else:
        X_noisy = X

    # Linkage attack using vectorised numpy distances
    # Pairwise squared Euclidean distance between X_noisy (N, D) and X (N, D)
    # dists_sq[i, j] = sum((X_noisy[i] - X[j])^2)
    dists_sq = np.sum((X_noisy[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
    nearest_indices = np.argmin(dists_sq, axis=1)
    
    re_identified = (nearest_indices == np.arange(len(X))).sum()
    return float(re_identified / len(X))


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
