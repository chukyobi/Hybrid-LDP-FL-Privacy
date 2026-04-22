"""
ldp/context_aware_ldp.py
========================
THE CORE INNOVATION of this thesis:
Context-Aware LDP Router.

Instead of applying a uniform ε (epsilon) to all data, this module:
  1. Reads the sensitivity label of each feature / record.
  2. Routes HIGH-sensitivity data → ε_high (small → more noise)
  3. Routes LOW-sensitivity data  → ε_low  (larger → less noise)

This preserves model utility while giving stronger guarantees exactly
where they matter most.
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    EPSILON_HIGH, EPSILON_LOW, GLOBAL_SENSITIVITY,
    HIGH_SENSITIVITY_FEATURES, LOW_SENSITIVITY_FEATURES
)
from ldp.laplace_mechanism import privatize_vector


def build_epsilon_vector(feature_names: list) -> np.ndarray:
    """
    Build a per-feature ε vector.
    HIGH-sensitivity features → EPSILON_HIGH
    LOW-sensitivity features  → EPSILON_LOW

    Parameters
    ----------
    feature_names : list of str — ordered feature column names

    Returns
    -------
    np.ndarray of shape (n_features,)
    """
    eps_vec = []
    for feat in feature_names:
        if feat in HIGH_SENSITIVITY_FEATURES:
            eps_vec.append(EPSILON_HIGH)
        else:
            eps_vec.append(EPSILON_LOW)
    return np.array(eps_vec, dtype=np.float32)


def build_sensitivity_vector(feature_names: list,
                              global_sensitivity: float = GLOBAL_SENSITIVITY) -> np.ndarray:
    """
    Build a per-feature sensitivity (Δf) vector.
    All features share the same global sensitivity in this implementation.
    Can be overridden per-feature for more granular control.
    """
    return np.full(len(feature_names), global_sensitivity, dtype=np.float32)


def apply_context_aware_ldp(df: pd.DataFrame,
                             feature_cols: list = None) -> pd.DataFrame:
    """
    Apply context-aware Laplace LDP to a DataFrame.

    For each row, HIGH-sensitivity columns get ε_high noise,
    LOW-sensitivity columns get ε_low noise.

    Parameters
    ----------
    df           : pd.DataFrame — raw or loaded dataset
    feature_cols : list of feature column names to privatize
                   (defaults to ALL_FEATURES from config)

    Returns
    -------
    pd.DataFrame with privatized feature columns, plus
    'label' and 'sensitivity_class' unchanged.
    """
    if feature_cols is None:
        feature_cols = HIGH_SENSITIVITY_FEATURES + LOW_SENSITIVITY_FEATURES

    # Keep non-feature columns unchanged
    meta_cols = [c for c in df.columns if c not in feature_cols]
    df_meta = df[meta_cols].copy()

    X = df[feature_cols].values.astype(np.float32)

    eps_vec  = build_epsilon_vector(feature_cols)
    sens_vec = build_sensitivity_vector(feature_cols)

    X_noisy = privatize_vector(X, sensitivities=sens_vec, epsilons=eps_vec)

    df_noisy = pd.DataFrame(X_noisy, columns=feature_cols, index=df.index)
    result = pd.concat([df_noisy, df_meta], axis=1)

    print(f"[ContextLDP] Applied context-aware LDP  "
          f"(ε_high={EPSILON_HIGH}, ε_low={EPSILON_LOW})  "
          f"to {len(df)} records × {len(feature_cols)} features.")
    return result


def apply_static_ldp(df: pd.DataFrame,
                     epsilon: float,
                     feature_cols: list = None) -> pd.DataFrame:
    """
    Baseline: apply a uniform/static ε to ALL features regardless of sensitivity.
    Used for comparison in experiments.
    """
    if feature_cols is None:
        feature_cols = HIGH_SENSITIVITY_FEATURES + LOW_SENSITIVITY_FEATURES

    meta_cols = [c for c in df.columns if c not in feature_cols]
    df_meta = df[meta_cols].copy()

    X = df[feature_cols].values.astype(np.float32)
    eps_vec  = np.full(len(feature_cols), epsilon, dtype=np.float32)
    sens_vec = build_sensitivity_vector(feature_cols)

    X_noisy = privatize_vector(X, sensitivities=sens_vec, epsilons=eps_vec)
    df_noisy = pd.DataFrame(X_noisy, columns=feature_cols, index=df.index)
    result = pd.concat([df_noisy, df_meta], axis=1)

    print(f"[StaticLDP] Applied uniform LDP  (ε={epsilon})  "
          f"to {len(df)} records × {len(feature_cols)} features.")
    return result
