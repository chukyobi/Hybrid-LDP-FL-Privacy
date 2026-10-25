"""
experiments/calibrate_epsilon.py
=================================
Privacy Budget Calibration Experiment (Chapter 4, Objective 4).

The Hybrid LDP-FL algorithm fixes ε_high = 0.5 and ε_low = 10.0
(config.py). This script justifies that choice empirically by sweeping
ε_high across a candidate grid — holding ε_low, the FL architecture,
the dataset, the client shards, and the model initialization fixed —
and measuring the resulting privacy-utility tradeoff (Accuracy, F1,
Re-ID Rate) at each point.

The candidate that maximizes Accuracy while keeping the Re-ID Rate
under a defined acceptable-risk threshold is selected as the
calibrated / optimal ε_high.

Usage:
    python experiments/calibrate_epsilon.py
"""

import json
import os
import sys
import random

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import NUM_ROUNDS, RESULTS_DIR, PLOTS_DIR, RANDOM_SEED, EPSILON_LOW
from data.data_generator import save_datasets
from fl.server import FLServer
from experiments.evaluate import evaluate
import ldp.context_aware_ldp as context_aware_ldp

# ── Calibration grid ───────────────────────────────────────────────
EPSILON_CANDIDATES = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]

# Acceptable-risk threshold: Re-ID rate must stay at/below this value
# for a candidate epsilon to be considered privacy-safe.
RE_ID_RISK_THRESHOLD = 0.25


def _reset_seeds(seed: int = RANDOM_SEED):
    """Reset all RNGs so every candidate starts from an identical state:
    same client-selection order, same model initialization. Only the
    Laplace noise draws (a function of epsilon itself) differ."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_candidate(epsilon_high: float, client_shards, df_test, num_rounds: int) -> dict:
    """Train one Hybrid LDP-FL run with a given ε_high and return final metrics."""
    context_aware_ldp.EPSILON_HIGH = epsilon_high
    _reset_seeds()

    server = FLServer(client_shards=client_shards, privacy_mode="hybrid")
    history = server.train(
        num_rounds=num_rounds,
        evaluator=evaluate,
        test_data=df_test,
        verbose=False,
    )
    final = history[-1]
    return {
        "epsilon_high": epsilon_high,
        "epsilon_low":  EPSILON_LOW,
        "accuracy":     final["accuracy"],
        "f1_score":     final["f1_score"],
        "re_id_rate":   final["re_id_rate"],
    }


def select_optimal(results: list) -> dict:
    """
    Select the calibrated epsilon: the candidate with the highest accuracy
    among those whose re_id_rate <= RE_ID_RISK_THRESHOLD. If no candidate
    satisfies the threshold, fall back to the lowest-risk candidate.
    """
    safe = [r for r in results if r["re_id_rate"] <= RE_ID_RISK_THRESHOLD]
    if safe:
        return max(safe, key=lambda r: r["accuracy"])
    return min(results, key=lambda r: r["re_id_rate"])


def run(num_rounds: int = NUM_ROUNDS):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n[Calibration] Generating synthetic footprint dataset (fixed across all candidates)...")
    _reset_seeds()
    df_train, df_test, client_shards = save_datasets()

    results = []
    for eps in EPSILON_CANDIDATES:
        print(f"\n[Calibration] === ε_high = {eps} (ε_low fixed = {EPSILON_LOW}) ===")
        metrics = run_candidate(eps, client_shards, df_test, num_rounds)
        print(f"[Calibration] ε_high={eps} -> accuracy={metrics['accuracy']:.4f} "
              f"f1={metrics['f1_score']:.4f} re_id_rate={metrics['re_id_rate']:.4f}")
        results.append(metrics)

    optimal = select_optimal(results)

    out_path = f"{RESULTS_DIR}/epsilon_calibration_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "num_rounds": num_rounds,
            "re_id_risk_threshold": RE_ID_RISK_THRESHOLD,
            "sweep": results,
            "selected_optimal": optimal,
        }, f, indent=2)

    print(f"\n[Calibration] Results saved -> {out_path}")
    print(f"[Calibration] Selected optimal epsilon_high = {optimal['epsilon_high']} "
          f"(accuracy={optimal['accuracy']:.4f}, re_id_rate={optimal['re_id_rate']:.4f})")

    return results, optimal


if __name__ == "__main__":
    run()
