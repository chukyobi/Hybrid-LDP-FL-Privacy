"""
experiments/run_hybrid_ldp_fl.py
================================
YOUR ALGORITHM — The main experiment.
Runs the full Hybrid Context-Aware LDP + Federated Learning pipeline.

Usage:
    python experiments/run_hybrid_ldp_fl.py
"""

import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import NUM_ROUNDS, RESULTS_DIR, PLOTS_DIR
from data.data_generator import save_datasets
from fl.server import FLServer
from experiments.evaluate import evaluate, print_metrics


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    # ── Step 1: Generate / load data ──────────────────────────────────
    print("\n[Experiment] Generating synthetic footprint dataset...")
    df_train, df_test, client_shards = save_datasets()

    # ── Step 2: Initialize server with HYBRID mode ────────────────────
    server = FLServer(client_shards=client_shards, privacy_mode="hybrid")

    # ── Step 3: Train ─────────────────────────────────────────────────
    history = server.train(
        num_rounds=NUM_ROUNDS,
        evaluator=evaluate,
        test_data=df_test,
        verbose=False,
    )

    # ── Step 4: Final evaluation ──────────────────────────────────────
    final_metrics = evaluate(server.global_model, df_test, privacy_mode="hybrid")
    print_metrics(final_metrics, label="Hybrid LDP-FL Final")

    # ── Step 5: Save results ──────────────────────────────────────────
    out_path = f"{RESULTS_DIR}/hybrid_ldp_fl_results.json"
    with open(out_path, "w") as f:
        json.dump({"history": history, "final": final_metrics}, f, indent=2)
    print(f"\n[Experiment] Results saved → {out_path}")


if __name__ == "__main__":
    run()
