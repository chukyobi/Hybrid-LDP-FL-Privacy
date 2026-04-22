"""
experiments/run_baseline_fl.py
==============================
Baseline: Standard FL with NO privacy protection.
Used as the upper-bound utility / lower-bound privacy comparison.

Usage:
    python experiments/run_baseline_fl.py
"""

import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import NUM_ROUNDS, RESULTS_DIR, PLOTS_DIR
from data.data_generator import save_datasets
from fl.server import FLServer
from experiments.evaluate import evaluate, print_metrics


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    print("\n[Baseline FL] No privacy — raw data training...")
    df_train, df_test, client_shards = save_datasets()

    server = FLServer(client_shards=client_shards, privacy_mode="none")
    history = server.train(
        num_rounds=NUM_ROUNDS,
        evaluator=evaluate,
        test_data=df_test,
    )

    final = evaluate(server.global_model, df_test)
    print_metrics(final, label="Baseline FL Final")

    out = f"{RESULTS_DIR}/baseline_fl_results.json"
    with open(out, "w") as f:
        json.dump({"history": history, "final": final}, f, indent=2)
    print(f"[Baseline FL] Results saved → {out}")


if __name__ == "__main__":
    run()
