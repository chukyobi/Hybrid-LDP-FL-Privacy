"""
experiments/run_static_ldp_fl.py
=================================
Baseline: FL with STATIC (uniform) LDP — same ε for all features.
Used to show that context-aware LDP outperforms static LDP on utility.

Usage:
    python experiments/run_static_ldp_fl.py
"""

import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import NUM_ROUNDS, RESULTS_DIR, PLOTS_DIR, EPSILON_HIGH
from data.data_generator import save_datasets
from fl.server import FLServer
from experiments.evaluate import evaluate, print_metrics


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    # Use ε_high for ALL features (worst-case uniform privacy)
    STATIC_EPSILON = EPSILON_HIGH

    print(f"\n[Static LDP-FL] Uniform ε={STATIC_EPSILON} on all features...")
    df_train, df_test, client_shards = save_datasets()

    server = FLServer(client_shards=client_shards, privacy_mode="static")
    history = server.train(
        num_rounds=NUM_ROUNDS,
        epsilon_static=STATIC_EPSILON,
        evaluator=evaluate,
        test_data=df_test,
    )

    final = evaluate(server.global_model, df_test)
    print_metrics(final, label="Static LDP-FL Final")

    out = f"{RESULTS_DIR}/static_ldp_fl_results.json"
    with open(out, "w") as f:
        json.dump({"history": history, "final": final,
                   "epsilon": STATIC_EPSILON}, f, indent=2)
    print(f"[Static LDP-FL] Results saved → {out}")


if __name__ == "__main__":
    run()
