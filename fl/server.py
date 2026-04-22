"""
fl/server.py
============
Federated Learning Server — FedAvg Aggregation.

Implements the FedAvg algorithm (McMahan et al., 2017):
    w_global(t+1) = Σ_{k=1}^{K} (n_k / N) * w_k

Where:
  w_k  = local model weights from client k
  n_k  = number of training samples at client k
  N    = total samples across all selected clients
  K    = number of selected clients this round
"""

import torch
import numpy as np
import pandas as pd
import copy
import random
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    NUM_CLIENTS, CLIENT_FRACTION, NUM_ROUNDS,
    HIGH_SENSITIVITY_FEATURES, LOW_SENSITIVITY_FEATURES, RANDOM_SEED
)
from fl.model import get_model
from fl.client import FLClient

random.seed(RANDOM_SEED)
FEATURE_COLS = HIGH_SENSITIVITY_FEATURES + LOW_SENSITIVITY_FEATURES


def fedavg(client_updates: list) -> dict:
    """
    FedAvg aggregation.

    Parameters
    ----------
    client_updates : list of (weight_dict, n_samples) tuples

    Returns
    -------
    Aggregated global weight dictionary
    """
    total_samples = sum(n for _, n, _ in client_updates)
    global_weights = None

    for weight_dict, n_samples, _ in client_updates:
        weight = n_samples / total_samples
        if global_weights is None:
            global_weights = {
                k: v * weight for k, v in weight_dict.items()
            }
        else:
            for k in global_weights:
                global_weights[k] += weight_dict[k] * weight

    return global_weights


class FLServer:
    """
    Central aggregation server.
    Orchestrates client selection, weight distribution, and FedAvg.
    """

    def __init__(self, client_shards: list, privacy_mode: str = "hybrid"):
        """
        Parameters
        ----------
        client_shards : list of DataFrames — one per client
        privacy_mode  : "none" | "static" | "hybrid"
        """
        self.privacy_mode   = privacy_mode
        self.global_model   = get_model()
        self.global_weights = copy.deepcopy(self.global_model.state_dict())
        self.history        = []           # per-round metrics log

        # Instantiate all clients
        self.clients = [
            FLClient(client_id=i, data=shard, privacy_mode=privacy_mode)
            for i, shard in enumerate(client_shards)
        ]
        print(f"[Server] Initialized {len(self.clients)} clients  "
              f"| privacy_mode='{privacy_mode}'")

    def select_clients(self) -> list:
        """Randomly select CLIENT_FRACTION of clients for this round."""
        k = max(1, int(len(self.clients) * CLIENT_FRACTION))
        return random.sample(self.clients, k)

    def run_round(self, round_num: int,
                  epsilon_static: float = 1.0,
                  verbose: bool = False) -> dict:
        """
        Execute one FL round:
          1. Select clients
          2. Distribute global weights
          3. Collect local updates
          4. FedAvg aggregation

        Returns
        -------
        dict with round metrics (train_time, n_clients, …)
        """
        selected = self.select_clients()
        updates  = []
        round_times = []

        for client in selected:
            weights, n_samples, t = client.train(
                copy.deepcopy(self.global_weights),
                epsilon_static=epsilon_static,
                verbose=verbose
            )
            updates.append((weights, n_samples, t))
            round_times.append(t)

        # FedAvg
        self.global_weights = fedavg(updates)
        self.global_model.load_state_dict(self.global_weights)

        metrics = {
            "round":        round_num,
            "n_clients":    len(selected),
            "avg_time_sec": np.mean(round_times),
        }
        print(f"  [Server] Round {round_num:2d} | clients={len(selected)} "
              f"| avg_train_time={np.mean(round_times):.3f}s")
        return metrics

    def train(self, num_rounds: int = NUM_ROUNDS,
              epsilon_static: float = 1.0,
              evaluator=None,
              test_data: pd.DataFrame = None,
              verbose: bool = False) -> list:
        """
        Run the full FL training loop.

        Parameters
        ----------
        num_rounds     : number of communication rounds
        epsilon_static : ε for static LDP baseline
        evaluator      : optional evaluate() function (accuracy, re-id)
        test_data      : DataFrame for evaluation after each round
        verbose        : log per-epoch client loss

        Returns
        -------
        list of per-round metric dicts
        """
        print(f"\n{'='*60}")
        print(f"  FL Training Start — mode='{self.privacy_mode}'  "
              f"rounds={num_rounds}")
        print(f"{'='*60}")

        all_metrics = []
        for r in range(1, num_rounds + 1):
            round_metrics = self.run_round(
                round_num=r, epsilon_static=epsilon_static, verbose=verbose
            )
            if evaluator and test_data is not None:
                eval_metrics = evaluator(self.global_model, test_data)
                round_metrics.update(eval_metrics)
                print(f"           accuracy={eval_metrics.get('accuracy', 0):.4f}  "
                      f"re_id_rate={eval_metrics.get('re_id_rate', 0):.4f}")

            all_metrics.append(round_metrics)
            self.history.append(round_metrics)

        print(f"{'='*60}")
        print(f"  FL Training Complete.\n")
        return all_metrics
