"""
fl/client.py
============
Federated Learning Client.

Each client:
  1. Receives global model weights from server
  2. Applies LDP to its local data (context-aware or static, or none)
  3. Trains locally for LOCAL_EPOCHS
  4. Clips gradients (DP-SGD step)
  5. Returns updated weight delta to server

This simulates an on-device training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import sys, os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    BATCH_SIZE, LOCAL_EPOCHS, LEARNING_RATE, MAX_GRAD_NORM,
    HIGH_SENSITIVITY_FEATURES, LOW_SENSITIVITY_FEATURES
)
from fl.model import FootprintClassifier, get_model
from fl.gradient_clipping import clip_gradients, get_weight_dict
from ldp.context_aware_ldp import apply_context_aware_ldp, apply_static_ldp


FEATURE_COLS = HIGH_SENSITIVITY_FEATURES + LOW_SENSITIVITY_FEATURES


class FLClient:
    """
    Simulates a single federated learning participant (user device).
    """

    def __init__(self, client_id: int, data: pd.DataFrame,
                 privacy_mode: str = "hybrid"):
        """
        Parameters
        ----------
        client_id    : int — unique identifier
        data         : pd.DataFrame — this client's local shard
        privacy_mode : str — one of:
                         "none"    → standard FL, no LDP
                         "static"  → uniform ε LDP
                         "hybrid"  → context-aware LDP (YOUR ALGORITHM)
        """
        self.client_id    = client_id
        self.raw_data     = data
        self.privacy_mode = privacy_mode
        self.model        = get_model()
        self.criterion    = nn.CrossEntropyLoss()

    def _prepare_data(self, epsilon_static: float = 1.0) -> DataLoader:
        """Apply LDP according to privacy mode, then build DataLoader."""
        if self.privacy_mode == "hybrid":
            df = apply_context_aware_ldp(self.raw_data.copy(), FEATURE_COLS)
        elif self.privacy_mode == "static":
            df = apply_static_ldp(self.raw_data.copy(), epsilon=epsilon_static)
        else:                               # "none" → raw data
            df = self.raw_data.copy()

        X = torch.tensor(
            df[FEATURE_COLS].values, dtype=torch.float32
        )
        y = torch.tensor(
            df["label"].values, dtype=torch.long
        )
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    def train(self, global_weights: dict,
              epsilon_static: float = 1.0,
              verbose: bool = False) -> tuple:
        """
        Load global weights, train locally, return updated weights.

        Parameters
        ----------
        global_weights : dict — server's current global model state_dict
        epsilon_static : float — ε for static LDP baseline
        verbose        : bool — print per-epoch loss

        Returns
        -------
        (weight_dict, n_samples, train_time_sec)
        """
        # Load global model weights
        self.model.load_state_dict(global_weights)
        self.model.train()

        optimizer  = optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        dataloader = self._prepare_data(epsilon_static)
        n_samples  = len(dataloader.dataset)

        t0 = time.time()
        for epoch in range(LOCAL_EPOCHS):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss    = self.criterion(outputs, y_batch)
                loss.backward()

                # ── Gradient Clipping (DP-SGD step) ───────────────────────
                clip_gradients(self.model, max_norm=MAX_GRAD_NORM)

                optimizer.step()
                epoch_loss += loss.item()

            if verbose:
                avg_loss = epoch_loss / len(dataloader)
                print(f"  [Client {self.client_id}] Epoch {epoch+1}/{LOCAL_EPOCHS}  "
                      f"loss={avg_loss:.4f}")

        train_time = time.time() - t0
        return get_weight_dict(self.model), n_samples, train_time
