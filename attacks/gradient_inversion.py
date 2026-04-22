"""
attacks/gradient_inversion.py
==============================
Simulated Gradient Inversion Attack (DLG — Deep Leakage from Gradients).

Reference: Zhu et al., "Deep Leakage from Gradients" (NeurIPS 2019)

This module simulates an adversary attempting to reconstruct
original training data from intercepted model gradients.

Used for:
  - Measuring the Re-identification Rate metric in Chapter 4
  - Demonstrating why FL alone is insufficient (no LDP)
  - Showing Hybrid LDP-FL resists this attack

Usage:
    python attacks/gradient_inversion.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fl.model import get_model
from config import HIGH_SENSITIVITY_FEATURES, LOW_SENSITIVITY_FEATURES, NUM_FEATURES

FEATURE_COLS = HIGH_SENSITIVITY_FEATURES + LOW_SENSITIVITY_FEATURES


def gradient_inversion_attack(model: nn.Module,
                               true_X: torch.Tensor,
                               true_y: torch.Tensor,
                               n_iterations: int = 300,
                               lr: float = 0.1) -> tuple:
    """
    DLG Attack: Reconstruct training data (X_rec) by minimizing the
    L2 distance between intercepted gradients and gradients of dummy data.

    Parameters
    ----------
    model        : target FL model
    true_X       : genuine input tensor (what attacker wants to recover)
    true_y       : genuine label tensor
    n_iterations : optimization steps
    lr           : attack learning rate

    Returns
    -------
    (X_reconstructed, reconstruction_error)
    """
    criterion = nn.CrossEntropyLoss()

    # Compute "intercepted" gradients from true data
    model.zero_grad()
    out_true  = model(true_X)
    loss_true = criterion(out_true, true_y)
    grad_true = torch.autograd.grad(loss_true, model.parameters(),
                                    create_graph=False)
    grad_true = [g.detach() for g in grad_true]

    # Initialize dummy data to optimize
    X_dummy = torch.randn_like(true_X, requires_grad=True)
    y_dummy = torch.randint(0, 2, true_y.shape, dtype=torch.long)
    optimizer_attack = torch.optim.Adam([X_dummy], lr=lr)

    for i in range(n_iterations):
        optimizer_attack.zero_grad()
        model.zero_grad()

        out_dummy  = model(X_dummy)
        loss_dummy = criterion(out_dummy, y_dummy)
        grad_dummy = torch.autograd.grad(loss_dummy, model.parameters(),
                                         create_graph=True)

        # Gradient matching loss
        grad_diff = sum(
            ((gd - gt) ** 2).sum()
            for gd, gt in zip(grad_dummy, grad_true)
        )
        grad_diff.backward()
        optimizer_attack.step()

        if (i + 1) % 100 == 0:
            print(f"  [Attack] Iter {i+1}/{n_iterations}  "
                  f"grad_diff={grad_diff.item():.6f}")

    X_rec = X_dummy.detach()
    mse   = float(((X_rec - true_X.detach()) ** 2).mean())
    print(f"\n[Attack] Reconstruction MSE: {mse:.6f}")
    print(f"[Attack] Lower MSE → better reconstruction → weaker privacy")
    return X_rec, mse


def attack_experiment(model: nn.Module, sample_X: torch.Tensor,
                      sample_y: torch.Tensor, label: str = "") -> float:
    """
    Run an inversion attack and return the MSE.
    Lower MSE = attacker succeeds = bad for privacy.
    """
    print(f"\n{'='*50}")
    print(f" Gradient Inversion Attack [{label}]")
    print(f"{'='*50}")
    _, mse = gradient_inversion_attack(model, sample_X, sample_y)
    return mse


if __name__ == "__main__":
    # Demo: show attack on an untrained model
    model = get_model()
    X_demo = torch.randn(4, NUM_FEATURES)
    y_demo = torch.randint(0, 2, (4,))
    attack_experiment(model, X_demo, y_demo, label="Demo (untrained)")
