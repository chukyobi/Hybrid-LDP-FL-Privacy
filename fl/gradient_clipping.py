"""
fl/gradient_clipping.py
=======================
Gradient clipping to bound the sensitivity of gradient updates.
This is the DP-SGD step: after computing gradients, clip each
gradient vector to max L2 norm = C.

Mathematical guarantee:
    ĝ = g / max(1, ‖g‖₂ / C)

This ensures that no single client's update can dominate the
aggregation or leak too much information.
"""

import torch
import torch.nn as nn
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MAX_GRAD_NORM


def clip_gradients(model: nn.Module, max_norm: float = MAX_GRAD_NORM) -> float:
    """
    Clip gradients of all model parameters in-place.

    Parameters
    ----------
    model    : nn.Module with computed gradients (after loss.backward())
    max_norm : C — maximum allowed L2 norm

    Returns
    -------
    total_norm : float — the pre-clipping gradient norm (for logging)
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

    return total_norm


def get_gradient_dict(model: nn.Module) -> dict:
    """
    Extract current gradients as a dictionary of tensors.
    Used before sending updates to the server.
    """
    return {
        name: param.grad.clone().detach()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def get_weight_dict(model: nn.Module) -> dict:
    """Extract current model weights as a detached dictionary."""
    return {
        name: param.data.clone().detach()
        for name, param in model.named_parameters()
    }
