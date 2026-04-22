"""
ldp/laplace_mechanism.py
========================
Implements the Laplace Mechanism for Local Differential Privacy.

Formula:  x' = x + Laplace(0, Δf / ε)

Where:
  x   = original feature value
  Δf  = global sensitivity of the function (max change one record can cause)
  ε   = privacy budget (smaller → more noise → stronger privacy)
  x'  = privatized (noisy) output

Reference: Dwork et al., "Calibrating Noise to Sensitivity in Private Data Analysis"
"""

import numpy as np
from typing import Union


def laplace_noise(sensitivity: float, epsilon: float, size=None) -> np.ndarray:
    """
    Sample noise from Laplace(0, sensitivity/epsilon).

    Parameters
    ----------
    sensitivity : float  — Δf, global sensitivity
    epsilon     : float  — privacy budget ε  (must be > 0)
    size        : int or tuple — shape of noise array

    Returns
    -------
    np.ndarray of Laplace noise
    """
    if epsilon <= 0:
        raise ValueError(f"ε must be positive, got {epsilon}")
    b = sensitivity / epsilon          # scale parameter
    return np.random.laplace(loc=0.0, scale=b, size=size)


def privatize_value(x: Union[float, np.ndarray],
                    sensitivity: float,
                    epsilon: float) -> np.ndarray:
    """
    Apply Laplace mechanism to a scalar or array of feature values.

    Parameters
    ----------
    x           : original value(s)
    sensitivity : Δf
    epsilon     : ε

    Returns
    -------
    Noisy x' = x + Laplace(0, Δf/ε)
    """
    x = np.asarray(x, dtype=np.float32)
    noise = laplace_noise(sensitivity, epsilon, size=x.shape).astype(np.float32)
    return x + noise


def privatize_vector(x_vec: np.ndarray,
                     sensitivities: np.ndarray,
                     epsilons: np.ndarray) -> np.ndarray:
    """
    Applies per-feature Laplace noise.
    Each feature i gets its own sensitivity[i] and epsilon[i].

    Parameters
    ----------
    x_vec         : shape (n_features,) or (n_samples, n_features)
    sensitivities : shape (n_features,)
    epsilons      : shape (n_features,)

    Returns
    -------
    Noisy array same shape as x_vec
    """
    x_vec = np.asarray(x_vec, dtype=np.float32)
    noise = np.zeros_like(x_vec)

    if x_vec.ndim == 1:
        for i in range(len(sensitivities)):
            noise[i] = laplace_noise(sensitivities[i], epsilons[i])
    else:
        for i in range(len(sensitivities)):
            noise[:, i] = laplace_noise(sensitivities[i], epsilons[i], size=x_vec.shape[0])

    return x_vec + noise
