"""PIKAN (Sec. 4.2): a KAN approximates sigma(t), trained on the L1-discretised
residual of the fractional Zener equation plus an initial-condition penalty.
example_01 imports zener_relaxation_loss from here.
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.abspath(os.path.join(_SRC, "..", ".."))
for _p in (_SRC, os.path.join(_PKG, "pykan")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from utils.fractional import caputo_l1_kernel, caputo_derivative_l1_series_fast


def build_pikan(width=(1, 10, 1), grid=12, k=3, device=None,
                grid_eps=1.0, noise_scale=0.1):
    """KAN approximator for sigma(t); feed it time mapped to [-1, 1]."""
    from kan import KAN
    model = KAN(width=list(width), grid=grid, k=k,
                grid_eps=grid_eps, noise_scale=noise_scale, device=device)
    try:
        model.speed()  # skip the symbolic branch
    except Exception:
        pass
    return model


def zener_relaxation_loss(sigma_pred, eps, dt, alpha, tau, E0, E_inf,
                          epsilon0, w_l1=None, ic_weight=10.0):
    """mean(residual[1:]^2) + ic_weight*(sigma[0]-E0*eps0)^2, residual via L1.

    Pass the precomputed L1 kernel w_l1 to avoid rebuilding it every iteration.
    """
    d_sigma = caputo_derivative_l1_series_fast(sigma_pred, dt=dt, alpha=alpha, w=w_l1)
    rhs = E_inf * eps
    residual = sigma_pred + (tau ** alpha) * d_sigma - rhs
    ic = sigma_pred[0] - (E0 * epsilon0)
    return torch.mean(residual[1:] ** 2) + ic_weight * (ic ** 2)


__all__ = ["build_pikan", "zener_relaxation_loss", "caputo_l1_kernel"]
