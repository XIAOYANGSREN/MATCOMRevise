"""PEKAN (Sec. 4.3): the closed-form Zener relaxation solution encoded as a
3-layer network, so the 4 trainable scalars are (tau, alpha, E0, E_inf).
example_03_viscoelastic.py carries an equivalent inline copy of these classes.
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import torch
import torch.nn as nn

from utils.fractional import mittag_leffler_torch


class PowerTransformLayer(nn.Module):
    """phi_1(t; tau, alpha) = -(t/tau)^alpha; log/logit reparam keeps tau>0, 0<alpha<1."""

    def __init__(self, tau_init=1.0, alpha_init=0.5):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau_init), dtype=torch.float32))
        alpha_init = np.clip(alpha_init, 0.01, 0.99)
        alpha_logit = np.log(alpha_init / (1 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_normalized = t / self.tau
        t_normalized = torch.clamp(t_normalized, min=1e-10)
        xi = -(t_normalized ** self.alpha)
        return xi


class MittagLefflerLayer(nn.Module):
    """phi_2(xi; alpha) = E_alpha(xi), the physics-fixed activation."""

    def forward(self, xi, alpha):
        # alpha must stay a tensor (no .item()) so its gradient propagates
        return mittag_leffler_torch(xi, alpha, beta=1.0)


class LinearScaleLayer(nn.Module):
    """phi_3(y; E0, E_inf) = eps0 [E_inf + (E0-E_inf) y]; log reparam keeps E0>E_inf>0."""

    def __init__(self, E0_init=100.0, E_inf_init=20.0, epsilon_0=1.0):
        super().__init__()
        self.epsilon_0 = epsilon_0
        self.log_E_inf = nn.Parameter(torch.tensor(np.log(E_inf_init), dtype=torch.float32))
        self.log_delta_E = nn.Parameter(torch.tensor(np.log(E0_init - E_inf_init), dtype=torch.float32))

    @property
    def E_inf(self) -> torch.Tensor:
        return torch.exp(self.log_E_inf)

    @property
    def E0(self) -> torch.Tensor:
        return self.E_inf + torch.exp(self.log_delta_E)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        sigma = self.epsilon_0 * (self.E_inf + (self.E0 - self.E_inf) * y)
        return sigma


class PhysicsEncodedKAN(nn.Module):
    """PEKAN: t -> -(t/tau)^alpha -> E_alpha(.) -> E_inf+(E0-E_inf)y.

    Fitting the four scalars to data is the parameter inversion;
    get_physical_params() reads off (tau, alpha, E0, E_inf).
    """

    def __init__(self, tau_init, alpha_init, E0_init, E_inf_init, epsilon_0=1.0):
        super().__init__()
        self.layer1 = PowerTransformLayer(tau_init, alpha_init)
        self.layer2 = MittagLefflerLayer()
        self.layer3 = LinearScaleLayer(E0_init, E_inf_init, epsilon_0)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        xi = self.layer1(t)
        y = self.layer2(xi, self.layer1.alpha)
        sigma = self.layer3(y)
        return sigma

    def get_physical_params(self) -> dict:
        """Return the four recovered material parameters."""
        return {
            'tau': self.layer1.tau.item(),
            'alpha': self.layer1.alpha.item(),
            'E0': self.layer3.E0.item(),
            'E_inf': self.layer3.E_inf.item(),
        }


__all__ = [
    "PowerTransformLayer",
    "MittagLefflerLayer",
    "LinearScaleLayer",
    "PhysicsEncodedKAN",
]
