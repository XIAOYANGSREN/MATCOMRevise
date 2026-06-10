"""MLP baselines used against the KAN models."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain tanh MLP, layers e.g. [1, 32, 32, 32, 1]."""

    def __init__(self, layers, activation=nn.Tanh):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(activation())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class MLPPINN(MLP):
    """Same backbone; the physics loss is applied by the runner."""
    pass


class PhysicsEncodedMLP(nn.Module):
    """MLP body in front of the analytical Zener output layer."""

    def __init__(self, layers=(1, 32, 32, 32, 1),
                 E_inf_init=20.0, E0_init=100.0, epsilon_0=1.0):
        super().__init__()
        self.body = MLP(layers)
        self.epsilon_0 = epsilon_0
        self.log_E_inf = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(E_inf_init)))))
        self.log_delta_E = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(E0_init - E_inf_init))))
        )

    @property
    def E_inf(self):
        return torch.exp(self.log_E_inf)

    @property
    def E0(self):
        return self.E_inf + torch.exp(self.log_delta_E)

    def forward(self, t):
        y = self.body(t)
        return self.epsilon_0 * (self.E_inf + (self.E0 - self.E_inf) * y)


class InverseMLP(nn.Module):
    """4-parameter analytical Zener fit (log/logit reparametrised)."""

    def __init__(self, tau_init=1.0, alpha_init=0.5,
                 E0_init=100.0, E_inf_init=20.0, epsilon_0=1.0):
        super().__init__()
        import numpy as np

        self.epsilon_0 = epsilon_0
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau_init), dtype=torch.float32))
        a = float(np.clip(alpha_init, 0.01, 0.99))
        self.alpha_logit = nn.Parameter(torch.tensor(np.log(a / (1 - a)), dtype=torch.float32))
        self.log_E_inf = nn.Parameter(torch.tensor(np.log(E_inf_init), dtype=torch.float32))
        self.log_delta_E = nn.Parameter(
            torch.tensor(np.log(E0_init - E_inf_init), dtype=torch.float32)
        )

    @property
    def tau(self):
        return torch.exp(self.log_tau)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    @property
    def E_inf(self):
        return torch.exp(self.log_E_inf)

    @property
    def E0(self):
        return self.E_inf + torch.exp(self.log_delta_E)

    def forward(self, t):
        from utils.fractional import mittag_leffler_torch

        t_safe = torch.clamp(t / self.tau, min=1e-10)
        xi = -(t_safe ** self.alpha)
        y = mittag_leffler_torch(xi, self.alpha, beta=1.0)
        return self.epsilon_0 * (self.E_inf + (self.E0 - self.E_inf) * y)

    def get_physical_params(self):
        return {
            "tau": float(self.tau.item()),
            "alpha": float(self.alpha.item()),
            "E0": float(self.E0.item()),
            "E_inf": float(self.E_inf.item()),
        }
