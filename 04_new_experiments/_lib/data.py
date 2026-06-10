"""Synthetic Zener relaxation data with a shared noise/seed protocol."""

from dataclasses import dataclass

import numpy as np
import torch

from . import paths  # noqa: F401  (puts 03_code/src on sys.path)
from utils.fractional import exact_solution_zener


@dataclass
class ZenerParams:
    tau: float = 2.0
    alpha: float = 0.5
    E0: float = 100.0
    E_inf: float = 20.0
    epsilon_0: float = 1.0
    t_max: float = 20.0


def make_time_grid(n_points, t_max, spacing="linear", device="cpu"):
    if spacing == "linear":
        return torch.linspace(0.0, t_max, n_points, device=device)
    if spacing == "log":
        return torch.logspace(-2, np.log10(t_max), n_points, device=device)
    raise ValueError(f"unknown spacing {spacing!r}")


def zener_exact(t, p):
    return exact_solution_zener(t, p.tau, p.alpha, p.E0, p.E_inf, p.epsilon_0)


def add_noise(sigma, noise_level, seed):
    """sigma * (1 + noise_level*N(0,1)), seeded."""
    if noise_level <= 0.0:
        return sigma.clone()
    g = torch.Generator(device=sigma.device).manual_seed(int(seed))
    eps = torch.randn(sigma.shape, generator=g, device=sigma.device, dtype=sigma.dtype)
    return sigma + noise_level * sigma * eps


def make_relaxation_dataset(p, n_points=100, spacing="log", noise_level=0.0,
                            seed=0, device="cpu"):
    """Return (t, sigma_clean, sigma_noisy)."""
    t = make_time_grid(n_points, p.t_max, spacing=spacing, device=device)
    sigma_clean = zener_exact(t, p)
    sigma_noisy = add_noise(sigma_clean, noise_level, seed)
    return t, sigma_clean, sigma_noisy
