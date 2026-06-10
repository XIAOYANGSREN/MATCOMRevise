"""Quick consistency checks (no training): L1 residual, L1 vs analytic step
relaxation, and PEKAN forward with true parameters. Run: python src/verify_math.py
"""

import math
import os
import sys
from dataclasses import dataclass

import torch

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "..", "..", "pykan"))

from utils.fractional import (
    caputo_derivative_l1_series,
    exact_solution_zener,
    solve_fractional_zener_l1,
)


@dataclass(frozen=True)
class ZenerParams:
    tau: float
    alpha: float
    E0: float
    E_inf: float


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rel_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = torch.maximum(torch.abs(b), torch.tensor(eps, device=b.device, dtype=b.dtype))
    return torch.abs(a - b) / denom


def check_l1_residual_general_loading(
    params: ZenerParams,
    t_max: float = 10.0,
    n: int = 200,
    atol: float = 5e-4,
) -> None:
    """
    Verify the discrete L1 solver is internally consistent:
      r_n = sigma_n + tau^alpha D^alpha sigma_n - (E_inf eps_n + E0 tau^alpha D^alpha eps_n)
    should be small (up to discretization error).
    """
    dev = _device()
    t = torch.linspace(0.0, t_max, n, device=dev, dtype=torch.float64)
    dt = float((t[1] - t[0]).item())

    # Smooth strain to avoid t=0 non-smoothness.
    omega = 2 * math.pi * 0.2
    gate = 1.0 - torch.exp(-t / 0.5)
    eps = 0.03 * gate * torch.sin(torch.tensor(omega, device=dev, dtype=t.dtype) * t)

    sigma = solve_fractional_zener_l1(
        t=t.to(dtype=torch.float32),
        epsilon=eps.to(dtype=torch.float32),
        alpha=float(params.alpha),
        tau_relax=float(params.tau),
        E0=float(params.E0),
        E_inf=float(params.E_inf),
    ).to(dtype=torch.float64)

    d_sigma = caputo_derivative_l1_series(sigma.to(dtype=torch.float32), dt=dt, alpha=float(params.alpha)).to(dtype=torch.float64)
    d_eps = caputo_derivative_l1_series(eps.to(dtype=torch.float32), dt=dt, alpha=float(params.alpha)).to(dtype=torch.float64)

    rhs = params.E_inf * eps + (params.E0 * (params.tau ** params.alpha)) * d_eps
    resid = sigma + (params.tau ** params.alpha) * d_sigma - rhs

    max_abs = float(torch.max(torch.abs(resid[1:])).item())
    if max_abs > atol:
        raise AssertionError(f"[L1 residual] max|resid|={max_abs:.3e} exceeds atol={atol:.3e}")


def check_step_relaxation_matches_exact(
    params: ZenerParams,
    epsilon0: float = 1.0,
    t_max: float = 20.0,
    n: int = 300,
    rel_tol: float = 0.03,
) -> None:
    """
    For step strain, compare L1 forward solve against analytic relaxation.
    This is a sanity check of `exact_solution_zener` + `solve_fractional_zener_l1`.
    """
    dev = _device()
    t = torch.linspace(0.0, t_max, n, device=dev, dtype=torch.float32)
    eps = torch.full_like(t, float(epsilon0))

    sigma_l1 = solve_fractional_zener_l1(
        t=t,
        epsilon=eps,
        alpha=float(params.alpha),
        tau_relax=float(params.tau),
        E0=float(params.E0),
        E_inf=float(params.E_inf),
    )
    sigma_exact = exact_solution_zener(t, float(params.tau), float(params.alpha), float(params.E0), float(params.E_inf), float(epsilon0))

    # Ignore the first few points (L1 discretization + initial singular behavior).
    k0 = 3
    rel = _rel_error(sigma_l1[k0:], sigma_exact[k0:])
    rel_max = float(torch.max(rel).item())
    if rel_max > rel_tol:
        raise AssertionError(f"[step relaxation] max relative error={rel_max:.3f} exceeds rel_tol={rel_tol:.3f}")


def check_pekan_forward_matches_exact(
    params: ZenerParams,
    epsilon0: float = 1.0,
    t_max: float = 20.0,
    n: int = 200,
    rel_tol: float = 5e-3,
) -> None:
    """
    PE-KAN forward with true parameters should match analytic relaxation very closely
    (up to Mittag-Leffler series truncation).
    """
    # Import here to avoid side effects during module import.
    from example_03_viscoelastic import PhysicsEncodedKAN

    dev = _device()
    t = torch.logspace(-2, math.log10(t_max), n, device=dev, dtype=torch.float32)

    model = PhysicsEncodedKAN(
        tau_init=float(params.tau),
        alpha_init=float(params.alpha),
        E0_init=float(params.E0),
        E_inf_init=float(params.E_inf),
        epsilon_0=float(epsilon0),
    ).to(dev)

    with torch.no_grad():
        sigma_pekan = model(t)
        sigma_exact = exact_solution_zener(t, float(params.tau), float(params.alpha), float(params.E0), float(params.E_inf), float(epsilon0))

    rel = _rel_error(sigma_pekan, sigma_exact)
    rel_max = float(torch.max(rel).item())
    if rel_max > rel_tol:
        raise AssertionError(f"[PE-KAN forward] max relative error={rel_max:.3e} exceeds rel_tol={rel_tol:.3e}")


def main() -> int:
    params = ZenerParams(tau=2.0, alpha=0.5, E0=100.0, E_inf=20.0)
    print("== verify_math ==")
    print("device:", _device())

    check_l1_residual_general_loading(params)
    print("PASS: L1 residual (general loading)")

    check_step_relaxation_matches_exact(params)
    print("PASS: L1 vs exact (step relaxation)")

    check_pekan_forward_matches_exact(params)
    print("PASS: PE-KAN forward vs exact (true params)")

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("FAIL:", e, file=sys.stderr)
        raise


