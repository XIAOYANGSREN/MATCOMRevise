"""Fractional-calculus core: L1 Caputo discretisation, Mittag-Leffler function,
analytical fractional-Zener solutions, and the L1 forward solver.
"""

import torch
import numpy as np
from scipy.special import gamma as scipy_gamma
import torch.nn.functional as F


def b_coeff(k, alpha):
    """L1 weight b_k = (k+1)^(1-a) - k^(1-a)."""
    return (k + 1)**(1 - alpha) - k**(1 - alpha)


def caputo_derivative_l1(u, tau, alpha, n):
    """L1 Caputo derivative at time step n (u of shape [nt] or [nx, nt], step tau)."""
    device = u.device
    gamma_coeff = torch.lgamma(torch.tensor(2 - alpha, device=device)).exp()

    D_alpha_u = torch.zeros_like(u[..., n] if u.dim() > 1 else torch.tensor(0.0, device=device))

    for k in range(n):
        bk = b_coeff(k, alpha)
        if u.dim() == 1:
            D_alpha_u += bk * (u[n - k] - u[n - k - 1])
        else:
            D_alpha_u += bk * (u[..., n - k] - u[..., n - k - 1])

    return tau**(-alpha) / gamma_coeff * D_alpha_u


def caputo_derivative_l1_series(u, dt, alpha):
    """L1 Caputo derivative of a whole 1D series; du[0]=0 (zero history)."""
    if u.dim() != 1:
        raise ValueError("caputo_derivative_l1_series expects u.shape=[nt]")

    device = u.device
    nt = u.shape[0]
    du = torch.zeros(nt, device=device, dtype=u.dtype)

    gamma_coeff = torch.lgamma(torch.tensor(2 - alpha, device=device, dtype=torch.float64)).exp()
    coeff = (dt ** (-alpha)) / gamma_coeff

    for n in range(1, nt):
        acc = torch.zeros((), device=device, dtype=u.dtype)
        for k in range(n):
            acc = acc + b_coeff(k, alpha) * (u[n - k] - u[n - k - 1])
        du[n] = coeff.to(u.dtype) * acc

    return du


def caputo_l1_kernel(nt, alpha, device, dtype=torch.float64):
    """Precompute the (flipped) L1 kernel for conv1d-based evaluation.

    With x[i] = u[i+1]-u[i], du[n] = coeff * sum_k b_k x[n-1-k] is a causal
    convolution; conv1d does cross-correlation, hence the flip.
    """
    if nt < 2:
        raise ValueError("caputo_l1_kernel: nt must be >= 2")
    K = nt - 1
    b = torch.tensor([b_coeff(k, alpha) for k in range(K)], device=device, dtype=dtype)
    return torch.flip(b, dims=[0]).view(1, 1, -1)


def caputo_derivative_l1_series_fast(u, dt, alpha, w=None):
    """Vectorised L1 Caputo derivative (conv1d); same result as the loop version.

    Pass a precomputed kernel w from caputo_l1_kernel to avoid rebuilding it
    every training iteration.
    """
    if u.dim() != 1:
        raise ValueError("caputo_derivative_l1_series_fast expects u.shape=[nt]")
    nt = int(u.shape[0])
    if nt < 2:
        return torch.zeros_like(u)

    device = u.device
    dtype_u = u.dtype

    gamma_coeff = torch.lgamma(torch.tensor(2 - alpha, device=device, dtype=torch.float64)).exp()
    coeff64 = (dt ** (-alpha)) / gamma_coeff

    if w is None:
        w = caputo_l1_kernel(nt=nt, alpha=alpha, device=device, dtype=torch.float64)

    x = (u[1:] - u[:-1]).to(dtype=torch.float64).view(1, 1, -1)
    K = nt - 1
    x_pad = F.pad(x, (K - 1, 0))  # left padding for causality
    y = F.conv1d(x_pad, w)

    du = torch.zeros(nt, device=device, dtype=dtype_u)
    du[1:] = (coeff64 * y.view(-1)).to(dtype=dtype_u)
    return du


def mittag_leffler_series(z, alpha, beta=1.0, max_terms=100, tol=1e-12,
                          clip_negative=True):
    """Truncated series E_{a,b}(z) = sum_k z^k / Gamma(ak+b), torch or numpy input.

    For z<0 the truncation can slightly leave (0,1]; clip_negative keeps the
    old clamped behaviour used by the analytical solutions and plots.
    """
    is_torch = isinstance(z, torch.Tensor)

    if is_torch:
        z = z.double()
        result = torch.zeros_like(z)
        z_power = torch.ones_like(z)
    else:
        z = np.asarray(z, dtype=np.float64)
        result = np.zeros_like(z)
        z_power = np.ones_like(z)

    for k in range(max_terms):
        gamma_val = scipy_gamma(alpha * k + beta)
        if np.isinf(gamma_val) or gamma_val == 0:
            break

        term = z_power / gamma_val
        result = result + term
        z_power = z_power * z

        if is_torch:
            max_term = torch.max(torch.abs(term)).item()
        else:
            max_term = np.max(np.abs(term))

        if max_term < tol:
            break

    if clip_negative:
        if is_torch:
            result = torch.where(z < 0, torch.clamp(result, 0.0, 1.0), result)
        else:
            result = np.where(z < 0, np.clip(result, 0.0, 1.0), result)

    if is_torch:
        return result.float()
    return result.astype(np.float32)


def mittag_leffler_torch(z, alpha, beta=1.0, max_terms=200, tol=1e-14,
                         clamp_negative=None):
    """Differentiable E_{a,b}(z) in pure torch (float64 internally).

    alpha may be a tensor so the gradient w.r.t. the fractional order
    propagates -- this is what PEKAN trains through. clamp_negative=(lo,hi)
    clamps for z<0 but kinks the gradient, so leave it None during training.
    """
    if not isinstance(z, torch.Tensor):
        raise TypeError("mittag_leffler_torch: z must be a torch.Tensor")

    device = z.device
    dtype_in = z.dtype
    z64 = z.to(dtype=torch.float64)

    if isinstance(alpha, torch.Tensor):
        alpha64 = alpha.to(device=device, dtype=torch.float64)
    else:
        alpha64 = torch.tensor(float(alpha), device=device, dtype=torch.float64)

    if isinstance(beta, torch.Tensor):
        beta64 = beta.to(device=device, dtype=torch.float64)
    else:
        beta64 = torch.tensor(float(beta), device=device, dtype=torch.float64)

    gamma0 = torch.exp(torch.lgamma(beta64))
    result = torch.ones_like(z64) / gamma0
    term = result.clone()

    z_power = torch.ones_like(z64)
    for k in range(1, max_terms):
        z_power = z_power * z64
        gamma_k = torch.exp(torch.lgamma(alpha64 * k + beta64))
        term = z_power / gamma_k
        result = result + term

        if float(torch.max(torch.abs(term)).item()) < tol:
            break

    if clamp_negative is not None:
        lo, hi = clamp_negative
        result = torch.where(z64 < 0, torch.clamp(result, lo, hi), result)

    return result.to(dtype=dtype_in)


def mittag_leffler(z, alpha, beta=1.0):
    """E_{a,b}(z); dispatches to the differentiable version when alpha is a tensor."""
    if isinstance(alpha, torch.Tensor):
        return mittag_leffler_torch(z, alpha, beta=beta)
    return mittag_leffler_series(z, float(alpha), beta)


def exact_solution_ode_1d(t, alpha):
    """u(t) = t^(5+alpha), exact solution of D^a u + u = f."""
    return t ** (5 + alpha)


def source_term_ode_1d(t, alpha):
    """f(t) = Gamma(6+alpha)/120 * t^5 + t^(5+alpha)."""
    gamma_part = torch.lgamma(torch.tensor(6 + alpha, device=t.device)).exp() / 120
    return gamma_part * t**5 + t**(5 + alpha)


def exact_solution_pde_2d(x, t, alpha):
    """u(x,t) = x^2 + 2 t^a / Gamma(1+a), exact solution of D_t^a u = u_xx."""
    gamma_part = torch.lgamma(torch.tensor(1 + alpha, device=x.device)).exp()
    t_safe = torch.clamp(t, min=1e-10)
    return x**2 + 2 * t_safe**alpha / gamma_part


def exact_solution_zener(t, tau, alpha, E0, E_inf, epsilon_0=1.0):
    """Fractional Zener stress relaxation: eps0 [E_inf + (E0-E_inf) E_a(-(t/tau)^a)]."""
    xi = -(t / tau) ** alpha
    ml_val = mittag_leffler(xi, alpha)
    return epsilon_0 * (E_inf + (E0 - E_inf) * ml_val)


def solve_fractional_zener_l1(t, epsilon, alpha, tau_relax, E0, E_inf):
    """L1 forward solve of sigma + tau^a D^a sigma = E_inf eps + E0 tau^a D^a eps
    on a uniform grid, with sigma(0) = E0 eps(0).
    """
    if t.shape[0] != epsilon.shape[0]:
        raise ValueError("t and epsilon must have the same length")
    if t.shape[0] < 2:
        raise ValueError("need nt >= 2")

    # tolerate float32 linspace jitter but reject genuinely non-uniform grids
    diffs = t[1:] - t[:-1]
    dt_mean = diffs.mean()
    tol = max(1e-5, 1e-4 * float(torch.abs(dt_mean).item()))
    if float((diffs.max() - diffs.min()).abs().item()) > tol:
        raise ValueError("solve_fractional_zener_l1 needs a (near-)uniform time grid")
    dt = float(dt_mean.item())

    device = t.device
    nt = t.shape[0]
    sigma = torch.zeros(nt, device=device, dtype=epsilon.dtype)
    sigma[0] = E0 * epsilon[0]

    d_eps = caputo_derivative_l1_series(epsilon, dt=dt, alpha=alpha)

    gamma_coeff = torch.lgamma(torch.tensor(2 - alpha, device=device, dtype=torch.float64)).exp()
    c = (dt ** (-alpha)) / gamma_coeff
    tau_a = (tau_relax ** alpha)

    for n in range(1, nt):
        sum_rest = torch.zeros((), device=device, dtype=sigma.dtype)
        for k in range(1, n):
            sum_rest = sum_rest + b_coeff(k, alpha) * (sigma[n - k] - sigma[n - k - 1])

        rhs = (E_inf * epsilon[n]) + (E0 * tau_a) * d_eps[n]
        A = 1.0 + tau_a * c.item()
        sigma[n] = (rhs + (tau_a * c.item()) * (sigma[n - 1] - sum_rest)) / A

    return sigma
