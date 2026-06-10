"""Torch L1 and L1-2 quadratures for the Caputo derivative on a uniform grid.

L1: O(dt^{2-alpha}), Sun & Wu 2006. L1-2: O(dt^{3-alpha}), Gao-Sun-Zhang,
JCP 259 (2014), eq. (2.13). Both return du with du[0]=0 (zero history).
"""
import math

import torch


def _a_coeff(k, alpha):
    return (k + 1) ** (1.0 - alpha) - k ** (1.0 - alpha)


def _b_coeff(k, alpha):
    # L1-2 correction weight (Gao-Sun-Zhang 2014, eq. 2.13)
    inv = 1.0 / (2.0 - alpha)
    return inv * ((k + 1) ** (2.0 - alpha) - k ** (2.0 - alpha)) \
        - 0.5 * ((k + 1) ** (1.0 - alpha) + k ** (1.0 - alpha))


def caputo_deriv_L1(u, dt, alpha):
    nt = int(u.shape[0])
    du = torch.zeros(nt, device=u.device, dtype=u.dtype)
    if nt < 2:
        return du

    coeff = (dt ** (-alpha)) / math.gamma(2.0 - alpha)
    diff = (u[1:] - u[:-1]).to(torch.float64)
    a = torch.tensor([_a_coeff(k, alpha) for k in range(nt - 1)],
                     device=u.device, dtype=torch.float64)
    for n in range(1, nt):
        acc = (a[:n] * torch.flip(diff[:n], dims=[0])).sum()
        du[n] = (coeff * acc).to(u.dtype)
    return du


def caputo_deriv_L1_2(u, dt, alpha):
    # weights: c_0 = a_0+b_0, c_k = a_k+b_k-b_{k-1} (1<=k<=n-2),
    # c_{n-1} = a_{n-1}-b_{n-2}; n=1 falls back to L1.
    nt = int(u.shape[0])
    du = torch.zeros(nt, device=u.device, dtype=u.dtype)
    if nt < 2:
        return du

    coeff = (dt ** (-alpha)) / math.gamma(2.0 - alpha)
    diff = (u[1:] - u[:-1]).to(torch.float64)
    a = [_a_coeff(k, alpha) for k in range(nt)]
    b = [_b_coeff(k, alpha) for k in range(nt)]

    du[1] = (coeff * a[0] * diff[0]).to(u.dtype)
    for n in range(2, nt):
        c = [0.0] * n
        c[0] = a[0] + b[0]
        for k in range(1, n - 1):
            c[k] = a[k] + b[k] - b[k - 1]
        c[n - 1] = a[n - 1] - b[n - 2]
        c_t = torch.tensor(c, device=u.device, dtype=torch.float64)
        acc = (c_t * torch.flip(diff[:n], dims=[0])).sum()
        du[n] = (coeff * acc).to(u.dtype)
    return du


SCHEMES = {
    "L1": caputo_deriv_L1,
    "L1-2": caputo_deriv_L1_2,
}


def caputo_deriv(scheme, u, dt, alpha):
    return SCHEMES[scheme](u, dt, alpha)


def caputo_exact_power(t, p, alpha):
    """Caputo derivative of t^p: Gamma(p+1)/Gamma(p+1-a) * t^{p-a}."""
    coeff = math.gamma(p + 1.0) / math.gamma(p + 1.0 - alpha)
    out = torch.zeros_like(t)
    mask = t > 0
    out[mask] = coeff * t[mask] ** (p - alpha)
    return out
