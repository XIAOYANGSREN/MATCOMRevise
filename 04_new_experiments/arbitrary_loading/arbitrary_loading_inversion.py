"""Parameter inversion under arbitrary (non-step) loading, through the L1 forward
solver. Two strain histories: one with a sustained component
(E_inf identifiable, the manuscript case) and a purely oscillatory one (E_inf
poorly constrained). True params: tau=1.5, alpha=0.6, E0=120, Einf=30.
"""
import os
import json
from math import gamma
from pathlib import Path

import numpy as np
from scipy.special import rgamma
from scipy.optimize import least_squares


def zener_forward_l1(t, eps, alpha, tau, E0, Einf):
    """L1 forward solve of the fractional Zener equation for a given strain history."""
    nt = len(t); dt = t[1] - t[0]
    k = np.arange(nt, dtype=float)
    b = (k + 1.0) ** (1 - alpha) - k ** (1 - alpha)          # L1 weights b_k
    c = dt ** (-alpha) / gamma(2 - alpha)
    taua = tau ** alpha
    deps = eps[1:] - eps[:-1]
    d_eps = np.zeros(nt)
    for n in range(1, nt):
        d_eps[n] = c * np.dot(b[:n], deps[:n][::-1])
    sigma = np.zeros(nt); sigma[0] = E0 * eps[0]
    A = 1.0 + taua * c
    dsig = np.zeros(nt - 1)
    for n in range(1, nt):
        sum_rest = np.dot(b[1:n], dsig[:n - 1][::-1]) if n >= 2 else 0.0
        rhs = Einf * eps[n] + E0 * taua * d_eps[n]
        sigma[n] = (rhs + taua * c * (sigma[n - 1] - sum_rest)) / A
        dsig[n - 1] = sigma[n] - sigma[n - 1]
    return sigma


def ml_neg(x, a, J=120, Ja=8, xs=5.0):
    x = np.asarray(x, float); o = np.empty_like(x); sm = x <= xs; xss = x[sm]
    if xss.size:
        s = np.ones_like(xss); zp = np.ones_like(xss); z = -xss
        for kk in range(1, J):
            zp = zp * z; tt = zp * rgamma(a * kk + 1.0); s = s + tt
            if kk > 8 and np.max(np.abs(tt)) < 1e-16:
                break
        o[sm] = s
    xb = x[~sm]
    if xb.size:
        ac = np.zeros_like(xb)
        for kk in range(1, Ja + 1):
            ac += ((-1.) ** (kk - 1)) * xb ** (-kk) * rgamma(1. - a * kk)
        o[~sm] = ac
    return np.clip(o, 0, 1)


TRUE = dict(tau=1.5, alpha=0.6, E0=120.0, Einf=30.0)
TRUEv = np.array([TRUE['tau'], TRUE['alpha'], TRUE['E0'], TRUE['Einf']])
INITS = [(0.5, 0.4, 50, 20), (3.0, 0.8, 200, 10), (1.0, 0.3, 80, 40)]


def model(p, t, eps):
    log_tau, logit_a, log_Einf, log_dE = p
    tau = np.exp(log_tau); a = 1 / (1 + np.exp(-logit_a))
    Einf = np.exp(log_Einf); E0 = Einf + np.exp(log_dE)
    return zener_forward_l1(t, eps, a, tau, E0, Einf)


def unpack(p):
    return np.array([np.exp(p[0]), 1 / (1 + np.exp(-p[1])), np.exp(p[2]) + np.exp(p[3]), np.exp(p[2])])


def to_v(init):
    tau0, a0, E00, Ei0 = init; a0 = np.clip(a0, 0.06, 0.94)
    return [np.log(tau0), np.log(a0 / (1 - a0)), np.log(Ei0), np.log(max(E00 - Ei0, 1.0))]


def invert_sweep(t, eps, sigma_clean, noise_levels, n_seeds=5, example_at=None):
    """Multi-start least-squares inversion through the L1 solver, swept over noise."""
    results = {}
    example = None
    for noise in noise_levels:
        errs = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + int(1000 * noise))
            y = sigma_clean + noise * np.abs(sigma_clean) * rng.standard_normal(sigma_clean.shape)
            best, bestloss = None, np.inf
            for init in INITS:
                r = least_squares(lambda p: model(p, t, eps) - y, to_v(init), method='lm', max_nfev=4000)
                loss = np.mean(r.fun ** 2)
                if loss < bestloss:
                    bestloss, best = loss, unpack(r.x)
            errs.append(np.abs(best - TRUEv) / TRUEv * 100)
            if example_at is not None and noise == example_at[0] and seed == example_at[1]:
                r2 = 1 - np.sum((model(to_v(tuple(best)), t, eps) - y) ** 2) / np.sum((y - y.mean()) ** 2)
                example = (best, float(r2), y)
        errs = np.array(errs); mean = errs.mean(0); std = errs.std(0)
        results[f"{int(noise * 100)}%"] = {"tau": [mean[0], std[0]], "alpha": [mean[1], std[1]],
                                           "E0": [mean[2], std[2]], "Einf": [mean[3], std[3]]}
        print(f"noise {int(noise * 100):>2}%  | rel err (mean+/-std over {n_seeds} seeds):  "
              f"tau {mean[0]:.2f}+/-{std[0]:.2f}  alpha {mean[1]:.2f}+/-{std[1]:.2f}  "
              f"E0 {mean[2]:.2f}+/-{std[2]:.2f}  Einf {mean[3]:.2f}+/-{std[3]:.2f}  (%)")
    return results, example


# Self-check: under step strain the L1 forward solve must match the analytical relaxation.
t_chk = np.linspace(0, 20, 200)
sig_num = zener_forward_l1(t_chk, np.ones_like(t_chk), **TRUE)
sig_ana = TRUE['Einf'] + (TRUE['E0'] - TRUE['Einf']) * ml_neg((t_chk / TRUE['tau']) ** TRUE['alpha'], TRUE['alpha'])
chk = float(np.max(np.abs(sig_num - sig_ana) / np.abs(sig_ana)))
print(f"[self-check] L1 forward vs analytical relaxation (step strain): max rel diff = {chk*100:.2f}% (L1 discretisation error)")

A, w = 0.02, 2 * np.pi * 0.15
t = np.linspace(0, 40, 160)

# Arbitrary loading: a decaying-onset oscillation on a sustained rise. The sustained part
# lets the response approach equilibrium, so E_inf is identifiable.
eps = A * (1 - np.exp(-t / 0.5)) * np.sin(w * t) + A * (1 - np.exp(-t / 3.0))
sigma_clean = zener_forward_l1(t, eps, **TRUE)

# Purely oscillatory loading: never approaches equilibrium -> E_inf poorly constrained.
eps_osc = A * (1 - np.exp(-t / 0.5)) * np.sin(w * t)
sigma_clean_osc = zener_forward_l1(t, eps_osc, **TRUE)

print(f"\n=== Inversion under arbitrary (sustained-component) loading ===")
print(f"true: tau={TRUE['tau']}, alpha={TRUE['alpha']}, E0={TRUE['E0']}, Einf={TRUE['Einf']}\n")
results, example = invert_sweep(t, eps, sigma_clean, [0.0, 0.02, 0.05], example_at=(0.02, 0))
ex, r2, _ = example

print("\n=== Inversion under purely oscillatory loading (E_inf poorly constrained) ===\n")
results_osc, _ = invert_sweep(t, eps_osc, sigma_clean_osc, [0.02])

print(f"\nExample recovered @2% noise (sustained): tau={ex[0]:.3f}, alpha={ex[1]:.4f}, "
      f"E0={ex[2]:.2f}, Einf={ex[3]:.2f}  (R2={r2:.5f})")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "arbitrary.json"), "w") as f:
    json.dump({"true": TRUE, "selfcheck_reldiff_%": chk * 100,
               "results": results, "results_oscillatory": results_osc}, f, indent=2)

# Figure 13: arbitrary loading history (a) and recovered response (b).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t, eps, 'b-')
ax[0].set_xlabel('time (s)'); ax[0].set_ylabel(r'strain $\varepsilon(t)$')
ax[0].set_title('(a) arbitrary loading history'); ax[0].grid(alpha=.3)
rng = np.random.default_rng(0)
y = sigma_clean + 0.02 * np.abs(sigma_clean) * rng.standard_normal(sigma_clean.shape)
ax[1].plot(t, y, 'o', ms=3, alpha=.4, color='gray', label='noisy data (2%)')
ax[1].plot(t, model(to_v(tuple(ex)), t, eps), 'r-', lw=1.6, label='inverted model')
ax[1].set_xlabel('time (s)'); ax[1].set_ylabel(r'stress $\sigma(t)$')
ax[1].set_title('(b) recovered response'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout()
for e in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'arbitrary_loading.{e}'), dpi=150)
figdir = Path(__file__).resolve().parents[2] / "figures_out"
figdir.mkdir(exist_ok=True)
for e in ('png', 'pdf'):
    fig.savefig(str(figdir / f'Figure12_arbitrary.{e}'), dpi=150)
print(f"saved results/arbitrary.json, results/arbitrary_loading.png and {figdir / 'Figure12_arbitrary.png'}")
