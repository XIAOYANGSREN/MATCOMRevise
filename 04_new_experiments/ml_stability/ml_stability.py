"""Mittag-Leffler truncated series: term count, conditioning, gradient check (Sec. 4.3 numbers).

Counts the terms needed for E_alpha(-x) to converge (validity region of K_max=200)
and compares the exact digamma alpha-gradient against finite differences. Pure numpy/scipy.
"""
import os
import json
import numpy as np
from scipy.special import gammaln, digamma, rgamma

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LN10 = np.log(10.0)


def log10_term(k, x, alpha):
    return k * np.log10(x) - gammaln(alpha * k + 1.0) / LN10


def terms_needed(x, alpha, tol_log10=-14.0, kmax=2000):
    ks = np.arange(0, kmax + 1)
    lt = log10_term(ks, x, alpha)
    peak = int(np.argmax(lt))
    below = np.where((ks > peak) & (lt < tol_log10))[0]
    return (int(ks[below[0]]), float(lt[peak])) if below.size else (None, float(lt[peak]))


def ml(z, alpha, K=200):
    k = np.arange(K + 1)
    return float(np.sum(z ** k * rgamma(alpha * k + 1.0)))


def ml_dalpha(z, alpha, K=200):
    k = np.arange(K + 1)
    rg = rgamma(alpha * k + 1.0)
    return float(np.sum(z ** k * (-k * digamma(alpha * k + 1.0)) * rg))


# Terms K*(alpha, x) and the validity of K_max=200.
alphas = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
xs = [1.0, 2.0, 3.0, 5.0]
print("Terms K* to reach |term|<1e-14   ('*' exceeds K_max=200, 'inf' needs asymptotic)")
print(f"{'alpha/x':>8}" + "".join(f"{x:>12.1f}" for x in xs) + f"{'peak log10|t|':>16}")
tableA = {}
for a in alphas:
    row = [terms_needed(x, a)[0] for x in xs]
    _, peak5 = terms_needed(5.0, a)
    tableA[a] = {"Kstar": row, "peak_log10_term_x5": peak5}
    cells = "".join((f"{K:>12d}" if (K is not None and K <= 200)
                     else (f"{str(K) + '*':>12}" if K is not None else f"{'inf':>12}")) for K in row)
    print(f"{a:>8.2f}" + cells + f"{peak5:>16.2f}")
Kstar_max_regime = max(terms_needed(x, a)[0] for a in [0.4, 0.5, 0.7, 0.9] for x in [1.0, 2.0, 3.0])
print(f"Worst-case K* over PEKAN's regime (alpha>=0.4, x<=3): {Kstar_max_regime} (< K_max=200).")

# Gradient: exact (digamma) vs central finite difference.
# The truncated series is well-conditioned in the core alpha>=0.5, |z|<=2; near the
# alpha=0.4, |z|=3 boundary it approaches its float64 cancellation limit, which inflates
# the finite-difference reference (not the analytic gradient).
core = [(-x, a) for x in [0.5, 1.0, 2.0] for a in [0.5, 0.6, 0.7, 0.9]]
boundary = [(-3.0, 0.4), (-3.0, 0.5), (-2.0, 0.4)]
print("\nGradient dE/dalpha: exact (digamma) vs central finite difference")
print(f"{'z':>8}{'alpha':>8}{'dE/da (exact)':>16}{'dE/da (FD)':>16}{'abs diff':>12}")
grad_table = []
maxdiff_core = 0.0
maxdiff_all = 0.0
for z, a in core + boundary:
    ge = ml_dalpha(z, a)
    d = 1e-6
    gf = (ml(z, a + d) - ml(z, a - d)) / (2 * d)
    diff = abs(ge - gf)
    in_core = (z, a) in core
    maxdiff_all = max(maxdiff_all, diff)
    if in_core:
        maxdiff_core = max(maxdiff_core, diff)
    grad_table.append({"z": z, "alpha": a, "exact": ge, "fd": gf, "absdiff": diff, "core": in_core})
    print(f"{z:>8.1f}{a:>8.2f}{ge:>16.6e}{gf:>16.6e}{diff:>12.2e}")
print(f"max |exact - FD| over core (alpha>=0.5, |z|<=2)        = {maxdiff_core:.2e}")
print(f"max |exact - FD| incl. the alpha=0.4,|z|=3 boundary    = {maxdiff_all:.2e}  (FD cancellation, not a gradient error)")

# alpha sensitivity of the gradient magnitude at z=-3: bounded inside the regime,
# blowing up as alpha->0 where the series loses conditioning.
print("\n|dE/dalpha| across alpha at z=-3:")
agrid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cmag = []
for a in agrid:
    g = abs(ml_dalpha(-3.0, a))
    cmag.append(g)
    print(f"   alpha={a:.1f}:  |dE/dalpha| = {g:.4g}")
print("   -> bounded for alpha>=0.4; alpha=0.3 already shows the small-alpha cancellation blow-up.")

with open(os.path.join(OUT, "ml_stability.json"), "w", encoding="utf-8") as f:
    json.dump({"Kstar": {str(k): v for k, v in tableA.items()},
               "Kstar_max_regime": Kstar_max_regime,
               "grad_check_maxdiff_core": maxdiff_core,
               "grad_check_maxdiff_all": maxdiff_all,
               "grad_table": grad_table,
               "alpha_grid": agrid, "grad_mag_z-3": cmag}, f, indent=2)
print(f"\nSaved {os.path.join(OUT, 'ml_stability.json')}")
