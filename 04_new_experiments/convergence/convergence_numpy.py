"""Empirical convergence orders of the L1 and L1-2 Caputo quadratures.

Manufactured solution u(t)=t^p with the closed-form Caputo derivative
Gamma(p+1)/Gamma(p+1-a) t^{p-a}; error measured at t=T on refined uniform grids.
Expected orders: 2-a (L1) and 3-a (L1-2, Gao-Sun-Zhang 2014). Pure numpy.
"""
import os
import json
import numpy as np
from math import gamma

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# alpha=0.5 is the reported case; 0.3/0.7 confirm the order tracks 2-a / 3-a
NS = (50, 100, 200, 400)
ALPHAS = (0.3, 0.5, 0.7)
P = 3.0
T = 1.0


def a_coeff(k, a):
    return (k + 1.0) ** (1.0 - a) - k ** (1.0 - a)


def b_coeff(k, a):
    return (1.0 / (2.0 - a)) * ((k + 1.0) ** (2.0 - a) - k ** (2.0 - a)) \
        - 0.5 * ((k + 1.0) ** (1.0 - a) + k ** (1.0 - a))


def caputo_final_L1(u, dt, a):
    N = len(u) - 1
    coeff = dt ** (-a) / gamma(2.0 - a)
    diff = u[1:] - u[:-1]
    ak = a_coeff(np.arange(N), a)
    return coeff * np.dot(ak, diff[::-1])


def caputo_final_L12(u, dt, a):
    N = len(u) - 1
    if N < 2:
        return caputo_final_L1(u, dt, a)
    coeff = dt ** (-a) / gamma(2.0 - a)
    diff = u[1:] - u[:-1]
    ak = a_coeff(np.arange(N), a)
    bk = b_coeff(np.arange(N), a)
    c = np.empty(N)
    c[0] = ak[0] + bk[0]
    c[1:N - 1] = ak[1:N - 1] + bk[1:N - 1] - bk[0:N - 2]
    c[N - 1] = ak[N - 1] - bk[N - 2]
    return coeff * np.dot(c, diff[::-1])


def exact_caputo_power(t, p, a):
    return gamma(p + 1.0) / gamma(p + 1.0 - a) * t ** (p - a)


def run(alpha, p=P, T=T, ns=NS):
    exact = exact_caputo_power(T, p, alpha)
    rows = []
    prev = None
    for N in ns:
        t = np.linspace(0.0, T, N + 1)
        u = t ** p
        dt = T / N
        eL1 = abs(caputo_final_L1(u, dt, alpha) - exact)
        eL12 = abs(caputo_final_L12(u, dt, alpha) - exact)
        if prev is None:
            oL1 = oL12 = None
        else:
            oL1 = float(np.log(prev[0] / eL1) / np.log(2.0))
            oL12 = float(np.log(prev[1] / eL12) / np.log(2.0))
        rows.append(dict(N=N, errL1=eL1, ordL1=oL1, errL12=eL12, ordL12=oL12))
        prev = (eL1, eL12)
    return rows


if __name__ == "__main__":
    all_res = {}
    for alpha in ALPHAS:
        rows = run(alpha)
        all_res[f"alpha={alpha}"] = rows
        print(f"\nalpha = {alpha}  (u(t)=t^{P:.0f};  theory: L1 -> {2-alpha:.2f}, L1-2 -> {3-alpha:.2f})")
        print(f"{'N':>6}{'err L1':>13}{'ord L1':>9}{'err L1-2':>14}{'ord L1-2':>10}")
        for r in rows:
            o1 = f"{r['ordL1']:.2f}" if r["ordL1"] is not None else "  -- "
            o2 = f"{r['ordL12']:.2f}" if r["ordL12"] is not None else "  -- "
            print(f"{r['N']:>6}{r['errL1']:>13.3e}{o1:>9}{r['errL12']:>14.3e}{o2:>10}")

    with open(os.path.join(OUT, "convergence_orders.json"), "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nSaved {os.path.join(OUT, 'convergence_orders.json')}")
