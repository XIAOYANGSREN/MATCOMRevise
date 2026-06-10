"""Times the 4-parameter analytical Zener least-squares fit:
same fit as noise_robustness/noise_sweep.py, Example-3 data at 2% noise, mean
over 5 realisations x 3 initial guesses. Writes results/analytical_ls.json.
"""
import json
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import rgamma

OUT = Path(__file__).resolve().parent / "results"

# true Example-3 parameters, identical to noise_sweep.py
TAU, ALPHA, E0, EINF = 2.0, 0.5, 100.0, 20.0
TRUE = np.array([TAU, ALPHA, E0, EINF])
LVL = 0.02
N_SEED, N_INIT = 5, 3
INITS = [  # (tau, alpha, E0, Einf)
    (0.5, 0.7, 60.0, 40.0),
    (5.0, 0.3, 150.0, 10.0),
    (1.0, 0.6, 80.0, 30.0),
]


def ml_neg(x, a, n_series=120, n_asym=8, x_split=5.0):
    x = np.asarray(x, float); out = np.empty_like(x)
    sm = x <= x_split; xs = x[sm]
    if xs.size:
        s = np.ones_like(xs); zp = np.ones_like(xs); z = -xs
        for k in range(1, n_series):
            zp = zp * z; term = zp * rgamma(a * k + 1.0); s = s + term
            if k > 8 and np.max(np.abs(term)) < 1e-16:
                break
        out[sm] = s
    xb = x[~sm]
    if xb.size:
        acc = np.zeros_like(xb)
        for k in range(1, n_asym + 1):
            acc += ((-1.0) ** (k - 1)) * xb ** (-k) * rgamma(1.0 - a * k)
        out[~sm] = acc
    return np.clip(out, 0.0, 1.0)


def zener(t, tau, alpha, e0, einf):
    return einf + (e0 - einf) * ml_neg((t / tau) ** alpha, alpha)


def resid(p, t, y):
    log_tau, alpha, einf, log_dE = p
    return zener(t, np.exp(log_tau), alpha, einf + np.exp(log_dE), einf) - y


def fit(t, y, init):
    tau0, a0, e00, einf0 = init
    p0 = [np.log(tau0), a0, einf0, np.log(max(e00 - einf0, 1e-2))]
    lb = [np.log(1e-2), 0.05, 1.0, np.log(1e-2)]
    ub = [np.log(1e3), 0.95, 2e2, np.log(5e2)]
    r = least_squares(resid, p0, args=(t, y), bounds=(lb, ub), method="trf", max_nfev=20000)
    log_tau, alpha, einf, log_dE = r.x
    return np.array([np.exp(log_tau), alpha, einf + np.exp(log_dE), einf])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = np.logspace(-2, np.log10(20.0), 100)
    clean = zener(t, *TRUE)

    fit(t, clean, INITS[0])  # warm-up (scipy / numpy first-call overhead)

    times = []
    for seed in range(N_SEED):
        rng = np.random.default_rng(int(seed + 10000 * LVL))
        noisy = clean + LVL * np.abs(clean) * rng.standard_normal(clean.shape)
        for init in INITS:
            t0 = time.perf_counter()
            fit(t, noisy, init)
            times.append(time.perf_counter() - t0)

    mean_t = statistics.mean(times)
    print(f"Analytical Zener least-squares fit -- {len(times)} fits "
          f"(5 seeds x 3 inits) at {int(LVL * 100)}% noise")
    print(f"  mean per-fit wall time = {mean_t:.4f} s "
          f"(min {min(times):.4f}, max {max(times):.4f})")

    payload = {
        "method": "Analytical Zener least-squares (4 par)",
        "noise_level": LVL, "n_seeds": N_SEED, "n_inits": N_INIT,
        "n_fits": len(times), "mean_time_s": mean_t,
        "min_time_s": min(times), "max_time_s": max(times),
        "per_fit_times_s": times,
    }
    with open(OUT / "analytical_ls.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT / 'analytical_ls.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
