"""Noise sweep for the analytical Zener inversion.

Inverts the Example-3 relaxation for (tau, alpha, E0, Einf) at 0/1/2/5/10 %
noise, 10 realisations x 3 initial guesses each, plus a J^T J identifiability check.
"""
import os, json
from pathlib import Path
import numpy as np
from scipy.special import rgamma
from scipy.optimize import least_squares

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
_FIGS = Path(__file__).resolve().parents[2] / "figures_out"

# true Example-3 parameters
TAU, ALPHA, E0, EINF = 2.0, 0.5, 100.0, 20.0
TRUE = np.array([TAU, ALPHA, E0, EINF])
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10]
N_SEED, N_INIT = 10, 3
INITS = [  # (tau, alpha, E0, Einf)
    (0.5, 0.7, 60.0, 40.0),
    (5.0, 0.3, 150.0, 10.0),
    (1.0, 0.6, 80.0, 30.0),
]

def ml_neg(x, a, n_series=120, n_asym=8, x_split=5.0):
    # truncated series for x<=x_split, asymptotic expansion beyond
    x = np.asarray(x, float); out = np.empty_like(x)
    sm = x <= x_split; xs = x[sm]
    if xs.size:
        s = np.ones_like(xs); zp = np.ones_like(xs); z = -xs
        for k in range(1, n_series):
            zp = zp * z; term = zp * rgamma(a * k + 1.0); s = s + term
            if k > 8 and np.max(np.abs(term)) < 1e-16: break
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
    lb = [np.log(1e-2), 0.05, 1.0,   np.log(1e-2)]
    ub = [np.log(1e3),  0.95, 2e2,   np.log(5e2)]
    r = least_squares(resid, p0, args=(t, y), bounds=(lb, ub), method="trf", max_nfev=20000)
    log_tau, alpha, einf, log_dE = r.x
    return np.array([np.exp(log_tau), alpha, einf + np.exp(log_dE), einf])

# data grid as in Example 3: log-uniform t in [1e-2, 20], 100 pts
t = np.logspace(-2, np.log10(20.0), 100)
clean = zener(t, *TRUE)

summary = {}
boxdata = {}
for lvl in NOISE_LEVELS:
    relerrs = []
    for seed in range(N_SEED):
        rng = np.random.default_rng(int(seed + 10000 * lvl))
        noisy = clean + lvl * np.abs(clean) * rng.standard_normal(clean.shape)
        for init in INITS:
            est = fit(t, noisy, init)
            relerrs.append(np.abs(est - TRUE) / TRUE)
    relerrs = np.array(relerrs) * 100.0  # percent
    mean, std = relerrs.mean(0), relerrs.std(0)
    summary[f"{int(lvl*100)}%"] = {
        "tau":   [mean[0], std[0]], "alpha": [mean[1], std[1]],
        "E0":    [mean[2], std[2]], "Einf":  [mean[3], std[3]],
        "n_fits": relerrs.shape[0],
    }
    boxdata[f"{int(lvl*100)}%"] = relerrs[:, 1].tolist()  # alpha rel err

# identifiability: J^T J at the true params on the clean signal
def jac(p, eps=1e-6):
    J = np.zeros((len(t), 4))
    for i in range(4):
        dp = np.zeros(4); dp[i] = eps * max(abs(p[i]), 1e-3)
        J[:, i] = (zener(t, *(p + dp)) - zener(t, *(p - dp))) / (2 * dp[i])
    return J
J = jac(TRUE.astype(float))
JTJ = J.T @ J
cond = np.linalg.cond(JTJ)
Cov = np.linalg.inv(JTJ)
d = np.sqrt(np.diag(Cov))
corr = Cov / np.outer(d, d)
names = ["tau", "alpha", "E0", "Einf"]

print("\nNoise robustness  (mean +/- std of |rel. error| %, over 10 seeds x 3 inits)")
print(f"{'noise':>7}{'tau':>14}{'alpha':>14}{'E0':>14}{'Einf':>14}")
for k, v in summary.items():
    print(f"{k:>7}" + "".join(f"{v[p][0]:>7.2f}±{v[p][1]:<6.2f}" for p in ['tau','alpha','E0','Einf']))
print(f"\nIdentifiability (J^T J at true params):  cond(J^T J) = {cond:.3e}")
print("Parameter correlation matrix:")
print(f"{'':>8}" + "".join(f"{n:>9}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>8}" + "".join(f"{corr[i, j]:>9.3f}" for j in range(4)))

with open(os.path.join(OUT, "noise_sweep.json"), "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "cond_JTJ": float(cond),
               "corr": corr.tolist(), "names": names}, f, indent=2)

import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.boxplot(list(boxdata.values()), tick_labels=list(boxdata.keys()), showmeans=True)
ax.set_xlabel("Gaussian noise level"); ax.set_ylabel(r"$|\hat\alpha-\alpha|/\alpha$  (%)")
ax.set_title("PEKAN inversion of $\\alpha$ vs noise (10 noise realisations $\\times$ 3 initial guesses)")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"noise_boxplot.{ext}"), dpi=300, bbox_inches="tight")
_FIGS.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    # historical filename; printed as Figure 7 in the manuscript
    fig.savefig(str(_FIGS / f"Figure8_noise_robustness.{ext}"), dpi=300, bbox_inches="tight")
print(f"\nSaved: {os.path.join(OUT, 'noise_sweep.json')}  and  noise_boxplot.png")
