"""Basin-of-attraction check for the inverse fractional-Zener problem (Example 3
setting). Three fits of the same data: raw least_squares, PEKAN-reparametrised
least_squares (log tau, logit a, log Einf, log dE), and the torch InverseMLP.
"""
import os, sys, json
from pathlib import Path
import numpy as np
from scipy.special import rgamma
from scipy.optimize import least_squares

base = str(Path(__file__).resolve().parents[2])  # package root
sys.path.insert(0, os.path.join(base, '04_new_experiments'))
for p in ['03_code/src', '03_code', 'pykan']:
    sys.path.insert(0, os.path.join(base, *p.split('/')))
import torch
from _lib.models import InverseMLP
from _lib.data import ZenerParams, make_relaxation_dataset

TRUE = np.array([2.0, 0.5, 100.0, 20.0])
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def ml_neg(x, a, n_series=120, n_asym=8, x_split=5.0):
    x = np.asarray(x, float); out = np.empty_like(x); sm = x <= x_split; xs = x[sm]
    if xs.size:
        s = np.ones_like(xs); zp = np.ones_like(xs); z = -xs
        for k in range(1, n_series):
            zp = zp * z; t = zp * rgamma(a * k + 1.0); s = s + t
            if k > 8 and np.max(np.abs(t)) < 1e-16: break
        out[sm] = s
    xb = x[~sm]
    if xb.size:
        acc = np.zeros_like(xb)
        for k in range(1, n_asym + 1): acc += ((-1.) ** (k - 1)) * xb ** (-k) * rgamma(1. - a * k)
        out[~sm] = acc
    return np.clip(out, 0, 1)

def model_raw(p, t):
    tau, a, E0, Ei = p
    return Ei + (E0 - Ei) * ml_neg((t / tau) ** a, a)

def fit_classical(t, y, init):
    lb = [1e-2, 0.05, 1.0, 0.1]; ub = [1e3, 0.95, 5e2, 5e2]
    p0 = list(np.clip(init, lb, ub))
    try:
        r = least_squares(lambda p: model_raw(p, t) - y, p0, bounds=(lb, ub), method='trf', max_nfev=5000)
        return np.array(r.x)
    except Exception:
        return np.array([np.nan] * 4)

def fit_reparam(t, y, init):
    tau0, a0, E00, Ei0 = init; a0 = float(np.clip(a0, 0.06, 0.94))
    v0 = [np.log(max(tau0, 1e-2)), np.log(a0 / (1 - a0)), np.log(max(Ei0, 0.1)), np.log(max(E00 - Ei0, 1e-2))]
    def res(v):
        tau = np.exp(v[0]); a = 1 / (1 + np.exp(-v[1])); Ei = np.exp(v[2]); E0 = Ei + np.exp(v[3])
        return model_raw([tau, a, E0, Ei], t) - y
    try:
        r = least_squares(res, v0, method='lm', max_nfev=5000); v = r.x
        return np.array([np.exp(v[0]), 1 / (1 + np.exp(-v[1])), np.exp(v[2]) + np.exp(v[3]), np.exp(v[2])])
    except Exception:
        return np.array([np.nan] * 4)

def fit_pekan(t, y, init):
    tt = torch.tensor(t, dtype=torch.float32, device=DEV)
    yy = torch.tensor(y, dtype=torch.float32, device=DEV)
    m = InverseMLP(tau_init=float(init[0]), alpha_init=float(np.clip(init[1], 0.05, 0.95)),
                   E0_init=float(init[2]), E_inf_init=float(init[3])).to(DEV)
    for lr, n in [(0.05, 600), (0.005, 600)]:
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for _ in range(n):
            opt.zero_grad(); l = torch.mean((m(tt) - yy) ** 2)
            if torch.isnan(l): return np.array([np.nan] * 4)
            l.backward(); opt.step()
    opt2 = torch.optim.LBFGS(m.parameters(), lr=0.3, max_iter=50, line_search_fn='strong_wolfe')
    def cl():
        opt2.zero_grad(); l = torch.mean((m(tt) - yy) ** 2)
        if not torch.isnan(l): l.backward()
        return l
    try: opt2.step(cl)
    except Exception: pass
    pp = m.get_physical_params()
    return np.array([pp['tau'], pp['alpha'], pp['E0'], pp['E_inf']])

INITS = [(1.0, 0.4, 80, 15), (0.5, 0.7, 60, 40), (5.0, 0.3, 150, 50), (0.2, 0.8, 40, 5),
         (20.0, 0.2, 200, 60), (0.1, 0.9, 300, 80), (10.0, 0.6, 50, 30), (3.0, 0.5, 120, 10),
         (0.3, 0.35, 200, 45), (8.0, 0.75, 70, 25)]
NOISES = [0.02, 0.05, 0.10]; SEEDS = [0, 1, 2]

def maxrel(est):
    if est is None or np.any(np.isnan(est)): return np.inf
    return float(np.max(np.abs(est - TRUE) / TRUE))

true = ZenerParams()
out = {}
print(f"battery: {len(INITS)} inits x {len(NOISES)} noise x {len(SEEDS)} seeds = "
      f"{len(INITS)*len(NOISES)*len(SEEDS)} cases/method\n")
for noise in NOISES:
    rec = {'classical': [], 'reparam': [], 'pekan': []}
    for seed in SEEDS:
        t, _, y = make_relaxation_dataset(true, n_points=100, spacing='log', noise_level=noise, seed=seed)
        t = t.cpu().numpy().astype(float); y = y.cpu().numpy().astype(float)
        for init in INITS:
            rec['classical'].append(maxrel(fit_classical(t, y, init)))
            rec['reparam'].append(maxrel(fit_reparam(t, y, init)))
            rec['pekan'].append(maxrel(fit_pekan(t, y, init)))
    row = {}
    for k, v in rec.items():
        v = np.array(v)
        row[k] = {'found_basin_%': float(np.mean(v < 0.30) * 100),
                  'good_%': float(np.mean(v < 0.15) * 100),
                  'median_maxrel_%': float(np.median(v[np.isfinite(v)]) * 100) if np.any(np.isfinite(v)) else None,
                  'nan_%': float(np.mean(~np.isfinite(v)) * 100)}
    out[f'{int(noise*100)}%'] = row
    print(f"=== noise {int(noise*100)}% ===")
    for k in ['classical', 'reparam', 'pekan']:
        r = row[k]
        print(f"  {k:10s}: found-correct-basin(<30%)={r['found_basin_%']:5.0f}%  "
              f"good(<15%)={r['good_%']:5.0f}%  median maxrel={r['median_maxrel_%']:.1f}%  fail/NaN={r['nan_%']:.0f}%")
    print()

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'robustness.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("saved results/robustness.json")
