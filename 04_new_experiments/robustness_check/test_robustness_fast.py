"""Scipy-only version of the robustness battery: raw least_squares vs the PEKAN
reparametrisation vs raw with 4 random restarts, over inits x noise x seeds.
Writes results/robustness_fast.json.
"""
import os, json
import numpy as np
from scipy.special import rgamma
from scipy.optimize import least_squares

TRUE = np.array([2.0, 0.5, 100.0, 20.0])

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

def model(p, t):
    tau, a, E0, Ei = p
    return Ei + (E0 - Ei) * ml_neg((t / tau) ** a, a)

LB = [1e-2, 0.05, 1.0, 0.1]; UB = [1e3, 0.95, 5e2, 5e2]

def fit_raw(t, y, init):
    p0 = list(np.clip(init, LB, UB))
    try:
        r = least_squares(lambda p: model(p, t) - y, p0, bounds=(LB, UB), method='trf', max_nfev=4000)
        return np.array(r.x)
    except Exception:
        return np.array([np.nan] * 4)

def fit_reparam(t, y, init):
    tau0, a0, E00, Ei0 = init; a0 = float(np.clip(a0, 0.06, 0.94))
    v0 = [np.log(max(tau0, 1e-2)), np.log(a0 / (1 - a0)), np.log(max(Ei0, 0.1)), np.log(max(E00 - Ei0, 1e-2))]
    def res(v):
        tau = np.exp(v[0]); a = 1 / (1 + np.exp(-v[1])); Ei = np.exp(v[2]); E0 = Ei + np.exp(v[3])
        return model([tau, a, E0, Ei], t) - y
    try:
        r = least_squares(res, v0, method='lm', max_nfev=4000); v = r.x
        return np.array([np.exp(v[0]), 1 / (1 + np.exp(-v[1])), np.exp(v[2]) + np.exp(v[3]), np.exp(v[2])])
    except Exception:
        return np.array([np.nan] * 4)

def fit_raw_multistart(t, y, rng, n=4):
    best = None; bestloss = np.inf
    starts = [(rng.uniform(0.2, 10), rng.uniform(0.2, 0.8), rng.uniform(40, 200), rng.uniform(5, 60)) for _ in range(n)]
    for s in starts:
        p = fit_raw(t, y, s)
        if np.any(np.isnan(p)): continue
        loss = np.mean((model(p, t) - y) ** 2)
        if loss < bestloss: bestloss = loss; best = p
    return best if best is not None else np.array([np.nan] * 4)

INITS = [(1.0, 0.4, 80, 15), (0.5, 0.7, 60, 40), (5.0, 0.3, 150, 50), (0.2, 0.8, 40, 5),
         (20.0, 0.2, 200, 60), (0.1, 0.9, 300, 80), (10.0, 0.6, 50, 30), (3.0, 0.5, 120, 10),
         (0.3, 0.35, 200, 45), (8.0, 0.75, 70, 25)]
NOISES = [0.02, 0.05, 0.10]; SEEDS = [0, 1, 2]

t = np.logspace(-2, np.log10(20.0), 100)
clean = model(TRUE, t)

def maxrel(est):
    if est is None or np.any(np.isnan(est)): return np.inf
    return float(np.max(np.abs(est - TRUE) / TRUE))

out = {}
print(f"battery: {len(INITS)} inits x {len(NOISES)} noise x {len(SEEDS)} seeds\n")
for noise in NOISES:
    rec = {'classical-raw': [], 'reparam': [], 'classical-MS': []}
    for seed in SEEDS:
        rng = np.random.default_rng(1234 + seed + int(1000 * noise))
        y = clean + noise * np.abs(clean) * rng.standard_normal(clean.shape)
        for init in INITS:
            rec['classical-raw'].append(maxrel(fit_raw(t, y, init)))
            rec['reparam'].append(maxrel(fit_reparam(t, y, init)))
            rec['classical-MS'].append(maxrel(fit_raw_multistart(t, y, rng)))
    row = {}
    for k, v in rec.items():
        v = np.array(v)
        row[k] = {'basin<30%': round(float(np.mean(v < 0.30) * 100)),
                  'good<15%': round(float(np.mean(v < 0.15) * 100)),
                  'median_maxrel%': round(float(np.median(v[np.isfinite(v)]) * 100), 1) if np.any(np.isfinite(v)) else None}
    out[f'{int(noise*100)}%'] = row
    print(f"=== noise {int(noise*100)}%  ({len(INITS)*len(SEEDS)} runs/method) ===")
    for k in ['classical-raw', 'reparam', 'classical-MS']:
        r = row[k]
        print(f"  {k:14s}: correct-basin(<30%)={r['basin<30%']:4d}%   good(<15%)={r['good<15%']:4d}%   median maxrel={r['median_maxrel%']}%")
    print()

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'robustness_fast.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("saved results/robustness_fast.json")
