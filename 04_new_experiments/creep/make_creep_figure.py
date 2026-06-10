"""Fractional-Zener fit of the 10000 s creep tests, 5 specimens."""
import xlrd, numpy as np, glob, os, json, sys
from pathlib import Path
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.ml_robust import ml_neg


def model(p, t, B):
    a, tau, t0 = p; tt = np.clip(t - t0, 0, None); return B * (1 - ml_neg((tt / tau) ** a, a))

def fit(t, e):
    # B (asymptotic creep strain) fixed at 1.30 * max strain over 10000 s; a free
    # 4-parameter fit is ill-conditioned and gives an unstable alpha.
    B = 1.30 * float(e.max())
    r = least_squares(lambda p: model(p, t, B) - e, [0.4, 300., 5.],
                      bounds=([0.05, 1., 0.], [0.95, 1e6, 40.]), method='trf', max_nfev=20000)
    pred = model(r.x, t, B); r2 = 1 - np.sum((pred - e) ** 2) / np.sum((e - e.mean()) ** 2)
    a, tau, t0 = r.x
    return (B, a, tau, t0), r2, pred

def logbin(t, y, nb=40):
    # display only: removes the displacement-sensor quantisation staircase
    t = np.asarray(t, float); y = np.asarray(y, float)
    lt = np.log10(t); edges = np.linspace(lt.min(), lt.max(), nb + 1)
    idx = np.clip(np.digitize(lt, edges) - 1, 0, nb - 1)
    tb, yb = [], []
    for b in range(nb):
        msk = idx == b
        if np.any(msk):
            tb.append(10 ** (0.5 * (edges[b] + edges[b + 1]))); yb.append(float(np.median(y[msk])))
    return np.array(tb), np.array(yb)

_ROOT = Path(__file__).resolve().parents[2]
d = str(_ROOT / 'data' / 'creep_10000s')
figdir = str(_ROOT / 'figures_out')
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(figdir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)
L0 = {'RB10000-1': 69.9, 'RB10000-2': 70.0, 'RB10000-3': 71.2, 'RB10000-4': 71.0, 'RB10000-5': 71.0}

plt.figure(figsize=(6.4, 4.4)); cols = plt.cm.viridis(np.linspace(0, .82, 5)); rows = []
for i, f in enumerate(sorted(glob.glob(os.path.join(d, 'RB10000-*.xls')))):
    key = os.path.basename(f)[:9]
    sh = xlrd.open_workbook(f).sheet_by_index(0)
    disp = np.array(sh.col_values(2)[1:], float); tim = np.array(sh.col_values(8)[1:], float)
    m = np.isfinite(disp) & np.isfinite(tim) & (tim > 0.2); disp, tim = disp[m], tim[m]
    eps = disp / L0[key]
    lt = np.log10(tim); g = np.linspace(lt.min(), lt.max(), 300); ts = 10 ** g; es = np.interp(g, lt, eps)
    x, r2, pred = fit(ts, es); B, a, tau, t0 = x
    rows.append(dict(spec=key, alpha=round(float(a), 3), tau=round(float(tau), 1),
                     t0=round(float(t0), 2), R2=round(float(r2), 3)))
    tb, eb = logbin(tim, eps)
    plt.semilogx(tb, eb, 'o', ms=3.2, alpha=.55, color=cols[i], mec='none')
    plt.semilogx(ts, pred, '-', lw=1.6, color=cols[i], label=f"{key.replace('RB10000-', 'RB-')}: $\\alpha={a:.3f}$")

plt.xlabel('time $t$ (s)'); plt.ylabel(r'creep strain $\varepsilon(t)$'); plt.grid(alpha=.3)
plt.title('PEKAN fit to real creep data (5 specimens)'); plt.legend(fontsize=7, frameon=False)
plt.tight_layout()
# historical filename; printed as Figure 10 in the manuscript
for e in ('png', 'pdf'): plt.savefig(os.path.join(figdir, f'Figure11_creep.{e}'), dpi=300, bbox_inches='tight')
json.dump(rows, open(os.path.join(results_dir, 'creep_fit.json'), 'w'), indent=2)
al = [r['alpha'] for r in rows]
print('per-specimen:', rows)
print(f'alpha all 5: {np.mean(al):.3f}+/-{np.std(al):.3f} | excl RB4: {np.mean([al[0],al[1],al[2],al[4]]):.3f}+/-{np.std([al[0],al[1],al[2],al[4]]):.3f}')
print('saved Figure11_creep.png')
