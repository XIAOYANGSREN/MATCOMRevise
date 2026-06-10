"""PEKAN fit of the real stress-relaxation .xls data.

Model: E(t) = E_inf + (E0 - E_inf) * E_alpha(-(t/tau)^alpha). E0/E_inf are read
off the curve endpoints and fixed; (tau, alpha) are fitted by least squares.
Data: data/relaxation_SongchiData/*.xls, sheet 2, cols [Stress, RelaxModulus, Time].
"""
import os, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import xlrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = str(_ROOT / 'data' / 'relaxation_SongchiData')
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES_DIR, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.ml_robust import mittag_leffler_neg


def logbin(t, y, nb=40):
    """Median of y in log-spaced time bins. Display only: hides the load-cell
    quantisation staircase (~1.2e-3 MPa steps); the fit uses the raw data."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    lt = np.log10(t)
    edges = np.linspace(lt.min(), lt.max(), nb + 1)
    idx = np.clip(np.digitize(lt, edges) - 1, 0, nb - 1)
    tb, yb = [], []
    for b in range(nb):
        msk = idx == b
        if np.any(msk):
            tb.append(10 ** (0.5 * (edges[b] + edges[b + 1])))
            yb.append(float(np.median(y[msk])))
    return np.array(tb), np.array(yb)


def load_xls(path, n_sample=250):
    wb = xlrd.open_workbook(path)
    sh = wb.sheet_by_index(2)
    E = np.array(sh.col_values(1)[1:], dtype=np.float64)
    t = np.array(sh.col_values(2)[1:], dtype=np.float64)
    m = t > 0
    t, E = t[m], E[m]
    # log-uniform downsample for a balanced fit across decades
    lt = np.log10(t)
    grid = np.linspace(lt.min(), lt.max(), n_sample)
    ts = 10 ** grid
    Es = np.interp(grid, lt, E)
    return t, E, ts, Es


def model(p, t, E0, E_inf):
    log_tau, alpha = p
    x = (t / np.exp(log_tau)) ** alpha
    return E_inf + (E0 - E_inf) * mittag_leffler_neg(x, alpha)


def fit_file(path):
    t_all, E_all, ts, Es = load_xls(path)
    # E0/E_inf are the observable endpoints; a free 4-parameter fit is
    # ill-conditioned (kappa ~ 1e5), fixing the amplitudes makes alpha stable.
    E0 = float(E_all[0]); E_inf = float(E_all[-1])
    En = (E_all - E_inf) / (E0 - E_inf + 1e-12)
    tau_g = max(t_all[np.argmin(np.abs(En - 0.3679))], 10.0)
    p0 = [np.log(tau_g), 0.40]
    lb = [np.log(1.0), 0.05]
    ub = [np.log(1e5), 0.95]
    res = least_squares(lambda p: model(p, ts, E0, E_inf) - Es, p0, bounds=(lb, ub),
                        method="trf", loss="soft_l1", f_scale=0.02, max_nfev=20000)
    log_tau, alpha = res.x
    tau = np.exp(log_tau)
    pred = model(res.x, ts, E0, E_inf)
    ss_res = np.sum((pred - Es) ** 2)
    ss_tot = np.sum((Es - Es.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((pred - Es) ** 2))
    maxrel = np.max(np.abs(pred - Es) / Es) * 100
    return dict(file=os.path.basename(path), tau=tau, alpha=alpha, E0=E0, E_inf=E_inf,
                R2=r2, RMSE=rmse, maxrel=maxrel,
                t_all=t_all, E_all=E_all, ts=ts, Es=Es, pred=pred)


if __name__ == "__main__":
    targets = ["SC-E05F01-1-20251124.xls", "SC-E09F01-1-20251212.xls", "SC-E13F01-1-20251211.xls"]
    targets = [os.path.join(DATA_DIR, f) for f in targets if os.path.exists(os.path.join(DATA_DIR, f))]
    results = [fit_file(p) for p in targets]

    print(f"\n{'File':<26}{'tau(s)':>10}{'alpha':>8}{'E0(MPa)':>10}{'Einf(MPa)':>11}{'R^2':>10}{'RMSE':>10}{'maxRel%':>9}")
    for r in results:
        print(f"{r['file']:<26}{r['tau']:>10.2f}{r['alpha']:>8.4f}{r['E0']:>10.4f}"
              f"{r['E_inf']:>11.4f}{r['R2']:>10.6f}{r['RMSE']:>10.5f}{r['maxrel']:>9.2f}")

    js = [{k: float(v) if isinstance(v, (np.floating, float)) else v
           for k, v in r.items() if k not in ("t_all", "E_all", "ts", "Es", "pred")} for r in results]
    with open(os.path.join(RES_DIR, "fit_results.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    labels = ["E05", "E09", "E13"]
    for r, c, lab in zip(results, colors, labels):
        tb, Eb = logbin(r["t_all"], r["E_all"])
        ax[0].semilogx(tb, Eb, "o", ms=4, alpha=0.55, color=c, mec="none")
        ax[0].semilogx(r["ts"], r["pred"], "-", lw=1.8, color=c,
                       label=f"{lab}: $\\alpha$={r['alpha']:.3f}, $\\tau$={r['tau']:.0f}s")
    ax[0].set_xlabel("Time $t$ (s)"); ax[0].set_ylabel("Relaxation modulus $E(t)$ (MPa)")
    ax[0].set_title("PEKAN fit to real relaxation data"); ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(alpha=0.3)
    # normalized overlay
    for r, c, lab in zip(results, colors, labels):
        tb, Eb = logbin(r["t_all"], r["E_all"])
        En = (Eb - r["E_inf"]) / (r["E0"] - r["E_inf"])
        Pn = (r["pred"] - r["E_inf"]) / (r["E0"] - r["E_inf"])
        ax[1].semilogx(tb, En, "o", ms=4, alpha=0.55, color=c, mec="none")
        ax[1].semilogx(r["ts"], Pn, "-", lw=1.8, color=c, label=lab)
    ax[1].set_xlabel("Time $t$ (s)"); ax[1].set_ylabel("Normalized modulus")
    ax[1].set_title("Normalized relaxation"); ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RES_DIR, f"real_data_pekan_fit.{ext}"), dpi=300, bbox_inches="tight")
    fig_dir = _ROOT / 'figures_out'
    fig_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        # historical filename; printed as Figure 9 in the manuscript
        fig.savefig(str(fig_dir / f"Figure10_real_data.{ext}"), dpi=300, bbox_inches="tight")
    print(f"\nSaved: {os.path.join(RES_DIR, 'fit_results.json')}")
    print(f"Saved: {os.path.join(RES_DIR, 'real_data_pekan_fit.png')}")
