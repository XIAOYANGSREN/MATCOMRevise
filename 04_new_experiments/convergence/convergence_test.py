"""Torch cross-check of the L1 / L1-2 convergence study.

Same setup as convergence_numpy.py but through the torch quadratures in
_lib/schemes.py. Run: python convergence_test.py [--alpha 0.7]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.schemes import SCHEMES, caputo_exact_power

OUT_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
NS = [50, 100, 200, 400]


def run_one(scheme, p, alpha, T, n):
    dt = T / n
    t = torch.linspace(0.0, T, n + 1, dtype=torch.float64)
    du_num = SCHEMES[scheme](t ** p, dt, alpha)
    du_exact = caputo_exact_power(t, p, alpha)
    return float(torch.abs(du_num[-1] - du_exact[-1]).item())


def study(p, alpha, T, ns):
    rows = []
    for scheme in SCHEMES:
        prev = None
        for n in ns:
            err = run_one(scheme, p, alpha, T, n)
            order = (math.log(prev / err) / math.log(2.0)) if prev and err > 0 else float("nan")
            rows.append({"scheme": scheme, "n": n, "dt": T / n, "err": err, "order": order})
            prev = err
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("scheme,n,dt,err,order\n")
        for r in rows:
            f.write(f"{r['scheme']},{r['n']},{r['dt']:.6e},{r['err']:.6e},{r['order']:.4f}\n")


def make_plot(rows, path, alpha, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    for s in sorted({r["scheme"] for r in rows}):
        sub = [r for r in rows if r["scheme"] == s]
        ax.loglog([r["dt"] for r in sub], [r["err"] for r in sub], marker="o", label=s)
    dts = sorted({r["dt"] for r in rows})
    ref_x = np.array([dts[0], dts[-1]])
    ax.loglog(ref_x, (ref_x / ref_x[0]) ** (2 - alpha) * 1e-3, "k--", alpha=0.5, label=f"slope {2-alpha:.2f}")
    ax.loglog(ref_x, (ref_x / ref_x[0]) ** (3 - alpha) * 1e-6, "k:", alpha=0.5, label=f"slope {3-alpha:.2f}")
    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel(r"$|D^\alpha u(T) - D^\alpha_h u(T)|$")
    ax.set_title(rf"Convergence: $u=t^{{{p}}}$, $\alpha={alpha}$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--p", type=float, default=3.0)
    ap.add_argument("--T", type=float, default=1.0)
    args = ap.parse_args()

    print(f"Convergence study: u(t)=t^{args.p}, alpha={args.alpha}, T={args.T}, n={NS}")
    rows = study(args.p, args.alpha, args.T, NS)
    print(f"{'scheme':<8} {'n':>5} {'dt':>10} {'err':>12} {'order':>7}")
    for r in rows:
        print(f"{r['scheme']:<8} {r['n']:>5} {r['dt']:>10.4e} {r['err']:>12.4e} {r['order']:>7.3f}")

    csv_path = OUT_DIR / f"convergence_alpha{args.alpha}_p{args.p}.csv"
    write_csv(rows, csv_path)
    try:
        make_plot(rows, FIG_DIR / f"convergence_alpha{args.alpha}_p{args.p}.png", args.alpha, args.p)
        print(f"\nSaved {csv_path}")
    except Exception as e:
        print(f"\nSaved {csv_path}  (plot skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
