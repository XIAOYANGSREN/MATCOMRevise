"""MLP-PINN forward baselines, same L1-residual loss and schedule as PIKAN.

Two tanh-MLPs: small [1,16,16,1] (~321 params, matched to PIKAN's 300) and
large [1,32,32,32,1] (~2209 params). Use --smoke for a quick run.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import paths  # noqa: F401
from _lib.bench import Timer
from _lib.data import ZenerParams, make_time_grid, zener_exact
from _lib.models import MLPPINN
from utils.fractional import caputo_derivative_l1_series_fast, caputo_l1_kernel
from utils.metrics import compute_errors

OUT_DIR = Path(__file__).resolve().parent / "results"


def train(p: ZenerParams, *, n_points, layers, adam_steps, lbfgs_steps, seed, device):
    torch.manual_seed(seed)
    t = make_time_grid(n_points, p.t_max, "linear", device=device)
    t_in = (2.0 * (t / p.t_max) - 1.0).view(-1, 1)  # rescale to [-1,1] like example_01

    sigma_exact = zener_exact(t, p)
    eps_full = torch.full_like(t, p.epsilon_0)
    dt = (t[1] - t[0]).item()
    w_l1 = caputo_l1_kernel(nt=n_points, alpha=p.alpha, device=t.device, dtype=torch.float64)

    model = MLPPINN(layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn():
        sigma = model(t_in).view(-1)
        d_sigma = caputo_derivative_l1_series_fast(sigma, dt=dt, alpha=p.alpha, w=w_l1)
        residual = sigma + (p.tau ** p.alpha) * d_sigma - p.E_inf * eps_full
        ic = sigma[0] - p.E0 * p.epsilon_0
        return torch.mean(residual[1:] ** 2) + 10.0 * ic ** 2

    with Timer("train") as tim:
        for _ in range(adam_steps):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()
        if lbfgs_steps > 0:
            opt2 = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=1,
                                     line_search_fn="strong_wolfe")
            for _ in range(lbfgs_steps):
                def closure():
                    opt2.zero_grad()
                    l = loss_fn()
                    l.backward()
                    return l
                opt2.step(closure)

    with torch.no_grad():
        sigma_pred = model(t_in).view(-1)
    errs = compute_errors(sigma_pred, sigma_exact)
    return {
        "method": "MLP-PINN",
        "size": "small" if model.n_params < 1000 else "large",
        "layers": layers,
        "n_params": model.n_params,
        "seed": seed,
        "n_points": n_points,
        "rmse": float(errs["rmse"]),
        "mae": float(errs["mae"]),
        "r2": float(errs["r2"]),
        "max_error": float(errs["max_error"]),
        "wall_s": tim.record.wall_seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        architectures = [[1, 16, 16, 1]]
        adam_steps, lbfgs_steps, n_points = 200, 0, 60
        seeds = list(range(min(args.seeds, 2)))
    else:
        architectures = [[1, 16, 16, 1], [1, 32, 32, 32, 1]]
        adam_steps, lbfgs_steps, n_points = 3000, 500, 120
        seeds = list(range(args.seeds))

    p = ZenerParams()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for layers in architectures:
        for s in seeds:
            row = train(p, n_points=n_points, layers=layers, adam_steps=adam_steps,
                        lbfgs_steps=lbfgs_steps, seed=s, device=device)
            rows.append(row)
            print(f"  {row['size']:<5} {str(layers):<18} seed={s} "
                  f"rmse={row['rmse']:.3e} r2={row['r2']:.5f} "
                  f"params={row['n_params']} wall={row['wall_s']:.1f}s")

    out_json = OUT_DIR / "mlp_pinn_forward.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
