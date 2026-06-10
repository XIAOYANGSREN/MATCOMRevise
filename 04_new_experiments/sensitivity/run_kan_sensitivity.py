"""PIKAN hyper-parameter sensitivity on Example 1.

Overrides one hyper-parameter at a time (grid G, width H, or N_t) and records
the RMSE against the analytical solution; one CSV per sweep.
Run: python run_kan_sensitivity.py --sweep {grid,width,n_points} [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import paths  # noqa: F401
from _lib.bench import Timer

OUT_DIR = Path(__file__).resolve().parent / "results"


def run_pikan_with(*, grid, width_hidden, n_points, steps, seed):
    """Run PIKAN once with the example_01 Config plus overrides."""
    from src.example_01_viscoelastic_forward_relaxation import Config, train_pi_kan

    torch.manual_seed(seed)
    cfg = Config()
    cfg.n_points = n_points
    cfg.grid = grid
    cfg.width = [1, width_hidden, 1]
    cfg.steps = steps
    cfg.lbfgs_steps = max(50, steps // 6)

    t = torch.linspace(0.0, cfg.t_max, cfg.n_points, device=cfg.device)
    with Timer("pikan_train") as tim:
        _, errs, _, _, _ = train_pi_kan(cfg, t)
    return {
        "grid": grid, "width": width_hidden, "n_points": n_points,
        "steps": steps, "seed": seed,
        "rmse": errs["rmse"], "mae": errs["mae"], "r2": errs["r2"],
        "wall_s": tim.record.wall_seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 values x 1 seed x few steps; for verification only")
    ap.add_argument("--sweep", choices=["grid", "width", "n_points"], default="grid")
    args = ap.parse_args()

    if args.smoke:
        values = {"grid": [5, 10], "width": [5, 10], "n_points": [60, 100]}[args.sweep]
        seeds = [0]
        steps = 200
    else:
        values = {"grid": [3, 5, 7, 10, 15], "width": [5, 10, 20, 40],
                  "n_points": [50, 100, 200, 500]}[args.sweep]
        seeds = [0, 1, 2]
        steps = 3000

    base = {"grid": 12, "width_hidden": 10, "n_points": 120, "steps": steps}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in values:
        for seed in seeds:
            kwargs = dict(base)
            kwargs[args.sweep if args.sweep != "width" else "width_hidden"] = v
            kwargs["seed"] = seed
            try:
                row = run_pikan_with(**kwargs)
            except Exception as e:
                print(f"  [skip] {args.sweep}={v} seed={seed} failed: {e}")
                continue
            row["sweep"] = args.sweep
            rows.append(row)
            print(f"  {args.sweep}={v} seed={seed} -> "
                  f"rmse={row['rmse']:.3e}  wall={row['wall_s']:.1f}s")

    csv_path = OUT_DIR / f"kan_sensitivity_{args.sweep}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {csv_path}")
    else:
        print("\n(no rows written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
