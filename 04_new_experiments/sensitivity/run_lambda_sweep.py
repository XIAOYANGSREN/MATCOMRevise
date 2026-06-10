"""Penalty-weight (lambda = ic_weight) row of the sensitivity study:
lambda in {1, 5, 10, 50, 100}, 3 seeds, same protocol as run_kan_sensitivity.py.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import paths  # noqa: F401

OUT_DIR = Path(__file__).resolve().parent / "results"


def run_one(lam, seed, steps):
    from src.example_01_viscoelastic_forward_relaxation import Config, train_pi_kan

    torch.manual_seed(seed)
    cfg = Config()
    cfg.lambda_ic = float(lam)
    cfg.steps = steps
    cfg.lbfgs_steps = max(50, steps // 6)

    t = torch.linspace(0.0, cfg.t_max, cfg.n_points, device=cfg.device)
    _, errs, _, _, _ = train_pi_kan(cfg, t)
    return {
        "sweep": "lambda", "lambda_ic": lam, "seed": seed,
        "n_points": cfg.n_points, "grid": cfg.grid, "width": cfg.width[1],
        "steps": steps, "rmse": float(errs["rmse"]), "r2": float(errs["r2"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 lambdas x 1 seed x few steps; for verification only")
    args = ap.parse_args()

    lams = [1, 5, 10, 50, 100]
    seeds = [0, 1, 2]
    steps = 3000
    if args.smoke:
        lams, seeds, steps = [1, 10], [0], 200

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for lam in lams:
        for seed in seeds:
            row = run_one(lam, seed, steps)
            rows.append(row)
            print(f"  lambda={lam:<4} seed={seed} -> rmse={row['rmse']:.4e}  r2={row['r2']:.4f}",
                  flush=True)

    csv_path = OUT_DIR / "kan_sensitivity_lambda.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nlambda   mean_rmse(Pa)   std")
    for lam in lams:
        rs = [r["rmse"] for r in rows if r["lambda_ic"] == lam]
        sd = statistics.pstdev(rs) if len(rs) > 1 else 0.0
        print(f"  {lam:<6} {statistics.mean(rs):.4f}        {sd:.4f}")
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
