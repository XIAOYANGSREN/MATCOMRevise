"""MLP baselines for the inverse problem on noisy Example-3 data.

PhysicsEncodedMLP is the ~2200-parameter black-box baseline; InverseMLP is the
4-parameter analytical fit used as a sanity check. Use --smoke for a quick run.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import paths  # noqa: F401
from _lib.bench import Timer
from _lib.data import ZenerParams, make_relaxation_dataset
from _lib.models import InverseMLP, PhysicsEncodedMLP

OUT_DIR = Path(__file__).resolve().parent / "results"


def fit_inverse_mlp(t, sigma_noisy, init, *, n_steps=800, lr=0.05):
    m = InverseMLP(
        tau_init=init["tau"], alpha_init=init["alpha"],
        E0_init=init["E0"], E_inf_init=init["E_inf"],
    )
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    with Timer("fit") as tim:
        for _ in range(n_steps):
            opt.zero_grad()
            loss = torch.mean((m(t) - sigma_noisy) ** 2)
            loss.backward()
            opt.step()
        opt2 = torch.optim.LBFGS(m.parameters(), lr=0.3, max_iter=50,
                                 line_search_fn="strong_wolfe")
        def closure():
            opt2.zero_grad()
            l = torch.mean((m(t) - sigma_noisy) ** 2)
            l.backward()
            return l
        opt2.step(closure)
        with torch.no_grad():
            mse = float(torch.mean((m(t) - sigma_noisy) ** 2).item())
    n_params = sum(p.numel() for p in m.parameters())
    return m.get_physical_params(), mse, tim.record.wall_seconds, n_params


def fit_pemlp(t, sigma_noisy, *, layers=(1, 32, 32, 32, 1), n_steps=800, lr=1e-3):
    m = PhysicsEncodedMLP(layers=layers)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    with Timer("fit") as tim:
        for _ in range(n_steps):
            opt.zero_grad()
            loss = torch.mean((m(t) - sigma_noisy) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            mse = float(torch.mean((m(t) - sigma_noisy) ** 2).item())
    return {
        "method": "PhysicsEncodedMLP",
        "mse": mse,
        "wall_s": tim.record.wall_seconds,
        "n_params": sum(p.numel() for p in m.parameters()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = [0, 1] if args.smoke else list(range(5))
    n_steps = 200 if args.smoke else 800

    true = ZenerParams()
    inits = [
        {"tau": 1.0, "alpha": 0.4, "E0": 80.0,  "E_inf": 15.0},
        {"tau": 0.5, "alpha": 0.7, "E0": 60.0,  "E_inf": 30.0},
        {"tau": 5.0, "alpha": 0.3, "E0": 150.0, "E_inf": 50.0},
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for j, init in enumerate(inits):
            t, _, sig_noisy = make_relaxation_dataset(
                true, n_points=100, spacing="log",
                noise_level=0.02, seed=seed,
            )
            est, mse, wall, n_params = fit_inverse_mlp(t, sig_noisy, init, n_steps=n_steps)
            row = {
                "method": "InverseMLP",
                "seed": seed, "init_id": j,
                "n_params": n_params,
                "mse": mse, "wall_s": wall,
                **est,
                "tau_err":   abs(est["tau"]   - true.tau)   / true.tau,
                "alpha_err": abs(est["alpha"] - true.alpha) / true.alpha,
                "E0_err":    abs(est["E0"]    - true.E0)    / true.E0,
                "Einf_err":  abs(est["E_inf"] - true.E_inf) / true.E_inf,
            }
            rows.append(row)
            print(f"  seed={seed} init={j} alpha_hat={est['alpha']:.4f} mse={mse:.3e}")

        # also fit a pure black-box MLP once per seed
        t, _, sig_noisy = make_relaxation_dataset(
            true, n_points=100, spacing="log", noise_level=0.02, seed=seed,
        )
        bb = fit_pemlp(t, sig_noisy, n_steps=n_steps)
        bb.update({"seed": seed, "init_id": -1})
        rows.append(bb)

    out_json = OUT_DIR / "mlp_inverse.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
