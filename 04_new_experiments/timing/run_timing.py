"""Wall-clock timing: analytical eval, L1 solver, PIKAN forward,
PEKAN inverse (Sec. 4.5 protocol: warm-up, cuda sync, mean of 3 seeds or median
of 9 calls). MLP rows are read from mlp_baselines/results. Writes hardware.txt.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import paths  # noqa: F401
from _lib.bench import Timer, save_hardware_report
from _lib.data import ZenerParams, make_time_grid, zener_exact
from utils.fractional import solve_fractional_zener_l1

from example_01_viscoelastic_forward_relaxation import Config as PikanConfig, train_pi_kan
from example_03_viscoelastic import (
    Config as PekanConfig, generate_experimental_data, train_physics_encoded_kan,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
EXP = HERE.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NT = 120


def median_time(fn, repeats=9):
    fn()  # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def time_analytical():
    p = ZenerParams()
    t = make_time_grid(NT, p.t_max, "linear").to(torch.float64)
    return median_time(lambda: zener_exact(t, p))


def time_l1():
    p = ZenerParams()
    t = make_time_grid(NT, p.t_max, "linear").to(torch.float64)
    eps = torch.full_like(t, p.epsilon_0)
    return median_time(lambda: solve_fractional_zener_l1(t, eps, p.alpha, p.tau, p.E0, p.E_inf))


def time_pikan(seeds=(0, 1, 2)):
    cfg = PikanConfig()
    t = torch.linspace(0.0, cfg.t_max, cfg.n_points, device=cfg.device)
    walls = []
    for s in seeds:
        torch.manual_seed(s)
        with Timer("pikan") as tim:
            train_pi_kan(cfg, t)
        walls.append(tim.record.wall_seconds)
    return walls


def time_pekan(seeds=(0, 1, 2)):
    # PEKAN's four-parameter Mittag-Leffler series is a launch-bound elementwise
    # loop, so it runs faster on the CPU than on the GPU; we time it on the CPU.
    cfg = PekanConfig()
    t, sigma_noisy, sigma_exact = generate_experimental_data(cfg, device="cpu")
    walls = []
    for s in seeds:
        torch.manual_seed(s)
        with Timer("pekan", sync_cuda=False) as tim:
            train_physics_encoded_kan(cfg, t, sigma_noisy, sigma_exact, "cpu")
        walls.append(tim.record.wall_seconds)
    return walls


def read_mean_wall(json_path, predicate):
    if not json_path.exists():
        return None
    rows = json.load(open(json_path, encoding="utf-8"))
    vals = [r["wall_s"] for r in rows if predicate(r)]
    return sum(vals) / len(vals) if vals else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = save_hardware_report(str(OUT / "hardware.txt"))
    print(f"hardware: {info.get('cpu_model')}, torch={info.get('torch')}, "
          f"cuda={info.get('cuda_available')}, gpu={info.get('gpu_models')}")

    analytical = time_analytical()
    l1 = time_l1()
    pikan = time_pikan()
    pekan = time_pekan()

    mlp_dir = EXP / "mlp_baselines" / "results"
    mlp_pinn_large = read_mean_wall(mlp_dir / "mlp_pinn_forward.json", lambda r: r["n_params"] >= 2000)
    blackbox_mlp = read_mean_wall(mlp_dir / "mlp_inverse.json", lambda r: r.get("method") == "PhysicsEncodedMLP")

    table = [
        ("Forward (relaxation)", "Analytical (Mittag-Leffler eval)", analytical, "median of 9"),
        ("Forward (relaxation)", "L1 recursive solver (CPU)", l1, "median of 9"),
        ("Forward (relaxation)", "MLP-PINN (large, 2209 par)", mlp_pinn_large, "mean of 3 seeds (mlp_baselines)"),
        ("Forward (relaxation)", "PIKAN (300 par)", statistics.mean(pikan), "mean of 3 seeds"),
        ("Inverse (2% noise)", "Black-box MLP (2211 par)", blackbox_mlp, "mean (mlp_baselines)"),
        ("Inverse (2% noise)", "PEKAN (4 par, CPU)", statistics.mean(pekan), "mean of 3 seeds (CPU)"),
    ]

    print(f"\n{'problem':<22}{'method':<34}{'time (s)':>10}")
    for prob, method, t_s, _ in table:
        print(f"{prob:<22}{method:<34}{('n/a' if t_s is None else f'{t_s:.4f}'):>10}")

    rows = [{"problem": p, "method": m, "time_s": t_s, "note": note} for p, m, t_s, note in table]
    payload = {"rows": rows, "pikan_seeds": pikan, "pekan_seeds": pekan,
               "analytical_median_s": analytical, "l1_median_s": l1, "Nt": NT}
    with open(OUT / "runtime.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(OUT / "runtime.csv", "w", encoding="utf-8") as f:
        f.write("problem,method,time_s,note\n")
        for p, m, t_s, note in table:
            f.write(f"{p},{m},{'' if t_s is None else f'{t_s:.6f}'},{note}\n")
    print(f"\nWrote {OUT / 'runtime.json'} and runtime.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
