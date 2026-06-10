"""Model-mismatch benchmark: data from a 12-mode
Prony series, fitted with the 4-parameter fractional Zener model, so the
recovered alpha is an effective exponent. Checks full-range fits under noise
and late-time extrapolation from the first 70% of log-time.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.special import rgamma, roots_genlaguerre


OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

NOISE_LEVELS = [0.0, 0.02, 0.05]
N_SEEDS = 10
N_POINTS = 300
TRAIN_FRACTION_LOGTIME = 0.70
PRONY_BASELINES = [3, 5, 8, 12]


@dataclass(frozen=True)
class FitResult:
    split: str
    noise: float
    seed: int
    model: str
    n_params: int
    rmse_clean_all: float
    r2_clean_all: float
    rmse_observed_train: float
    aic_observed_train: float
    bic_observed_train: float
    rmse_clean_test: float | None
    relative_rmse_clean_test: float | None
    r2_clean_test: float | None
    tau_eff: float | None = None
    alpha_eff: float | None = None
    E0_eff: float | None = None
    Einf_eff: float | None = None


def mittag_leffler_neg(x, alpha, n_series=300, n_gl=96):
    """E_alpha(-x): Taylor series while well conditioned, Gauss-Laguerre form of
    the spectral integral beyond. Smooth over the fitted alpha range [0.3, 0.9]."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    x_split = min(4.0, max(1.5, 8.0 * alpha))
    small = x <= x_split
    xs = x[small]
    if xs.size:
        series = np.ones_like(xs)
        z_power = np.ones_like(xs)
        z = -xs
        for k in range(1, n_series):
            z_power = z_power * z
            term = z_power * rgamma(alpha * k + 1.0)
            series = series + term
            if k > 10 and float(np.max(np.abs(term))) < 1e-14:
                break
        out[small] = series

    xb = x[~small]
    if xb.size:
        nodes, weights = roots_genlaguerre(n_gl, alpha - 1.0)
        s = nodes[:, None]
        w = weights[:, None]
        inv_x = 1.0 / xb[None, :]
        s_alpha = s ** alpha
        denom = (s_alpha * inv_x) ** 2 + 2.0 * (s_alpha * inv_x) * math.cos(math.pi * alpha) + 1.0
        out[~small] = (
            math.sin(math.pi * alpha) / math.pi
            * np.sum(w * inv_x / denom, axis=0)
        )

    return np.clip(out, 0.0, 1.0)


def zener(t: np.ndarray, tau: float, alpha: float, e0: float, einf: float) -> np.ndarray:
    return einf + (e0 - einf) * mittag_leffler_neg((t / tau) ** alpha, alpha)


def make_prony_truth():
    """Return t, clean modulus, true Prony taus, true non-negative weights."""
    t = np.logspace(-2, 4, N_POINTS)
    taus = np.logspace(-2, 4, 12)
    log_tau = np.log10(taus)

    # broad positive relaxation spectrum
    weights = np.exp(-0.5 * ((log_tau - 1.0) / 1.7) ** 2)
    weights = 80.0 * weights / weights.sum()
    einf = 20.0
    clean = einf + np.exp(-t[:, None] / taus[None, :]) @ weights
    return t, clean, taus, weights


def add_noise(y: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    if noise_level <= 0:
        return y.copy()
    rng = np.random.default_rng(seed + int(round(noise_level * 10000)))
    return y * (1.0 + noise_level * rng.standard_normal(y.shape))


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rss = float(np.sum((y_pred - y_true) ** 2))
    rmse = math.sqrt(rss / len(y_true))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - rss / denom if denom > 0 else float("nan")
    return rmse, r2


def information_criteria(y_obs: np.ndarray, y_pred: np.ndarray, k: int) -> tuple[float, float, float]:
    n = len(y_obs)
    rss = float(np.sum((y_pred - y_obs) ** 2))
    rmse = math.sqrt(rss / n)
    aic = n * math.log(rss / n + 1e-300) + 2 * k
    bic = n * math.log(rss / n + 1e-300) + k * math.log(n)
    return rmse, aic, bic


def fit_zener(t_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    e0_guess = float(np.percentile(y_train, 98))
    einf_guess = float(np.percentile(y_train, 2))
    best = None

    initial_guesses = []
    for tau0 in (0.03, 0.3, 3.0, 10.0, 30.0, 300.0, 3000.0):
        for alpha0 in (0.35, 0.45, 0.50, 0.65):
            initial_guesses.append((tau0, alpha0, e0_guess, einf_guess))
    initial_guesses.extend([
        (10.0, 0.50, 100.0, 20.0),
        (10.0, 0.50, 95.0, 23.0),
        (30.0, 0.40, 90.0, 20.0),
        (100.0, 0.35, 85.0, 20.0),
    ])

    for tau0, alpha0, e0_init, einf_init in initial_guesses:
        p0 = [
            math.log(tau0),
            alpha0,
            einf_init,
            math.log(max(e0_init - einf_init, 1e-3)),
        ]
        lower = [math.log(1e-4), 0.30, 0.0, math.log(1e-3)]
        upper = [math.log(1e7), 0.95, 200.0, math.log(300.0)]

        def residual(p: np.ndarray) -> np.ndarray:
            log_tau, alpha, einf, log_delta = p
            e0 = einf + math.exp(log_delta)
            return zener(t_train, math.exp(log_tau), alpha, e0, einf) - y_train

        fit = least_squares(
            residual,
            p0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=8000,
        )
        if best is None or fit.cost < best.cost:
            best = fit

    log_tau, alpha, einf, log_delta = best.x
    e0 = einf + math.exp(log_delta)
    return np.array([math.exp(log_tau), alpha, e0, einf], dtype=float)


def prony_design(t: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    taus = np.logspace(-2, 4, n_modes)
    design = np.column_stack([np.ones_like(t), np.exp(-t[:, None] / taus[None, :])])
    return design, taus


def fit_prony_fixed_grid(t_train: np.ndarray, y_train: np.ndarray,
                         n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    design, taus = prony_design(t_train, n_modes)
    coef, _ = nnls(design, y_train)
    return coef, taus


def predict_prony(t: np.ndarray, coef: np.ndarray, taus: np.ndarray) -> np.ndarray:
    return coef[0] + np.exp(-t[:, None] / taus[None, :]) @ coef[1:]


def evaluate_one(split_name: str, train_mask: np.ndarray, noise: float, seed: int,
                 t: np.ndarray, clean: np.ndarray) -> list[FitResult]:
    observed = add_noise(clean, noise, seed)
    t_train = t[train_mask]
    y_train = observed[train_mask]
    clean_test = clean[~train_mask]
    has_test = clean_test.size > 0
    out: list[FitResult] = []

    p = fit_zener(t_train, y_train)
    pred_all = zener(t, *p)
    pred_train = pred_all[train_mask]
    rmse_all, r2_all = metric_pack(clean, pred_all)
    rmse_train_obs, aic_train, bic_train = information_criteria(y_train, pred_train, 4)
    if has_test:
        pred_test = pred_all[~train_mask]
        rmse_test, r2_test = metric_pack(clean_test, pred_test)
        rel_rmse_test = rmse_test / float(np.mean(clean_test))
    else:
        rmse_test = r2_test = rel_rmse_test = None
    out.append(FitResult(
        split=split_name,
        noise=noise,
        seed=seed,
        model="PEKAN/Zener",
        n_params=4,
        rmse_clean_all=rmse_all,
        r2_clean_all=r2_all,
        rmse_observed_train=rmse_train_obs,
        aic_observed_train=aic_train,
        bic_observed_train=bic_train,
        rmse_clean_test=rmse_test,
        relative_rmse_clean_test=rel_rmse_test,
        r2_clean_test=r2_test,
        tau_eff=float(p[0]),
        alpha_eff=float(p[1]),
        E0_eff=float(p[2]),
        Einf_eff=float(p[3]),
    ))

    for n_modes in PRONY_BASELINES:
        coef, taus = fit_prony_fixed_grid(t_train, y_train, n_modes)
        pred_all = predict_prony(t, coef, taus)
        pred_train = pred_all[train_mask]
        k = n_modes + 1
        rmse_all, r2_all = metric_pack(clean, pred_all)
        rmse_train_obs, aic_train, bic_train = information_criteria(y_train, pred_train, k)
        if has_test:
            pred_test = pred_all[~train_mask]
            rmse_test, r2_test = metric_pack(clean_test, pred_test)
            rel_rmse_test = rmse_test / float(np.mean(clean_test))
        else:
            rmse_test = r2_test = rel_rmse_test = None
        out.append(FitResult(
            split=split_name,
            noise=noise,
            seed=seed,
            model=f"Prony-{n_modes}",
            n_params=k,
            rmse_clean_all=rmse_all,
            r2_clean_all=r2_all,
            rmse_observed_train=rmse_train_obs,
            aic_observed_train=aic_train,
            bic_observed_train=bic_train,
            rmse_clean_test=rmse_test,
            relative_rmse_clean_test=rel_rmse_test,
            r2_clean_test=r2_test,
        ))
    return out


def aggregate(results: list[FitResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = sorted({(r.split, r.noise, r.model) for r in results})
    for split, noise, model in keys:
        subset = [r for r in results if r.split == split and r.noise == noise and r.model == model]
        n_params = subset[0].n_params

        def mean_std(attr: str) -> tuple[float | None, float | None]:
            vals = [getattr(r, attr) for r in subset]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if not vals:
                return None, None
            arr = np.asarray(vals, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=0))

        row: dict[str, object] = {
            "split": split,
            "noise_percent": int(round(noise * 100)),
            "model": model,
            "n_params": n_params,
        }
        for attr in (
            "rmse_clean_all",
            "r2_clean_all",
            "rmse_observed_train",
            "aic_observed_train",
            "bic_observed_train",
            "rmse_clean_test",
            "relative_rmse_clean_test",
            "r2_clean_test",
            "tau_eff",
            "alpha_eff",
            "E0_eff",
            "Einf_eff",
        ):
            mean, std = mean_std(attr)
            row[f"{attr}_mean"] = mean
            row[f"{attr}_std"] = std
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_example(t: np.ndarray, clean: np.ndarray, train_mask: np.ndarray) -> None:
    noise = 0.02
    seed = 3
    observed = add_noise(clean, noise, seed)
    fits: dict[str, np.ndarray] = {}

    p_full = fit_zener(t, observed)
    fits["PEKAN/Zener full"] = zener(t, *p_full)
    for n_modes in (5, 12):
        coef, taus = fit_prony_fixed_grid(t, observed, n_modes)
        fits[f"Prony-{n_modes} full"] = predict_prony(t, coef, taus)

    p_early = fit_zener(t[train_mask], observed[train_mask])
    fits["PEKAN/Zener early"] = zener(t, *p_early)
    for n_modes in (5, 12):
        coef, taus = fit_prony_fixed_grid(t[train_mask], observed[train_mask], n_modes)
        fits[f"Prony-{n_modes} early"] = predict_prony(t, coef, taus)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    colors = {
        "PEKAN/Zener full": "#1f77b4",
        "Prony-5 full": "#2ca02c",
        "Prony-12 full": "#d62728",
        "PEKAN/Zener early": "#1f77b4",
        "Prony-5 early": "#2ca02c",
        "Prony-12 early": "#d62728",
    }
    for ax, suffix, title in (
        (axes[0], "full", "Full-range fit (2% noisy Prony data)"),
        (axes[1], "early", "Late-time extrapolation from first 70% log-time"),
    ):
        ax.semilogx(t, clean, "k-", lw=2.0, label="clean 12-mode Prony truth")
        ax.semilogx(t, observed, ".", ms=3.0, alpha=0.30, color="0.45", label="noisy observations")
        for name, pred in fits.items():
            if name.endswith(suffix):
                ax.semilogx(t, pred, "-", lw=1.5, color=colors[name], label=name.replace(f" {suffix}", ""))
        if suffix == "early":
            ax.axvspan(t[train_mask][-1], t[-1], color="0.92", zorder=-2, label="held-out region")
        ax.set_xlabel("time $t$ (s)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("relaxation modulus $E(t)$")
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[1].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"prony_mismatch_example.{ext}", dpi=300, bbox_inches="tight")
    fig_dir = Path(__file__).resolve().parents[2] / "figures_out"
    fig_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        # historical filename; printed as Figure 11 in the manuscript
        fig.savefig(fig_dir / f"Figure13_prony_mismatch.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: list[dict[str, object]]) -> None:
    order = {"PEKAN/Zener": 0, "Prony-3": 1, "Prony-5": 2, "Prony-8": 3, "Prony-12": 4}
    for split in ("full", "early70"):
        print("\n" + "=" * 104)
        print(f"{split}: mean +/- std over {N_SEEDS} noise seeds")
        print("=" * 104)
        for noise in NOISE_LEVELS:
            rows = [
                r for r in summary
                if r["split"] == split and r["noise_percent"] == int(round(noise * 100))
            ]
            rows.sort(key=lambda r: order[str(r["model"])])
            print(f"\nNoise = {int(round(noise * 100))}%")
            if split == "full":
                print(f"{'model':<13}{'k':>4}{'RMSE clean':>20}{'R2 clean':>14}{'BIC train':>16}{'alpha':>16}")
                for r in rows:
                    alpha = r["alpha_eff_mean"]
                    alpha_s = r["alpha_eff_std"]
                    alpha_txt = "" if alpha is None else f"{alpha:.3f}+/-{alpha_s:.3f}"
                    print(
                        f"{str(r['model']):<13}{int(r['n_params']):>4}"
                        f"{r['rmse_clean_all_mean']:>10.3f}+/-{r['rmse_clean_all_std']:<7.3f}"
                        f"{r['r2_clean_all_mean']:>14.5f}"
                        f"{r['bic_observed_train_mean']:>16.1f}"
                        f"{alpha_txt:>16}"
                    )
            else:
                print(f"{'model':<13}{'k':>4}{'RMSE all':>18}{'RMSE test':>18}{'rel test':>14}{'alpha':>16}")
                for r in rows:
                    alpha = r["alpha_eff_mean"]
                    alpha_s = r["alpha_eff_std"]
                    alpha_txt = "" if alpha is None else f"{alpha:.3f}+/-{alpha_s:.3f}"
                    rel = r["relative_rmse_clean_test_mean"]
                    print(
                        f"{str(r['model']):<13}{int(r['n_params']):>4}"
                        f"{r['rmse_clean_all_mean']:>9.3f}+/-{r['rmse_clean_all_std']:<6.3f}"
                        f"{r['rmse_clean_test_mean']:>9.3f}+/-{r['rmse_clean_test_std']:<6.3f}"
                        f"{100.0 * rel:>13.2f}%"
                        f"{alpha_txt:>16}"
                    )


def main() -> None:
    t, clean, true_taus, true_weights = make_prony_truth()
    train_end = int(round(TRAIN_FRACTION_LOGTIME * len(t)))
    masks = {
        "full": np.ones_like(t, dtype=bool),
        "early70": np.arange(len(t)) < train_end,
    }

    results: list[FitResult] = []
    for split_name, mask in masks.items():
        for noise in NOISE_LEVELS:
            for seed in range(N_SEEDS):
                results.extend(evaluate_one(split_name, mask, noise, seed, t, clean))

    raw_rows = [r.__dict__ for r in results]
    summary = aggregate(results)
    write_csv(OUT / "prony_mismatch_raw.csv", raw_rows)
    write_csv(OUT / "prony_mismatch_summary.csv", summary)
    with (OUT / "prony_mismatch.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "design": {
                    "generator": "12-mode Prony series",
                    "noise_levels": NOISE_LEVELS,
                    "n_seeds": N_SEEDS,
                    "n_points": N_POINTS,
                    "train_fraction_logtime": TRAIN_FRACTION_LOGTIME,
                    "true_taus": true_taus.tolist(),
                    "true_weights": true_weights.tolist(),
                    "true_Einf": 20.0,
                },
                "summary": summary,
            },
            f,
            indent=2,
        )
    plot_example(t, clean, masks["early70"])
    print_summary(summary)
    print("\nSaved results under: 04_new_experiments/prony_mismatch/results")


if __name__ == "__main__":
    main()
