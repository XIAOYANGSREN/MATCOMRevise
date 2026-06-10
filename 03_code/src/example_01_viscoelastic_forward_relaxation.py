"""Example 01: PIKAN forward solve of fractional Zener stress relaxation.

sigma + tau^a D^a sigma = E_inf eps0 (step strain), sigma(0)=E0 eps0; the
analytical Mittag-Leffler solution is the benchmark.
"""

import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_SRC, "..", ".."))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_PKG, "pykan"))
_RESULTS_ROOT = os.path.join(_PKG, "results_paper")

import numpy as np
import torch
from tqdm import tqdm

from kan import KAN
from utils.plotting import NaturePlotStyle
from utils.metrics import compute_errors, print_error_table
from utils.fractional import caputo_derivative_l1_series_fast, caputo_l1_kernel, exact_solution_zener
from models.pikan import zener_relaxation_loss


class Config:
    # physical parameters (given; forward problem)
    tau = 2.0
    alpha = 0.5
    E0 = 100.0
    E_inf = 20.0
    epsilon0 = 1.0

    # time grid; the PI loss carries an O(N^2) history term, keep N moderate
    t_max = 20.0
    n_points = 120

    # PIKAN (KAN approximating sigma(t))
    width = [1, 10, 1]
    grid = 12
    k = 3
    steps = 3000
    lr = 0.01
    lbfgs_steps = 500
    lbfgs_lr = 0.5

    # initial-condition penalty weight (ic_weight in zener_relaxation_loss)
    lambda_ic = 10.0

    output_dir = os.path.join(_RESULTS_ROOT, "example_01")
    figure_dir = os.path.join(_RESULTS_ROOT, "example_01", "figures")
    data_dir = os.path.join(_RESULTS_ROOT, "example_01", "data")

    device = "cuda" if torch.cuda.is_available() else "cpu"


def build_step_strain(t, epsilon0):
    # constant strain for t>0; the t=0 initial value is penalised separately
    return torch.full_like(t, float(epsilon0))


def train_pi_kan(config, t):
    """Train the KAN on the physics residual; returns model, errors, losses, pred, exact."""
    device = torch.device(config.device)
    t = t.to(device)
    # map time to [-1,1], the default B-spline grid range
    t_in = (2.0 * (t / float(config.t_max)) - 1.0).view(-1, 1)

    model = KAN(
        width=config.width,
        grid=config.grid,
        k=config.k,
        grid_eps=1.0,
        noise_scale=0.1,
        device=device,
    )
    try:
        model.speed()  # physics-only solve, skip the symbolic branch
    except Exception:
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    eps = build_step_strain(t, config.epsilon0)

    with torch.no_grad():
        sigma_exact = exact_solution_zener(t, config.tau, config.alpha, config.E0, config.E_inf, config.epsilon0)

    losses = []
    dt = (t[1] - t[0]).item()
    # precompute the L1 kernel once
    w_l1 = caputo_l1_kernel(nt=int(t.shape[0]), alpha=config.alpha, device=device, dtype=torch.float64)
    pbar = tqdm(range(config.steps), desc="PIKAN example_01")

    def loss_fn():
        sigma_pred_local = model(t_in).view(-1)
        return zener_relaxation_loss(
            sigma_pred_local, eps, dt, config.alpha, config.tau,
            config.E0, config.E_inf, config.epsilon0, w_l1,
            ic_weight=getattr(config, "lambda_ic", 10.0),
        )

    for step in pbar:
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if step % 50 == 0:
            with torch.no_grad():
                sigma_tmp = model(t_in).view(-1)
                ic_tmp = sigma_tmp[0] - (config.E0 * config.epsilon0)
            pbar.set_postfix({"loss": f"{loss.item():.2e}", "ic": f"{ic_tmp.item():.2e}"})

    if getattr(config, "lbfgs_steps", 0) and config.lbfgs_steps > 0:
        optimizer2 = torch.optim.LBFGS(model.parameters(), lr=config.lbfgs_lr, max_iter=1, line_search_fn="strong_wolfe")
        pbar2 = tqdm(range(config.lbfgs_steps), desc="PIKAN LBFGS")
        for _ in pbar2:
            def closure():
                optimizer2.zero_grad()
                l = loss_fn()
                l.backward()
                return l
            lval = optimizer2.step(closure)
            losses.append(float(lval.item() if hasattr(lval, "item") else lval))

    with torch.no_grad():
        sigma_pred = model(t_in).view(-1)
        errors = compute_errors(sigma_pred, sigma_exact)

    return model, errors, losses, sigma_pred.detach().cpu(), sigma_exact.detach().cpu()


def make_figures(config, t, sigma_exact, sigma_pred, losses, errors):
    import matplotlib.pyplot as plt

    style = NaturePlotStyle()

    fig1, axes = style.create_figure(width="double", height_ratio=0.5, nrows=1, ncols=2)
    ax1, ax2 = axes

    ax1.plot(t, sigma_exact, **style.get_line_style("blue", linewidth=1.5), label="Exact")
    ax1.plot(t, sigma_pred, **style.get_line_style("red", linestyle="--", linewidth=1.2), label="PIKAN")
    style.format_axis(ax1, xlabel="Time $t$ (s)", ylabel="Stress $\\sigma$ (Pa)", title="Stress Relaxation")
    ax1.legend(loc="upper right", frameon=False)
    style.add_panel_label(ax1, "a")

    abs_err = np.abs(sigma_pred - sigma_exact)
    ax2.semilogy(t, abs_err + 1e-12, **style.get_line_style("purple", linewidth=1.0))
    style.format_axis(ax2, xlabel="Time $t$ (s)", ylabel="Absolute error (Pa)", title="Error")
    style.add_panel_label(ax2, "b")

    plt.tight_layout()
    style.save_figure(fig1, "fig1_solution_comparison", config.figure_dir)

    fig2, ax = style.create_figure(width="single", height_ratio=0.75)
    ax.semilogy(np.arange(len(losses)), np.array(losses), **style.get_line_style("blue", linewidth=1.0))
    style.format_axis(ax, xlabel="Epoch", ylabel="Loss", title="Training loss")
    style.add_panel_label(ax, "a", x=-0.12, y=1.05)
    plt.tight_layout()
    style.save_figure(fig2, "fig2_training_loss", config.figure_dir)

    fig3, ax3 = style.create_figure(width="single", height_ratio=0.6)
    ax3.axis("off")
    table_data = [
        ["RMSE", f"{errors['rmse']:.2e}"],
        ["MAE", f"{errors['mae']:.2e}"],
        ["Max Error", f"{errors['max_error']:.2e}"],
        [r"$R^2$", f"{errors['r2']:.6f}"],
    ]
    table = ax3.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center", cellLoc="center", edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_linewidth(0)
    ax3.set_title("Error Metrics", fontweight="bold", fontsize=10, pad=10)
    plt.tight_layout()
    style.save_figure(fig3, "fig3_error_metrics", config.figure_dir)


def main():
    config = Config()
    torch.manual_seed(0)
    os.makedirs(config.figure_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)

    device = torch.device(config.device)
    t = torch.linspace(0.0, config.t_max, config.n_points, device=device)
    t[0] = 0.0

    model, errors, losses, sigma_pred, sigma_exact = train_pi_kan(config, t)
    print_error_table(errors, "PIKAN forward relaxation")

    np.savez(
        os.path.join(config.data_dir, "results.npz"),
        t=t.detach().cpu().numpy(),
        sigma_exact=sigma_exact.numpy(),
        sigma_pred_pi=sigma_pred.numpy(),
        errors_pi=errors,
        config=dict(
            tau=config.tau,
            alpha=config.alpha,
            E0=config.E0,
            E_inf=config.E_inf,
            epsilon0=config.epsilon0,
            t_max=config.t_max,
            n_points=config.n_points,
            width=config.width,
            grid=config.grid,
            k=config.k,
        ),
        loss_history=np.array(losses, dtype=np.float64),
    )
    torch.save(model.state_dict(), os.path.join(config.data_dir, "pi_kan_model.pth"))

    make_figures(
        config,
        t.detach().cpu().numpy(),
        sigma_exact.numpy(),
        sigma_pred.numpy(),
        losses,
        errors,
    )

    print("Example 01 done (results_paper/example_01/).")
    return errors


if __name__ == "__main__":
    main()
