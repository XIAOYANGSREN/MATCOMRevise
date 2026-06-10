"""Example 02: PIKAN forward solve under general loading, against the L1 reference.

sigma + tau^a D^a sigma = E_inf eps + E0 tau^a D^a eps with a smooth oscillatory
strain history (no closed-form solution).
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
from utils.fractional import (
    caputo_derivative_l1_series_fast,
    caputo_l1_kernel,
    solve_fractional_zener_l1,
)


class Config:
    # physical parameters (given; forward problem)
    tau = 1.5
    alpha = 0.6
    E0 = 120.0
    E_inf = 30.0

    # strain history
    epsilon_amp = 0.02
    omega = 2 * np.pi * 0.15  # rad/s

    # uniform time grid
    t_max = 40.0
    n_points = 160

    # PIKAN
    width = [1, 12, 1]
    grid = 12
    k = 3
    steps = 4000
    lr = 0.01
    # strong_wolfe line search re-evaluates the (expensive) history term many
    # times per step, so plain LBFGS steps are used here
    lbfgs_steps = 200
    lbfgs_lr = 0.5
    lbfgs_line_search = None

    output_dir = os.path.join(_RESULTS_ROOT, "example_02")
    figure_dir = os.path.join(_RESULTS_ROOT, "example_02", "figures")
    data_dir = os.path.join(_RESULTS_ROOT, "example_02", "data")

    device = "cuda" if torch.cuda.is_available() else "cpu"


def build_general_strain(t, amp, omega):
    # smooth onset avoids the t=0 kink: eps(t) = A(1-exp(-t/0.5)) sin(wt)
    gate = 1.0 - torch.exp(-t / (0.5 + 1e-6))
    return amp * gate * torch.sin(torch.tensor(float(omega), device=t.device, dtype=t.dtype) * t)


def train_pi_kan(config, t, epsilon):
    device = torch.device(config.device)
    t = t.to(device)
    epsilon = epsilon.to(device)
    # map time to [-1,1] for the B-spline grid; the residual still uses real dt
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
        model.speed()
    except Exception:
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    dt = (t[1] - t[0]).item()
    w_l1 = caputo_l1_kernel(nt=int(t.shape[0]), alpha=config.alpha, device=device, dtype=torch.float64)
    d_eps = caputo_derivative_l1_series_fast(epsilon, dt=dt, alpha=config.alpha, w=w_l1)

    # L1 numerical reference (not used in training)
    with torch.no_grad():
        sigma_ref = solve_fractional_zener_l1(
            t=t,
            epsilon=epsilon,
            alpha=config.alpha,
            tau_relax=config.tau,
            E0=config.E0,
            E_inf=config.E_inf,
        )

    losses = []
    pbar = tqdm(range(config.steps), desc="PIKAN example_02")

    def loss_fn():
        sigma_pred_local = model(t_in).view(-1)
        d_sigma_local = caputo_derivative_l1_series_fast(sigma_pred_local, dt=dt, alpha=config.alpha, w=w_l1)
        rhs_local = (config.E_inf * epsilon) + (config.E0 * (config.tau ** config.alpha)) * d_eps
        residual_local = sigma_pred_local + (config.tau ** config.alpha) * d_sigma_local - rhs_local
        ic_local = sigma_pred_local[0] - (config.E0 * epsilon[0])
        return torch.mean(residual_local[1:] ** 2) + 10.0 * (ic_local ** 2)

    for step in pbar:
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if step % 50 == 0:
            with torch.no_grad():
                sigma_tmp = model(t_in).view(-1)
                ic_tmp = sigma_tmp[0] - (config.E0 * epsilon[0])
            pbar.set_postfix({"loss": f"{loss.item():.2e}", "ic": f"{ic_tmp.item():.2e}"})

    if getattr(config, "lbfgs_steps", 0) and config.lbfgs_steps > 0:
        optimizer2 = torch.optim.LBFGS(
            model.parameters(),
            lr=config.lbfgs_lr,
            max_iter=1,
            line_search_fn=getattr(config, "lbfgs_line_search", None),
        )
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
        errors = compute_errors(sigma_pred, sigma_ref)

    return model, errors, losses, sigma_pred.detach().cpu(), sigma_ref.detach().cpu()


def make_figures(config, t, epsilon, sigma_ref, sigma_pred, losses, errors):
    import matplotlib.pyplot as plt

    style = NaturePlotStyle()

    fig1, axes = style.create_figure(width="double", height_ratio=0.55, nrows=1, ncols=3)
    ax1, ax2, ax3 = axes

    ax1.plot(t, epsilon, **style.get_line_style("black", linewidth=1.2))
    style.format_axis(ax1, xlabel="Time $t$ (s)", ylabel="Strain $\\epsilon(t)$", title="General loading")
    style.add_panel_label(ax1, "a")

    ax2.plot(t, sigma_ref, **style.get_line_style("blue", linewidth=1.5), label="L1 Reference")
    ax2.plot(t, sigma_pred, **style.get_line_style("red", linestyle="--", linewidth=1.1), label="PIKAN")
    style.format_axis(ax2, xlabel="Time $t$ (s)", ylabel="Stress $\\sigma$ (Pa)", title="Forward Response")
    ax2.legend(loc="upper right", frameon=False)
    style.add_panel_label(ax2, "b")

    abs_err = np.abs(sigma_pred - sigma_ref)
    ax3.semilogy(t, abs_err + 1e-12, **style.get_line_style("purple", linewidth=1.0))
    style.format_axis(ax3, xlabel="Time $t$ (s)", ylabel="Absolute error (Pa)", title="Error")
    style.add_panel_label(ax3, "c")

    plt.tight_layout()
    style.save_figure(fig1, "fig1_forward_general_loading", config.figure_dir)

    fig2, ax = style.create_figure(width="single", height_ratio=0.75)
    ax.semilogy(np.arange(len(losses)), np.array(losses), **style.get_line_style("blue", linewidth=1.0))
    style.format_axis(ax, xlabel="Epoch", ylabel="Loss", title="Training loss")
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
    epsilon = build_general_strain(t, config.epsilon_amp, config.omega)

    model, errors, losses, sigma_pred, sigma_ref = train_pi_kan(config, t, epsilon)
    print_error_table(errors, "PIKAN general loading (vs L1 reference)")

    np.savez(
        os.path.join(config.data_dir, "results.npz"),
        t=t.detach().cpu().numpy(),
        epsilon=epsilon.detach().cpu().numpy(),
        sigma_ref=sigma_ref.numpy(),
        sigma_pred_pi=sigma_pred.numpy(),
        errors_pi=errors,
        config=dict(
            tau=config.tau,
            alpha=config.alpha,
            E0=config.E0,
            E_inf=config.E_inf,
            epsilon_amp=config.epsilon_amp,
            omega=config.omega,
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
        epsilon.detach().cpu().numpy(),
        sigma_ref.numpy(),
        sigma_pred.numpy(),
        losses,
        errors,
    )

    print("Example 02 done (results_paper/example_02/).")
    return errors


if __name__ == "__main__":
    main()
