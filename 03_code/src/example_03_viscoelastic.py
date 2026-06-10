"""Example 03: parameter inversion on noisy relaxation data, Standard KAN vs PEKAN.

PEKAN encodes sigma(t) = eps0[E_inf + (E0-E_inf) E_alpha(-(t/tau)^alpha)] as a
3-layer network whose 4 trainable scalars are the material parameters.
"""

import sys
import os

_SRC = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_SRC, "..", ".."))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_PKG, "pykan"))
_RESULTS_ROOT = os.path.join(_PKG, "results_paper")

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from kan import KAN, LBFGS
from utils.plotting import NaturePlotStyle
from utils.fractional import exact_solution_zener, mittag_leffler_torch
from utils.metrics import compute_errors, print_error_table, print_parameter_comparison


class Config:
    # true parameters (to be inverted)
    tau_true = 2.0
    alpha_true = 0.5
    E0_true = 100.0
    E_inf_true = 20.0
    epsilon_0 = 1.0

    # data
    t_max = 20.0
    n_points = 100
    noise_level = 0.02

    # Standard KAN
    width_kan = [1, 10, 1]
    grid_kan = 12
    k_kan = 3
    steps_kan = 1500

    # PEKAN
    tau_init = 0.5
    alpha_init = 0.7
    E0_init = 60.0
    E_inf_init = 40.0
    steps_pekan_stage1 = 600
    steps_pekan_stage2 = 600
    steps_pekan_lbfgs = 400
    pekan_lbfgs_lr = 0.8

    output_dir = os.path.join(_RESULTS_ROOT, 'example_03')
    figure_dir = os.path.join(_RESULTS_ROOT, 'example_03', 'figures')
    data_dir = os.path.join(_RESULTS_ROOT, 'example_03', 'data')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PowerTransformLayer(nn.Module):
    """phi_1(t; tau, alpha) = -(t/tau)^alpha; log/logit reparam keeps tau>0, 0<alpha<1."""

    def __init__(self, tau_init=1.0, alpha_init=0.5):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau_init), dtype=torch.float32))
        alpha_init = np.clip(alpha_init, 0.01, 0.99)
        alpha_logit = np.log(alpha_init / (1 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))

    @property
    def tau(self):
        return torch.exp(self.log_tau)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, t):
        t_normalized = torch.clamp(t / self.tau, min=1e-10)
        return -(t_normalized ** self.alpha)


class MittagLefflerLayer(nn.Module):
    """phi_2(xi; alpha) = E_alpha(xi), the physics-fixed activation."""

    def forward(self, xi, alpha):
        # alpha must stay a tensor (no .item()) so its gradient propagates
        return mittag_leffler_torch(xi, alpha, beta=1.0)


class LinearScaleLayer(nn.Module):
    """phi_3(y; E0, E_inf) = eps0 [E_inf + (E0-E_inf) y]; log reparam keeps E0>E_inf>0."""

    def __init__(self, E0_init=100.0, E_inf_init=20.0, epsilon_0=1.0):
        super().__init__()
        self.epsilon_0 = epsilon_0
        self.log_E_inf = nn.Parameter(torch.tensor(np.log(E_inf_init), dtype=torch.float32))
        self.log_delta_E = nn.Parameter(torch.tensor(np.log(E0_init - E_inf_init), dtype=torch.float32))

    @property
    def E_inf(self):
        return torch.exp(self.log_E_inf)

    @property
    def E0(self):
        return self.E_inf + torch.exp(self.log_delta_E)

    def forward(self, y):
        return self.epsilon_0 * (self.E_inf + (self.E0 - self.E_inf) * y)


class PhysicsEncodedKAN(nn.Module):
    """PEKAN: t -> -(t/tau)^alpha -> E_alpha(.) -> E_inf+(E0-E_inf)y; 4 physical params."""

    def __init__(self, tau_init, alpha_init, E0_init, E_inf_init, epsilon_0=1.0):
        super().__init__()
        self.layer1 = PowerTransformLayer(tau_init, alpha_init)
        self.layer2 = MittagLefflerLayer()
        self.layer3 = LinearScaleLayer(E0_init, E_inf_init, epsilon_0)

    def forward(self, t):
        if t.dim() == 2:
            t = t.squeeze(-1)
        xi = self.layer1(t)
        y = self.layer2(xi, self.layer1.alpha)
        return self.layer3(y)

    def get_physical_params(self):
        return {
            'tau': self.layer1.tau.item(),
            'alpha': self.layer1.alpha.item(),
            'E0': self.layer3.E0.item(),
            'E_inf': self.layer3.E_inf.item(),
        }


def generate_experimental_data(config, device='cpu'):
    """Noisy synthetic relaxation data on a log-uniform grid (seed 42)."""
    t = torch.logspace(-2, np.log10(config.t_max), config.n_points, device=device)

    sigma_exact = exact_solution_zener(t, config.tau_true, config.alpha_true,
                                       config.E0_true, config.E_inf_true,
                                       config.epsilon_0)

    if config.noise_level > 0:
        torch.manual_seed(42)
        noise = torch.randn_like(sigma_exact) * config.noise_level * sigma_exact
        sigma_noisy = sigma_exact + noise
    else:
        sigma_noisy = sigma_exact.clone()

    return t, sigma_noisy, sigma_exact


def train_standard_kan(config, t, sigma_target, sigma_exact, device):
    """B-spline KAN curve fit (no physical parameters)."""
    print("\n--- Standard KAN (B-spline activations) ---")

    model = KAN(width=config.width_kan, grid=config.grid_kan, k=config.k_kan,
                grid_eps=1.0, noise_scale=0.1, device=device)
    print(f"  width={config.width_kan}, params={sum(p.numel() for p in model.parameters())}")

    # map time to the default [-1,1] spline range (PEKAN uses physical t)
    t_input = (2.0 * (t / float(config.t_max)) - 1.0).view(-1, 1)

    def loss_fn():
        pred = model(t_input).view(-1)
        return torch.mean((pred - sigma_target) ** 2)

    optimizer = LBFGS(model.parameters(), lr=0.1, history_size=20,
                      line_search_fn="strong_wolfe")

    pbar = tqdm(range(config.steps_kan), desc='Standard KAN')
    losses = []

    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        optimizer.step(closure)

        if step % 50 == 0:
            with torch.no_grad():
                loss = loss_fn()
                losses.append(loss.item())
                pbar.set_postfix({'Loss': f'{loss.item():.2e}'})

    with torch.no_grad():
        sigma_pred = model(t_input).view(-1)

    errors = compute_errors(sigma_pred, sigma_exact)
    print_error_table(errors, "Standard KAN fit")

    # symbolic regression attempt; E_alpha is not in the symbolic library,
    # which is exactly the point of the comparison
    try:
        n_hidden_kan = config.width_kan[1] if isinstance(config.width_kan[1], int) else 8
        for j in range(n_hidden_kan):
            suggestion = model.suggest_symbolic(0, 0, j,
                                                a_range=(0.01, 100),
                                                b_range=(-10, 10),
                                                weight_simple=0.0,
                                                topk=3,
                                                verbose=False)
            if suggestion[0] is not None:
                print(f"    edge (0,0)->(1,{j}): {suggestion[0]}, R^2 = {suggestion[2]:.4f}")
        print("  (Mittag-Leffler is not in the symbolic library, so no physical parameters here.)")
    except Exception as e:
        print(f"  symbolic regression failed: {e}")

    return model, sigma_pred, errors, losses


def train_physics_encoded_kan(config, t, sigma_target, sigma_exact, device):
    """PEKAN inversion: Adam coarse, Adam fine, then LBFGS."""
    print("\n--- PEKAN (Physics-Encoded KAN) ---")

    model = PhysicsEncodedKAN(
        tau_init=config.tau_init,
        alpha_init=config.alpha_init,
        E0_init=config.E0_init,
        E_inf_init=config.E_inf_init,
        epsilon_0=config.epsilon_0
    )

    init_params = model.get_physical_params()
    print(f"  init: tau={init_params['tau']:.4f}, alpha={init_params['alpha']:.4f}, "
          f"E0={init_params['E0']:.2f}, E_inf={init_params['E_inf']:.2f}")

    history = {'loss': [], 'params': []}

    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.05)
    pbar1 = tqdm(range(config.steps_pekan_stage1), desc='PEKAN stage 1')
    for step in pbar1:
        optimizer1.zero_grad()
        pred = model(t)
        loss = torch.mean((pred - sigma_target) ** 2)
        loss.backward()
        optimizer1.step()

        history['loss'].append(loss.item())
        history['params'].append(model.get_physical_params())

        if step % 50 == 0:
            params = model.get_physical_params()
            pbar1.set_postfix({'Loss': f'{loss.item():.2e}',
                               'tau': f'{params["tau"]:.3f}',
                               'alpha': f'{params["alpha"]:.3f}'})

    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.005)
    pbar2 = tqdm(range(config.steps_pekan_stage2), desc='PEKAN stage 2')
    for step in pbar2:
        optimizer2.zero_grad()
        pred = model(t)
        loss = torch.mean((pred - sigma_target) ** 2)
        loss.backward()
        optimizer2.step()

        history['loss'].append(loss.item())
        history['params'].append(model.get_physical_params())

        if step % 50 == 0:
            params = model.get_physical_params()
            pbar2.set_postfix({'Loss': f'{loss.item():.2e}',
                               'tau': f'{params["tau"]:.3f}',
                               'alpha': f'{params["alpha"]:.3f}'})

    if getattr(config, "steps_pekan_lbfgs", 0) and config.steps_pekan_lbfgs > 0:
        optimizer3 = torch.optim.LBFGS(
            model.parameters(),
            lr=float(config.pekan_lbfgs_lr),
            max_iter=1,
            line_search_fn="strong_wolfe",
        )
        pbar3 = tqdm(range(config.steps_pekan_lbfgs), desc='PEKAN stage 3 (LBFGS)')
        for step in pbar3:
            def closure():
                optimizer3.zero_grad()
                pred = model(t)
                l = torch.mean((pred - sigma_target) ** 2)
                l.backward()
                return l
            lval = optimizer3.step(closure)

            history['loss'].append(float(lval.item() if hasattr(lval, "item") else lval))
            history['params'].append(model.get_physical_params())

            if step % 25 == 0:
                params = model.get_physical_params()
                pbar3.set_postfix({'Loss': f'{history["loss"][-1]:.2e}',
                                   'tau': f'{params["tau"]:.3f}',
                                   'alpha': f'{params["alpha"]:.3f}'})

    with torch.no_grad():
        sigma_pred = model(t)

    errors = compute_errors(sigma_pred, sigma_exact)
    print_error_table(errors, "PEKAN fit")

    true_params = {
        'tau': config.tau_true,
        'alpha': config.alpha_true,
        'E0': config.E0_true,
        'E_inf': config.E_inf_true
    }
    final_params = model.get_physical_params()
    print_parameter_comparison(true_params, final_params, "Parameter inversion")

    return model, sigma_pred, errors, history, final_params


def create_visualizations(config, t, sigma_noisy, sigma_exact,
                          sigma_kan, sigma_pekan,
                          errors_kan, errors_pekan,
                          history_pekan, final_params):
    import matplotlib.pyplot as plt

    style = NaturePlotStyle()

    t_np = t.cpu().numpy()
    sigma_noisy_np = sigma_noisy.cpu().numpy()
    sigma_exact_np = sigma_exact.cpu().numpy()
    sigma_kan_np = sigma_kan.cpu().numpy()
    sigma_pekan_np = sigma_pekan.cpu().numpy()

    t_fine = torch.logspace(-2, np.log10(config.t_max), 500)
    sigma_fine = exact_solution_zener(t_fine, config.tau_true, config.alpha_true,
                                      config.E0_true, config.E_inf_true,
                                      config.epsilon_0).numpy()
    t_fine_np = t_fine.numpy()

    # figure 1: solutions, errors, parameter convergence
    fig1, axes = style.create_figure(width='double', height_ratio=0.55,
                                     nrows=1, ncols=3)

    ax1 = axes[0]
    ax1.scatter(t_np, sigma_noisy_np, s=15, alpha=0.5, c=style.colors['gray'],
                label='Noisy data', zorder=1)
    ax1.plot(t_fine_np, sigma_fine, **style.get_line_style('blue', linewidth=1.5),
             label='Exact', zorder=2)
    ax1.plot(t_np, sigma_kan_np, **style.get_line_style('orange', linestyle='--'),
             label='Standard KAN', zorder=3)
    ax1.plot(t_np, sigma_pekan_np, **style.get_line_style('green', linestyle=':'),
             label='PEKAN', zorder=4)
    style.format_axis(ax1, xlabel='Time $t$ (s)', ylabel='Stress $\\sigma$ (Pa)',
                      title='Stress Relaxation', legend=False)
    ax1.legend(loc='upper right', frameon=False, fontsize=6)
    style.add_panel_label(ax1, 'a')

    ax2 = axes[1]
    ax2.semilogy(t_np, np.abs(sigma_kan_np - sigma_exact_np),
                 **style.get_line_style('orange'), label='Standard KAN')
    ax2.semilogy(t_np, np.abs(sigma_pekan_np - sigma_exact_np),
                 **style.get_line_style('green'), label='PEKAN')
    style.format_axis(ax2, xlabel='Time $t$ (s)', ylabel='Absolute Error (Pa)',
                      title='Error Comparison', legend=False)
    ax2.legend(loc='upper right', frameon=False, fontsize=6)
    style.add_panel_label(ax2, 'b')

    ax3 = axes[2]
    epochs = np.arange(len(history_pekan['params']))
    tau_history = [p['tau'] for p in history_pekan['params']]
    alpha_history = [p['alpha'] for p in history_pekan['params']]

    ax3_twin = ax3.twinx()
    l1, = ax3.plot(epochs, tau_history, **style.get_line_style('blue'), label='$\\tau$')
    l2, = ax3_twin.plot(epochs, alpha_history, **style.get_line_style('red'), label='$\\alpha$')
    ax3.axhline(y=config.tau_true, color=style.colors['blue'], linestyle=':', alpha=0.5)
    ax3_twin.axhline(y=config.alpha_true, color=style.colors['red'], linestyle=':', alpha=0.5)
    ax3.axvline(x=config.steps_pekan_stage1, color='gray', linestyle='--', alpha=0.3)

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('$\\tau$ (s)', color=style.colors['blue'])
    ax3_twin.set_ylabel('$\\alpha$', color=style.colors['red'])
    ax3.set_title('Parameter Convergence', fontweight='bold', fontsize=10)
    ax3.legend([l1, l2], ['$\\tau$', '$\\alpha$'], loc='center right')
    style.add_panel_label(ax3, 'c')

    plt.tight_layout()
    style.save_figure(fig1, 'fig1_method_comparison', config.figure_dir)

    # figure 2: detailed analysis
    fig2, axes2 = style.create_figure(width='double', height_ratio=0.75,
                                      nrows=2, ncols=3)

    ax21 = axes2[0, 0]
    ax21.semilogx(t_fine_np, sigma_fine, **style.get_line_style('blue', linewidth=1.5),
                  label='Exact')
    ax21.semilogx(t_np, sigma_pekan_np, **style.get_line_style('green', linestyle='--'),
                  label='PEKAN')
    ax21.scatter(t_np, sigma_noisy_np, s=10, alpha=0.3, c=style.colors['gray'])
    style.format_axis(ax21, xlabel='Time $t$ (s) [log]', ylabel='Stress $\\sigma$ (Pa)',
                      title='Log Time Scale', legend=False)
    ax21.legend(loc='upper right', frameon=False, fontsize=6)
    style.add_panel_label(ax21, 'a')

    ax22 = axes2[0, 1]
    ax22.semilogy(history_pekan['loss'], **style.get_line_style('blue', linewidth=0.8))
    ax22.axvline(x=config.steps_pekan_stage1, color='gray', linestyle='--', alpha=0.5,
                 label='Stage 2 start')
    style.format_axis(ax22, xlabel='Epoch', ylabel='Loss',
                      title='Training Loss')
    style.add_panel_label(ax22, 'b')

    ax23 = axes2[0, 2]
    E0_history = [p['E0'] for p in history_pekan['params']]
    Einf_history = [p['E_inf'] for p in history_pekan['params']]
    ax23.plot(epochs, E0_history, **style.get_line_style('blue'), label='$E_0$')
    ax23.plot(epochs, Einf_history, **style.get_line_style('red'), label='$E_\\infty$')
    ax23.axhline(y=config.E0_true, color=style.colors['blue'], linestyle=':', alpha=0.5)
    ax23.axhline(y=config.E_inf_true, color=style.colors['red'], linestyle=':', alpha=0.5)
    style.format_axis(ax23, xlabel='Epoch', ylabel='Modulus (Pa)',
                      title='Modulus Convergence')
    style.add_panel_label(ax23, 'c')

    ax24 = axes2[1, 0]
    rel_err_kan = np.abs(sigma_kan_np - sigma_exact_np) / sigma_exact_np * 100
    rel_err_pekan = np.abs(sigma_pekan_np - sigma_exact_np) / sigma_exact_np * 100
    ax24.plot(t_np, rel_err_kan, **style.get_line_style('orange'), label='Standard KAN')
    ax24.plot(t_np, rel_err_pekan, **style.get_line_style('green'), label='PEKAN')
    ax24.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    style.format_axis(ax24, xlabel='Time $t$ (s)', ylabel='Relative Error (%)',
                      title='Relative Error', legend=False)
    ax24.legend(loc='upper right', frameon=False, fontsize=6)
    style.add_panel_label(ax24, 'd')

    ax25 = axes2[1, 1]
    residuals = sigma_pekan_np - sigma_exact_np
    ax25.scatter(t_np, residuals, s=20, alpha=0.6, c=style.colors['purple'])
    ax25.axhline(y=0, color='black', linewidth=0.5)
    ax25.fill_between(t_np, -np.std(residuals)*2, np.std(residuals)*2,
                      alpha=0.2, color='gray')
    style.format_axis(ax25, xlabel='Time $t$ (s)', ylabel='Residual (Pa)',
                      title='PEKAN Residuals')
    style.add_panel_label(ax25, 'e')

    ax26 = axes2[1, 2]
    ax26.axis('off')

    param_labels = [r'$\tau$', r'$\alpha$', r'$E_0$', r'$E_\infty$']
    true_vals = [config.tau_true, config.alpha_true, config.E0_true, config.E_inf_true]
    inv_vals = [final_params['tau'], final_params['alpha'], final_params['E0'], final_params['E_inf']]

    table_data = []
    for label, true_val, inv_val in zip(param_labels, true_vals, inv_vals):
        rel_err = abs(inv_val - true_val) / true_val * 100
        table_data.append([label, f'{true_val:.4f}', f'{inv_val:.4f}', f'{rel_err:.2f}%'])

    table = ax26.table(cellText=table_data,
                       colLabels=['Param', 'True', 'Inverted', 'Error'],
                       loc='center',
                       cellLoc='center',
                       edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_linewidth(0)
    ax26.set_title('Parameter Inversion', fontweight='bold', fontsize=10, pad=10)
    style.add_panel_label(ax26, 'f')

    plt.tight_layout()
    style.save_figure(fig2, 'fig2_detailed_analysis', config.figure_dir)

    # figure 3: method comparison bars
    fig3, ax3 = style.create_figure(width='single', height_ratio=0.85)

    methods = ['Standard\nKAN', 'PEKAN']
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax3.bar(x - width/2, [errors_kan['rmse']*1000, errors_pekan['rmse']*1000],
                    width, label=r'RMSE ($\times 10^{-3}$)', color=style.colors['blue'], alpha=0.8)

    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, [errors_kan['r2'], errors_pekan['r2']],
                         width, label=r'$R^2$', color=style.colors['green'], alpha=0.8)

    ax3.set_ylabel(r'RMSE ($\times 10^{-3}$)')
    ax3_twin.set_ylabel(r'$R^2$')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.set_title('Method Comparison', fontweight='bold')

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)

    plt.tight_layout()
    style.save_figure(fig3, 'fig3_method_comparison_bar', config.figure_dir)

    return fig1, fig2, fig3


def apply_output_root(config, output_root):
    config.output_dir = os.path.join(output_root, 'example_03')
    config.figure_dir = os.path.join(config.output_dir, 'figures')
    config.data_dir = os.path.join(config.output_dir, 'data')


def main(output_root=None):
    config = Config()
    torch.manual_seed(0)
    if output_root:
        apply_output_root(config, output_root)

    os.makedirs(config.figure_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)

    print("\n=== Example 03: parameter inversion, Standard KAN vs PEKAN ===")
    print(f"true params: tau={config.tau_true}, alpha={config.alpha_true}, "
          f"E0={config.E0_true}, E_inf={config.E_inf_true}, noise={config.noise_level*100:.0f}%")

    device = torch.device(config.device)

    t, sigma_noisy, sigma_exact = generate_experimental_data(config, device)

    model_kan, sigma_kan, errors_kan, losses_kan = train_standard_kan(
        config, t, sigma_noisy, sigma_exact, device
    )

    model_pekan, sigma_pekan, errors_pekan, history_pekan, final_params = train_physics_encoded_kan(
        config, t, sigma_noisy, sigma_exact, device
    )

    print(f"\n  {'metric':<14} {'Standard KAN':<16} {'PEKAN':<16}")
    print(f"  {'n_params':<14} {sum(p.numel() for p in model_kan.parameters()):<16} {sum(p.numel() for p in model_pekan.parameters()):<16}")
    print(f"  {'RMSE':<14} {errors_kan['rmse']:<16.6e} {errors_pekan['rmse']:<16.6e}")
    print(f"  {'R^2':<14} {errors_kan['r2']:<16.6f} {errors_pekan['r2']:<16.6f}")
    print(f"\n  PEKAN inversion:")
    print(f"    tau  : {config.tau_true:.4f} -> {final_params['tau']:.4f} (err: {abs(final_params['tau']-config.tau_true)/config.tau_true*100:.2f}%)")
    print(f"    alpha: {config.alpha_true:.4f} -> {final_params['alpha']:.4f} (err: {abs(final_params['alpha']-config.alpha_true)/config.alpha_true*100:.2f}%)")
    print(f"    E0   : {config.E0_true:.2f} -> {final_params['E0']:.2f} (err: {abs(final_params['E0']-config.E0_true)/config.E0_true*100:.2f}%)")
    print(f"    E_inf: {config.E_inf_true:.2f} -> {final_params['E_inf']:.2f} (err: {abs(final_params['E_inf']-config.E_inf_true)/config.E_inf_true*100:.2f}%)")

    create_visualizations(config, t, sigma_noisy, sigma_exact,
                          sigma_kan, sigma_pekan,
                          errors_kan, errors_pekan,
                          history_pekan, final_params)

    torch.save(model_pekan.state_dict(), f"{config.data_dir}/pekan_model.pth")
    torch.save(model_kan.state_dict(), f"{config.data_dir}/kan_model.pth")

    tau_history = np.array([p['tau'] for p in history_pekan['params']])
    alpha_history = np.array([p['alpha'] for p in history_pekan['params']])
    E0_history = np.array([p['E0'] for p in history_pekan['params']])
    E_inf_history = np.array([p['E_inf'] for p in history_pekan['params']])
    loss_history = np.array(history_pekan['loss'])

    np.savez(f"{config.data_dir}/results.npz",
             t=t.cpu().numpy(),
             sigma_noisy=sigma_noisy.cpu().numpy(),
             sigma_exact=sigma_exact.cpu().numpy(),
             sigma_kan=sigma_kan.cpu().numpy(),
             sigma_pekan=sigma_pekan.cpu().numpy(),
             true_params=dict(tau=config.tau_true, alpha=config.alpha_true,
                              E0=config.E0_true, E_inf=config.E_inf_true),
             inverted_params=final_params,
             errors_kan=errors_kan,
             errors_pekan=errors_pekan,
             tau_history=tau_history,
             alpha_history=alpha_history,
             E0_history=E0_history,
             E_inf_history=E_inf_history,
             loss_history=loss_history,
             stage1_steps=config.steps_pekan_stage1)

    print("Example 03 done (results_paper/example_03/).")
    return model_kan, model_pekan, final_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default=None)
    args = parser.parse_args()

    main(output_root=args.output_root)
