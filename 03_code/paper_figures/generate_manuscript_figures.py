"""Result figures from the saved npz files: PIKAN relaxation, PIKAN general
loading, PEKAN inversion, plus a summary.
Needs results_paper/ (run 03_code/src/run_all_examples.py first).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

COLORS = {
    'blue': '#0173B2',
    'red': '#D55E00',
    'green': '#029E73',
    'orange': '#DE8F05',
    'purple': '#CC78BC',
    'cyan': '#56B4E9',
    'gray': '#949494',
    'black': '#000000',
    'light_gray': '#E5E5E5',
}

def setup_nature_style():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,

        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,

        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,

        'legend.frameon': False,
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,

        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.06,

        'axes.spines.top': True,
        'axes.spines.right': True,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,

        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
    })

SINGLE_COL = 89 / 25.4
DOUBLE_COL = 183 / 25.4
MEDIUM_COL = 140 / 25.4

def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=9):
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left',
            clip_on=False, zorder=20)

def save_figure(fig, filename, output_dir, formats=['pdf', 'png', 'tiff']):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"
        if fmt == 'tiff':
            fig.savefig(filepath, dpi=600, facecolor='white', format='tiff',
                       pil_kwargs={'compression': 'tiff_lzw'})
        elif fmt == 'png':
            fig.savefig(filepath, dpi=600, facecolor='white')
        else:
            fig.savefig(filepath, facecolor='white')
        print(f"  Saved: {filepath}")


def generate_figure1(data_dir, output_dir):
    """Figure 4: PIKAN forward stress relaxation against the analytic benchmark."""
    print("\n[Figure 4] PIKAN stress relaxation...")

    data = np.load(os.path.join(data_dir, 'example_01/data/results.npz'), allow_pickle=True)
    t = data['t']
    sigma_exact = data['sigma_exact']
    sigma_pred = data['sigma_pred_pi']
    losses = data['loss_history']
    errors = data['errors_pi'].item()
    config = data['config'].item()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.32))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, sigma_exact, color=COLORS['blue'], linewidth=1.2, label='Analytical', zorder=2)
    ax1.plot(t, sigma_pred, color=COLORS['red'], linewidth=1.0, linestyle='--', label='PIKAN', zorder=3)
    ax1.set_xlabel('Time $t$ (s)')
    ax1.set_ylabel('Stress $\\sigma$ (Pa)')
    ax1.set_xlim([0, t.max()])
    ax1.set_ylim([0, 110])
    ax1.legend(loc='upper right', frameon=False)

    param_text = f"$\\tau$={config['tau']}, $\\alpha$={config['alpha']}\n$E_0$={config['E0']}, $E_\\infty$={config['E_inf']}"
    ax1.text(0.98, 0.55, param_text, transform=ax1.transAxes, fontsize=5.5,
             ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    add_panel_label(ax1, 'a')

    ax2 = fig.add_subplot(gs[1])
    abs_error = np.abs(sigma_pred - sigma_exact)
    ax2.semilogy(t, abs_error + 1e-10, color=COLORS['purple'], linewidth=0.8)
    ax2.set_xlabel('Time $t$ (s)')
    ax2.set_ylabel('Absolute error (Pa)')
    ax2.set_xlim([0, t.max()])
    ax2.set_ylim([1e-3, 1e1])

    metrics_text = f"RMSE = {errors['rmse']:.2e}\n$R^2$ = {errors['r2']:.4f}"
    ax2.text(0.98, 0.98, metrics_text, transform=ax2.transAxes, fontsize=5.5,
             ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    add_panel_label(ax2, 'b')

    ax3 = fig.add_subplot(gs[2])
    epochs = np.arange(len(losses))
    ax3.semilogy(epochs, losses, color=COLORS['blue'], linewidth=0.6)
    ax3.axvline(x=3000, color=COLORS['gray'], linestyle=':', linewidth=0.5, alpha=0.7)
    ax3.text(3000, losses[0]*0.8, 'L-BFGS', fontsize=5, ha='left', va='top', color=COLORS['gray'])
    ax3.set_xlabel('Training epoch')
    ax3.set_ylabel('Loss')
    ax3.set_xlim([0, len(losses)])
    add_panel_label(ax3, 'c')

    plt.tight_layout()
    save_figure(fig, 'Figure4_PIKAN_relaxation', output_dir)
    plt.close(fig)

    return fig


def generate_figure2(data_dir, output_dir):
    """Figure 5: PIKAN forward solution under general loading (L1 numerical reference)."""
    print("\n[Figure 5] PIKAN general loading...")

    data = np.load(os.path.join(data_dir, 'example_02/data/results.npz'), allow_pickle=True)
    t = data['t']
    epsilon = data['epsilon']
    sigma_ref = data['sigma_ref']
    sigma_pred = data['sigma_pred_pi']
    losses = data['loss_history']
    errors = data['errors_pi'].item()
    config = data['config'].item()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], wspace=0.3, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, epsilon * 1000, color=COLORS['black'], linewidth=0.8)  # millistrain
    ax1.set_xlabel('Time $t$ (s)')
    ax1.set_ylabel('Strain $\\varepsilon$ (×10$^{-3}$)')
    ax1.set_xlim([0, t.max()])
    ax1.axhline(y=0, color=COLORS['gray'], linestyle='-', linewidth=0.3)

    load_text = "$\\varepsilon(t) = A(1-e^{-t/0.5})\\sin(\\omega t)$"
    ax1.text(0.98, 0.98, load_text, transform=ax1.transAxes, fontsize=5.5,
             ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9, edgecolor='none'))
    add_panel_label(ax1, 'a')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, sigma_ref, color=COLORS['blue'], linewidth=1.0, label='L1 reference')
    ax2.plot(t, sigma_pred, color=COLORS['red'], linewidth=0.8, linestyle='--', label='PIKAN')
    ax2.set_xlabel('Time $t$ (s)')
    ax2.set_ylabel('Stress $\\sigma$ (Pa)')
    ax2.set_xlim([0, t.max()])
    ax2.legend(loc='upper right', frameon=False)
    ax2.axhline(y=0, color=COLORS['gray'], linestyle='-', linewidth=0.3)
    add_panel_label(ax2, 'b')

    ax3 = fig.add_subplot(gs[1, 0])
    abs_error = np.abs(sigma_pred - sigma_ref)
    ax3.semilogy(t, abs_error + 1e-10, color=COLORS['purple'], linewidth=0.6)
    ax3.set_xlabel('Time $t$ (s)')
    ax3.set_ylabel('Absolute error (Pa)')
    ax3.set_xlim([0, t.max()])

    metrics_text = f"RMSE = {errors['rmse']:.2e}\nMAE = {errors['mae']:.2e}\n$R^2$ = {errors['r2']:.8f}"
    ax3.text(0.98, 0.98, metrics_text, transform=ax3.transAxes, fontsize=5.5,
             ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    add_panel_label(ax3, 'c')

    ax4 = fig.add_subplot(gs[1, 1])
    epochs = np.arange(len(losses))
    ax4.semilogy(epochs, losses, color=COLORS['blue'], linewidth=0.5)
    ax4.set_xlabel('Training epoch')
    ax4.set_ylabel('Loss')
    ax4.set_xlim([0, len(losses)])

    adam_steps = 4000
    if len(losses) > adam_steps:
        ax4.axvline(x=adam_steps, color=COLORS['gray'], linestyle=':', linewidth=0.5, alpha=0.7)
        ax4.text(adam_steps + 50, losses[0]*0.5, 'L-BFGS', fontsize=5, ha='left', color=COLORS['gray'])
    add_panel_label(ax4, 'd')

    plt.tight_layout(pad=0.8)
    save_figure(fig, 'Figure5_PIKAN_general_loading', output_dir)
    plt.close(fig)

    return fig


def generate_figure3(data_dir, output_dir):
    """Figure 6: PEKAN parameter inversion compared against a standard KAN."""
    print("\n[Figure 6] PEKAN parameter inversion...")

    data = np.load(os.path.join(data_dir, 'example_03/data/results.npz'), allow_pickle=True)
    t = data['t']
    sigma_noisy = data['sigma_noisy']
    sigma_exact = data['sigma_exact']
    sigma_kan = data['sigma_kan']
    sigma_pekan = data['sigma_pekan']
    true_params = data['true_params'].item()
    inverted_params = data['inverted_params'].item()
    errors_kan = data['errors_kan'].item()
    errors_pekan = data['errors_pekan'].item()

    tau_history = data['tau_history']
    alpha_history = data['alpha_history']
    E0_history = data['E0_history']
    E_inf_history = data['E_inf_history']
    loss_history = data['loss_history']
    stage1_steps = data['stage1_steps']

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.7))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], wspace=0.35, hspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(t, sigma_noisy, s=8, c=COLORS['gray'], alpha=0.4, label='Noisy data', zorder=1, edgecolors='none')
    ax1.plot(t, sigma_exact, color=COLORS['blue'], linewidth=1.2, label='Exact', zorder=2)
    ax1.plot(t, sigma_kan, color=COLORS['orange'], linewidth=0.8, linestyle='--', label='Standard KAN', zorder=3)
    ax1.plot(t, sigma_pekan, color=COLORS['green'], linewidth=0.8, linestyle=':', label='PEKAN', zorder=4)
    ax1.set_xlabel('Time $t$ (s)')
    ax1.set_ylabel('Stress $\\sigma$ (Pa)')
    ax1.set_xlim([0, t.max()])
    ax1.legend(loc='upper right', frameon=False, fontsize=5, markerscale=0.8)
    add_panel_label(ax1, 'a')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(t, np.abs(sigma_kan - sigma_exact) + 1e-10, color=COLORS['orange'],
                 linewidth=0.8, label='Standard KAN')
    ax2.semilogy(t, np.abs(sigma_pekan - sigma_exact) + 1e-10, color=COLORS['green'],
                 linewidth=0.8, label='PEKAN')
    ax2.set_xlabel('Time $t$ (s)')
    ax2.set_ylabel('Absolute error (Pa)')
    ax2.set_xlim([0, t.max()])
    ax2.legend(loc='upper right', frameon=False, fontsize=5)

    rmse_text = f"RMSE:\nKAN: {errors_kan['rmse']:.2e}\nPEKAN: {errors_pekan['rmse']:.2e}"
    ax2.text(0.98, 0.02, rmse_text, transform=ax2.transAxes, fontsize=5,
             ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))
    add_panel_label(ax2, 'b')

    ax3 = fig.add_subplot(gs[0, 2])
    epochs = np.arange(len(tau_history))
    ax3.plot(epochs, tau_history, color=COLORS['blue'], linewidth=0.8)
    ax3.axhline(y=true_params['tau'], color=COLORS['blue'], linestyle=':', linewidth=0.5, alpha=0.7)
    ax3.axvline(x=stage1_steps, color=COLORS['gray'], linestyle='--', linewidth=0.3, alpha=0.5)
    ax3.set_xlabel('Training epoch')
    ax3.set_ylabel('$\\tau$ (s)')
    ax3.set_xlim([0, len(tau_history)])

    ax3.text(0.98, 0.98, f"True: {true_params['tau']:.2f}\nInv: {inverted_params['tau']:.3f}",
             transform=ax3.transAxes, fontsize=5, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    add_panel_label(ax3, 'c')

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, alpha_history, color=COLORS['red'], linewidth=0.8)
    ax4.axhline(y=true_params['alpha'], color=COLORS['red'], linestyle=':', linewidth=0.5, alpha=0.7)
    ax4.axvline(x=stage1_steps, color=COLORS['gray'], linestyle='--', linewidth=0.3, alpha=0.5)
    ax4.set_xlabel('Training epoch')
    ax4.set_ylabel('$\\alpha$')
    ax4.set_xlim([0, len(alpha_history)])
    ax4.set_ylim([0.4, 0.8])

    ax4.text(0.98, 0.98, f"True: {true_params['alpha']:.2f}\nInv: {inverted_params['alpha']:.4f}",
             transform=ax4.transAxes, fontsize=5, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    add_panel_label(ax4, 'd')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, E0_history, color=COLORS['blue'], linewidth=0.8, label='$E_0$')
    ax5.plot(epochs, E_inf_history, color=COLORS['green'], linewidth=0.8, label='$E_\\infty$')
    ax5.axhline(y=true_params['E0'], color=COLORS['blue'], linestyle=':', linewidth=0.5, alpha=0.7)
    ax5.axhline(y=true_params['E_inf'], color=COLORS['green'], linestyle=':', linewidth=0.5, alpha=0.7)
    ax5.axvline(x=stage1_steps, color=COLORS['gray'], linestyle='--', linewidth=0.3, alpha=0.5)
    ax5.set_xlabel('Training epoch')
    ax5.set_ylabel('Modulus (Pa)')
    ax5.set_xlim([0, len(E0_history)])
    ax5.legend(loc='right', frameon=False, fontsize=5)
    add_panel_label(ax5, 'e')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    params = ['$\\tau$ (s)', '$\\alpha$', '$E_0$ (Pa)', '$E_\\infty$ (Pa)']
    true_vals = [true_params['tau'], true_params['alpha'], true_params['E0'], true_params['E_inf']]
    inv_vals = [inverted_params['tau'], inverted_params['alpha'], inverted_params['E0'], inverted_params['E_inf']]
    errors = [abs(inv - true) / true * 100 for inv, true in zip(inv_vals, true_vals)]

    table_data = []
    for p, t_val, i_val, err in zip(params, true_vals, inv_vals, errors):
        table_data.append([p, f'{t_val:.2f}', f'{i_val:.4f}', f'{err:.2f}%'])

    table = ax6.table(cellText=table_data,
                      colLabels=['Parameter', 'True', 'Inverted', 'Error'],
                      loc='center',
                      cellLoc='center',
                      colWidths=[0.3, 0.2, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor(COLORS['light_gray'])
        else:
            cell.set_facecolor('white')

    ax6.set_title('PEKAN Inversion Results', fontsize=7, fontweight='bold', pad=8)
    add_panel_label(ax6, 'f', x=-0.12, y=1.06)

    plt.tight_layout(pad=0.8)
    save_figure(fig, 'Figure6_PEKAN_inversion', output_dir)
    plt.close(fig)

    return fig


def generate_figure4(data_dir, output_dir):
    """Method summary: per-example R2, KAN vs PEKAN RMSE, PEKAN schematic."""
    print("\n[Summary] Method comparison summary...")

    data1 = np.load(os.path.join(data_dir, 'example_01/data/results.npz'), allow_pickle=True)
    data2 = np.load(os.path.join(data_dir, 'example_02/data/results.npz'), allow_pickle=True)
    data3 = np.load(os.path.join(data_dir, 'example_03/data/results.npz'), allow_pickle=True)

    errors1 = data1['errors_pi'].item()
    errors2 = data2['errors_pi'].item()
    errors_kan = data3['errors_kan'].item()
    errors_pekan = data3['errors_pekan'].item()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.4)

    ax1 = fig.add_subplot(gs[0])

    examples = ['Ex.1\n(Relaxation)', 'Ex.2\n(General)', 'Ex.3\n(Inversion)']
    r2_pikan = [errors1['r2'], errors2['r2'], errors_pekan['r2']]

    x = np.arange(len(examples))
    bars = ax1.bar(x, r2_pikan, width=0.5, color=[COLORS['blue'], COLORS['green'], COLORS['purple']],
                   edgecolor='black', linewidth=0.3)

    ax1.set_ylabel('$R^2$ coefficient')
    ax1.set_xticks(x)
    ax1.set_xticklabels(examples, fontsize=5.5)
    ax1.set_ylim([0.99, 1.001])
    ax1.set_yticks([0.99, 0.995, 1.0])

    for bar, r2 in zip(bars, r2_pikan):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.6f}', ha='center', va='bottom', fontsize=5, rotation=0)

    add_panel_label(ax1, 'a')

    ax2 = fig.add_subplot(gs[1])

    methods = ['Standard\nKAN', 'PEKAN']
    rmse_vals = [errors_kan['rmse'], errors_pekan['rmse']]

    x = np.arange(len(methods))
    bars = ax2.bar(x, rmse_vals, width=0.4, color=[COLORS['orange'], COLORS['green']],
                   edgecolor='black', linewidth=0.3)

    ax2.set_ylabel('RMSE (Pa)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=6)
    ax2.set_yscale('log')

    for bar, rmse in zip(bars, rmse_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.3,
                f'{rmse:.2e}', ha='center', va='bottom', fontsize=5)

    improvement = rmse_vals[0] / rmse_vals[1]
    ax2.annotate('', xy=(1, rmse_vals[1]), xytext=(0, rmse_vals[0]),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=0.8))
    ax2.text(0.5, np.sqrt(rmse_vals[0] * rmse_vals[1]), f'{improvement:.1f}×',
            ha='center', va='center', fontsize=6, color=COLORS['gray'])

    add_panel_label(ax2, 'b')

    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    ax3.add_patch(plt.Circle((0.1, 0.5), 0.06, color=COLORS['blue'], ec='black', lw=0.5))
    ax3.text(0.1, 0.5, '$t$', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax3.add_patch(plt.Rectangle((0.25, 0.35), 0.15, 0.3, color=COLORS['cyan'], ec='black', lw=0.5, alpha=0.7))
    ax3.text(0.325, 0.5, '$\\phi_1$', ha='center', va='center', fontsize=7)
    ax3.text(0.325, 0.25, '$\\xi=-(t/\\tau)^\\alpha$', ha='center', va='top', fontsize=4.5)

    ax3.add_patch(plt.Rectangle((0.45, 0.35), 0.15, 0.3, color=COLORS['green'], ec='black', lw=0.5, alpha=0.7))
    ax3.text(0.525, 0.5, '$\\phi_2$', ha='center', va='center', fontsize=7)
    ax3.text(0.525, 0.25, '$y=E_\\alpha(\\xi)$', ha='center', va='top', fontsize=4.5)

    ax3.add_patch(plt.Rectangle((0.65, 0.35), 0.15, 0.3, color=COLORS['orange'], ec='black', lw=0.5, alpha=0.7))
    ax3.text(0.725, 0.5, '$\\phi_3$', ha='center', va='center', fontsize=7)
    ax3.text(0.725, 0.25, '$\\sigma=E_\\infty+(E_0-E_\\infty)y$', ha='center', va='top', fontsize=4)

    ax3.add_patch(plt.Circle((0.9, 0.5), 0.06, color=COLORS['red'], ec='black', lw=0.5))
    ax3.text(0.9, 0.5, '$\\sigma$', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax3.annotate('', xy=(0.24, 0.5), xytext=(0.17, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax3.annotate('', xy=(0.44, 0.5), xytext=(0.41, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax3.annotate('', xy=(0.64, 0.5), xytext=(0.61, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax3.annotate('', xy=(0.83, 0.5), xytext=(0.81, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax3.text(0.325, 0.72, '$\\tau, \\alpha$', ha='center', va='bottom', fontsize=5.5, color=COLORS['blue'])
    ax3.text(0.525, 0.72, '$\\alpha$', ha='center', va='bottom', fontsize=5.5, color=COLORS['green'])
    ax3.text(0.725, 0.72, '$E_0, E_\\infty$', ha='center', va='bottom', fontsize=5.5, color=COLORS['orange'])

    ax3.set_xlim([0, 1])
    ax3.set_ylim([0.1, 0.85])
    ax3.set_title('PEKAN Architecture', fontsize=7, fontweight='bold', pad=3)
    add_panel_label(ax3, 'c', x=-0.02)

    plt.tight_layout()
    save_figure(fig, 'Figure4_method_summary', output_dir)
    plt.close(fig)

    return fig


def main():
    pkg_root = Path(__file__).resolve().parents[2]
    data_dir = pkg_root / 'results_paper'
    output_dir = pkg_root / 'figures_out'

    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")

    if not os.path.exists(data_dir):
        print(f"\n  ERROR: data directory not found: {data_dir}")
        print("  Run the examples first: python src/run_all_examples.py")
        return

    output_dir.mkdir(exist_ok=True)
    setup_nature_style()

    generate_figure1(data_dir, output_dir)
    generate_figure2(data_dir, output_dir)
    generate_figure3(data_dir, output_dir)
    generate_figure4(data_dir, output_dir)


if __name__ == "__main__":
    main()
