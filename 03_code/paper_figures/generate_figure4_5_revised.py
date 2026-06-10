"""Performance-comparison figure (Figure13_performance_comparison): per-example
R^2 and Standard KAN vs PEKAN RMSE, plus a standalone PEKAN architecture sketch.
Needs the example results in results_paper/ (run 03_code/src/run_all_examples.py first).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))
_PKG = _HERE.parents[2]
DATA_DIR = _PKG / "results_paper"
OUTPUT_DIR = _PKG / "figures_out"

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
    'dark_blue': '#005B96',
    'dark_green': '#017351',
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
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.frameon': False,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
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
            fontsize=fontsize, fontweight='bold', va='top', ha='left')

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


def generate_figure4_revised(data_dir, output_dir):
    """R^2 per example and Standard KAN vs PEKAN RMSE."""
    print("\nGenerating Figure13_performance_comparison...")

    data1 = np.load(os.path.join(data_dir, 'example_01/data/results.npz'), allow_pickle=True)
    data2 = np.load(os.path.join(data_dir, 'example_02/data/results.npz'), allow_pickle=True)
    data3 = np.load(os.path.join(data_dir, 'example_03/data/results.npz'), allow_pickle=True)

    errors1 = data1['errors_pi'].item()
    errors2 = data2['errors_pi'].item()
    errors_kan = data3['errors_kan'].item()
    errors_pekan = data3['errors_pekan'].item()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    # (a) R^2 per example, y axis truncated to the high-accuracy range
    ax1 = fig.add_subplot(gs[0])

    examples = ['Ex.1\n(Relaxation)', 'Ex.2\n(General)', 'Ex.3\n(Inversion)']
    r2_values = [errors1['r2'], errors2['r2'], errors_pekan['r2']]
    colors = [COLORS['blue'], COLORS['green'], COLORS['purple']]

    x = np.arange(len(examples))

    bars = ax1.bar(x, r2_values, width=0.55, color=colors,
                   edgecolor='black', linewidth=0.5, zorder=3)

    ax1.set_ylim([0.996, 1.0005])
    ax1.set_ylabel('$R^2$ coefficient', fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(examples, fontsize=6)

    for bar, r2 in zip(bars, r2_values):
        height = bar.get_height()
        ax1.annotate(f'{r2:.6f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6, fontweight='bold')

    ax1.axhline(y=1.0, color=COLORS['gray'], linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.axhline(y=0.999, color=COLORS['gray'], linestyle=':', linewidth=0.3, alpha=0.5)

    ax1.text(0.02, 0.98, 'Higher is better', transform=ax1.transAxes,
             fontsize=5, style='italic', color=COLORS['gray'], va='top')

    add_panel_label(ax1, 'a')

    # (b) RMSE, log scale
    ax2 = fig.add_subplot(gs[1])

    methods = ['Standard KAN', 'PEKAN']
    rmse_values = [errors_kan['rmse'], errors_pekan['rmse']]

    x = np.arange(len(methods))

    bars_rmse = ax2.bar(x, rmse_values, width=0.5,
                        color=[COLORS['orange'], COLORS['green']],
                        edgecolor='black', linewidth=0.5, zorder=3)

    ax2.set_yscale('log')
    ax2.set_ylabel('RMSE (Pa)', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=7)
    ax2.set_ylim([1e-2, 1e0])

    for bar, rmse in zip(bars_rmse, rmse_values):
        height = bar.get_height()
        ax2.annotate(f'{rmse:.2e}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6, fontweight='bold')

    improvement = rmse_values[0] / rmse_values[1]

    arrow_y = np.sqrt(rmse_values[0] * rmse_values[1])
    ax2.annotate('',
                xy=(1, rmse_values[1] * 1.5),
                xytext=(0, rmse_values[0] * 0.7),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark_blue'],
                               lw=1.5, connectionstyle='arc3,rad=-0.2'))

    ax2.text(0.5, arrow_y * 0.8, f'{improvement:.1f}× better',
             ha='center', va='center', fontsize=7, fontweight='bold',
             color=COLORS['dark_blue'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['dark_blue'], alpha=0.9))

    ax2.text(0.98, 0.98, 'Lower is better', transform=ax2.transAxes,
             fontsize=5, style='italic', color=COLORS['gray'], va='top', ha='right')

    add_panel_label(ax2, 'b')

    plt.tight_layout()
    save_figure(fig, 'Figure13_performance_comparison', output_dir)
    plt.close(fig)

    return fig


def generate_figure5_pekan_architecture(output_dir):
    """Standalone PEKAN architecture diagram (nodes, edge activations, trainable params)."""
    print("\nGenerating Figure5_PEKAN_architecture...")

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 1.1, DOUBLE_COL * 0.45))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.5, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    input_x = 0.5
    input_y = 1.5
    layer1_x = 3.0   # PowerTransformLayer
    layer1_y = 1.5
    layer2_x = 5.5   # MittagLefflerLayer
    layer2_y = 1.5
    layer3_x = 8.0   # LinearScaleLayer
    layer3_y = 1.5
    output_x = 10.0
    output_y = 1.5

    node_radius = 0.35

    input_circle = Circle((input_x, input_y), node_radius,
                          facecolor=COLORS['blue'], edgecolor='black',
                          linewidth=1.5, zorder=10)
    ax.add_patch(input_circle)
    ax.text(input_x, input_y, '$t$', ha='center', va='center',
            fontsize=12, color='white', fontweight='bold', zorder=11)
    ax.text(input_x, input_y - node_radius - 0.3, 'Input',
            ha='center', va='top', fontsize=7, color=COLORS['gray'])

    node1_circle = Circle((layer1_x, layer1_y), node_radius * 0.9,
                          facecolor='white', edgecolor=COLORS['cyan'],
                          linewidth=2, zorder=10)
    ax.add_patch(node1_circle)
    ax.text(layer1_x, layer1_y, r'$\xi$', ha='center', va='center',
            fontsize=11, color=COLORS['cyan'], fontweight='bold', zorder=11)

    node2_circle = Circle((layer2_x, layer2_y), node_radius * 0.9,
                          facecolor='white', edgecolor=COLORS['green'],
                          linewidth=2, zorder=10)
    ax.add_patch(node2_circle)
    ax.text(layer2_x, layer2_y, '$y$', ha='center', va='center',
            fontsize=11, color=COLORS['green'], fontweight='bold', zorder=11)

    node3_circle = Circle((layer3_x, layer3_y), node_radius * 0.9,
                          facecolor='white', edgecolor=COLORS['orange'],
                          linewidth=2, zorder=10)
    ax.add_patch(node3_circle)
    ax.text(layer3_x, layer3_y, '$\\sigma$', ha='center', va='center',
            fontsize=11, color=COLORS['orange'], fontweight='bold', zorder=11)

    output_circle = Circle((output_x, output_y), node_radius,
                          facecolor=COLORS['red'], edgecolor='black',
                          linewidth=1.5, zorder=10)
    ax.add_patch(output_circle)
    ax.text(output_x, output_y, '$\\sigma$', ha='center', va='center',
            fontsize=12, color='white', fontweight='bold', zorder=11)
    ax.text(output_x, output_y - node_radius - 0.3, 'Output',
            ha='center', va='top', fontsize=7, color=COLORS['gray'])

    def draw_edge_with_function(ax, x1, y1, x2, y2, func_label, param_label,
                                color, func_y_offset=0.8):
        mid_x = (x1 + x2) / 2

        ax.plot([x1 + node_radius, mid_x - 0.4], [y1, y1],
                color=color, linewidth=1.5, zorder=5)
        ax.plot([mid_x + 0.4, x2 - node_radius], [y2, y2],
                color=color, linewidth=1.5, zorder=5)

        box = FancyBboxPatch((mid_x - 0.45, y1 - 0.35), 0.9, 0.7,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor='white', edgecolor=color,
                            linewidth=1.5, zorder=8)
        ax.add_patch(box)

        # small spline-like curve as the activation glyph
        curve_x = np.linspace(mid_x - 0.3, mid_x + 0.3, 50)
        curve_y = y1 + 0.15 * np.sin(np.pi * (curve_x - mid_x + 0.3) / 0.6)
        ax.plot(curve_x, curve_y, color=color, linewidth=1.2, zorder=9)

        ax.text(mid_x, y1 + func_y_offset, func_label,
                ha='center', va='bottom', fontsize=8, color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor='none', alpha=0.9))

        if param_label:
            ax.text(mid_x, y1 - 0.55, param_label,
                    ha='center', va='top', fontsize=7, color=color,
                    style='italic')

        ax.annotate('', xy=(x2 - node_radius, y2),
                   xytext=(x2 - node_radius - 0.3, y2),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    draw_edge_with_function(ax, input_x, input_y, layer1_x, layer1_y,
                           r'$\phi_1$', r'$\tau, \alpha$',
                           COLORS['cyan'])
    draw_edge_with_function(ax, layer1_x, layer1_y, layer2_x, layer2_y,
                           r'$\phi_2$', r'$\alpha$',
                           COLORS['green'])
    draw_edge_with_function(ax, layer2_x, layer2_y, layer3_x, layer3_y,
                           r'$\phi_3$', r'$E_0, E_\infty$',
                           COLORS['orange'])

    ax.plot([layer3_x + node_radius * 0.9, output_x - node_radius],
            [layer3_y, output_y],
            color=COLORS['gray'], linewidth=1.5, linestyle='-', zorder=5)
    ax.annotate('', xy=(output_x - node_radius, output_y),
               xytext=(output_x - node_radius - 0.3, output_y),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.2))

    formula_y = -0.7
    ax.text(1.75, formula_y,
            r'$\xi = -\left(\frac{t}{\tau}\right)^\alpha$',
            ha='center', va='center', fontsize=9, color=COLORS['cyan'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['cyan'],
                     edgecolor='none', alpha=0.15))
    ax.text(4.25, formula_y,
            r'$y = E_\alpha(\xi)$',
            ha='center', va='center', fontsize=9, color=COLORS['green'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['green'],
                     edgecolor='none', alpha=0.15))
    ax.text(7.0, formula_y,
            r'$\sigma = E_\infty + (E_0 - E_\infty) y$',
            ha='center', va='center', fontsize=9, color=COLORS['orange'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['orange'],
                     edgecolor='none', alpha=0.15))

    layer_y = 3.2
    ax.text(1.75, layer_y, 'Layer 1\nPowerTransform', ha='center', va='bottom',
            fontsize=7, color=COLORS['cyan'], fontweight='bold')
    ax.text(4.25, layer_y, 'Layer 2\nMittag-Leffler', ha='center', va='bottom',
            fontsize=7, color=COLORS['green'], fontweight='bold')
    ax.text(6.75, layer_y, 'Layer 3\nLinearScale', ha='center', va='bottom',
            fontsize=7, color=COLORS['orange'], fontweight='bold')

    ax.text(5.25, 3.8, 'Physics-Encoded KAN (PEKAN) Architecture',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    param_box_y = -1.2
    param_text = (r'Trainable Parameters:  '
                  r'$\tau$ (relaxation time)  |  '
                  r'$\alpha$ (fractional order)  |  '
                  r'$E_0$ (instantaneous modulus)  |  '
                  r'$E_\infty$ (equilibrium modulus)')
    ax.text(5.25, param_box_y, param_text,
            ha='center', va='top', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light_gray'],
                     edgecolor='none', alpha=0.5))

    # manual margins; tight_layout fights the aspect-equal axis
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.15)
    save_figure(fig, 'Figure5_PEKAN_architecture', output_dir)
    plt.close(fig)

    return fig


def main():
    data_dir = str(DATA_DIR)
    output_dir = str(OUTPUT_DIR)

    if not os.path.exists(data_dir):
        print(f"ERROR: results not found at {data_dir}")
        print("Run the examples first: python src/run_all_examples.py")
        return

    setup_nature_style()
    generate_figure4_revised(data_dir, output_dir)
    generate_figure5_pekan_architecture(output_dir)
    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
