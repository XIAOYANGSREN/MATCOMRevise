"""Network-architecture schematics in the KAN-paper style:
PIKAN vs PEKAN. Self-contained drawing, no training."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from pathlib import Path

_HERE = Path(__file__).resolve()

COLORS = {
    'blue': '#0173B2',
    'red': '#D55E00',
    'green': '#029E73',
    'orange': '#DE8F05',
    'purple': '#CC78BC',
    'cyan': '#56B4E9',
    'gray': '#949494',
    'light_gray': '#CCCCCC',
    'black': '#000000',
    'white': '#FFFFFF',
}

def setup_style():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
    })

SINGLE_COL = 89 / 25.4
DOUBLE_COL = 183 / 25.4

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


def draw_node(ax, x, y, radius=0.08, color='black'):
    circle = Circle((x, y), radius, facecolor=color, edgecolor=color,
                    linewidth=1, zorder=10)
    ax.add_patch(circle)

def draw_activation_box(ax, x, y, width=0.25, height=0.2, func_type='spline',
                        alpha=1.0, color='black'):
    """Small box on an edge with a sketched activation curve inside."""
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="square,pad=0",
                          facecolor='white', edgecolor=color,
                          linewidth=0.8, alpha=alpha, zorder=8)
    ax.add_patch(rect)

    t = np.linspace(-0.4, 0.4, 50)

    if func_type == 'spline':
        curve = 0.3 * np.sin(np.pi * t / 0.4) * np.exp(-t**2 / 0.2)
    elif func_type == 'parabola':
        curve = 0.8 * (t / 0.4) ** 2 - 0.4
    elif func_type == 'sin':
        curve = 0.35 * np.sin(2.5 * np.pi * t / 0.4)
    elif func_type == 'exp':
        curve = 0.35 * (np.exp(2 * t) - 1) / (np.exp(0.8) - 1) - 0.1
    elif func_type == 'linear':
        curve = 0.8 * t
    elif func_type == 'mittag_leffler':
        curve = 0.35 * np.exp(-np.abs(t) * 3)
    elif func_type == 'power':
        curve = 0.35 * np.sign(t) * np.abs(t / 0.4) ** 0.5
    else:
        curve = 0.3 * np.sin(np.pi * t / 0.4)

    plot_x = x + t * width / 0.8
    plot_y = y + curve * height / 0.8

    ax.plot(plot_x, plot_y, color=color, linewidth=1.0, alpha=alpha, zorder=9)

def draw_edge(ax, x1, y1, x2, y2, color='black', linewidth=0.8, alpha=1.0):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth,
            alpha=alpha, zorder=5)


def generate_figure5_pekan_kan_style(output_dir):
    """Standalone PEKAN schematic: 1-1-1-1 chain, bottom-up."""
    print("\n[Figure 5] PEKAN architecture (KAN paper style)...")

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.8))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    node_x = 0
    layer_y = [0, 1.2, 2.4, 3.6]

    node_radius = 0.1
    box_width = 0.35
    box_height = 0.28

    for i, y in enumerate(layer_y):
        draw_node(ax, node_x, y, radius=node_radius, color='black')

    # t -> xi (PowerTransform)
    mid_y1 = (layer_y[0] + layer_y[1]) / 2
    draw_edge(ax, node_x, layer_y[0] + node_radius, node_x, mid_y1 - box_height/2 - 0.05)
    draw_activation_box(ax, node_x, mid_y1, box_width, box_height, func_type='power')
    draw_edge(ax, node_x, mid_y1 + box_height/2 + 0.05, node_x, layer_y[1] - node_radius)

    # xi -> y (Mittag-Leffler)
    mid_y2 = (layer_y[1] + layer_y[2]) / 2
    draw_edge(ax, node_x, layer_y[1] + node_radius, node_x, mid_y2 - box_height/2 - 0.05)
    draw_activation_box(ax, node_x, mid_y2, box_width, box_height, func_type='mittag_leffler')
    draw_edge(ax, node_x, mid_y2 + box_height/2 + 0.05, node_x, layer_y[2] - node_radius)

    # y -> sigma (LinearScale)
    mid_y3 = (layer_y[2] + layer_y[3]) / 2
    draw_edge(ax, node_x, layer_y[2] + node_radius, node_x, mid_y3 - box_height/2 - 0.05)
    draw_activation_box(ax, node_x, mid_y3, box_width, box_height, func_type='linear')
    draw_edge(ax, node_x, mid_y3 + box_height/2 + 0.05, node_x, layer_y[3] - node_radius)

    label_offset = 0.35
    ax.text(node_x + label_offset, layer_y[0], '$t$', fontsize=14,
            ha='left', va='center', fontweight='bold')
    ax.text(node_x + label_offset, layer_y[1], r'$\xi$', fontsize=14,
            ha='left', va='center')
    ax.text(node_x + label_offset, layer_y[2], '$y$', fontsize=14,
            ha='left', va='center')
    ax.text(node_x + label_offset, layer_y[3], r'$\sigma$', fontsize=14,
            ha='left', va='center', fontweight='bold')

    formula_offset = -0.55
    ax.text(formula_offset, mid_y1, r'$\phi_1$: $-\left(\frac{t}{\tau}\right)^\alpha$',
            fontsize=9, ha='right', va='center', color=COLORS['cyan'])
    ax.text(formula_offset, mid_y2, r'$\phi_2$: $E_\alpha(\xi)$',
            fontsize=9, ha='right', va='center', color=COLORS['green'])
    ax.text(formula_offset, mid_y3, r'$\phi_3$: $E_\infty + (E_0-E_\infty)y$',
            fontsize=9, ha='right', va='center', color=COLORS['orange'])

    ax.text(0, 4.3, 'PEKAN', fontsize=14, ha='center', va='bottom', fontweight='bold')
    ax.text(0, 4.0, r'$\sigma(t) = E_\infty + (E_0-E_\infty)E_\alpha(-(t/\tau)^\alpha)$',
            fontsize=9, ha='center', va='bottom')

    ax.text(0, -0.35, r'Trainable: $\tau, \alpha, E_0, E_\infty$',
            fontsize=8, ha='center', va='top', style='italic', color=COLORS['gray'])

    plt.tight_layout()
    save_figure(fig, 'Figure5_PEKAN_architecture', output_dir)
    plt.close(fig)

    return fig


def generate_figure6_pikan_kan_style(output_dir):
    """Standalone PIKAN schematic: [1, 10, 1] with B-spline boxes on edges."""
    print("\n[Figure 6] PIKAN architecture (KAN paper style)...")

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.7, DOUBLE_COL * 0.55))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    n_hidden = 10

    input_y = 0
    hidden_y = 1.5
    output_y = 3.0

    input_x = 0

    hidden_width = 4.0
    hidden_x = np.linspace(-hidden_width/2, hidden_width/2, n_hidden)

    output_x = 0

    node_radius = 0.08
    box_width = 0.22
    box_height = 0.18

    func_types = ['spline', 'parabola', 'sin', 'exp', 'spline',
                  'parabola', 'sin', 'exp', 'spline', 'parabola']

    for i, hx in enumerate(hidden_x):
        mid_x = (input_x + hx) / 2
        mid_y = (input_y + hidden_y) / 2

        # random transparency mimics varying edge weights
        alpha = 0.5 + 0.5 * np.random.random()

        draw_edge(ax, input_x, input_y + node_radius, mid_x, mid_y - box_height/2 - 0.03,
                 alpha=alpha)
        draw_activation_box(ax, mid_x, mid_y, box_width, box_height,
                           func_type=func_types[i % len(func_types)], alpha=alpha)
        draw_edge(ax, mid_x, mid_y + box_height/2 + 0.03, hx, hidden_y - node_radius,
                 alpha=alpha)

    for i, hx in enumerate(hidden_x):
        mid_x = (hx + output_x) / 2
        mid_y = (hidden_y + output_y) / 2

        alpha = 0.5 + 0.5 * np.random.random()

        draw_edge(ax, hx, hidden_y + node_radius, mid_x, mid_y - box_height/2 - 0.03,
                 alpha=alpha)
        draw_activation_box(ax, mid_x, mid_y, box_width, box_height,
                           func_type=func_types[(i+3) % len(func_types)], alpha=alpha)
        draw_edge(ax, mid_x, mid_y + box_height/2 + 0.03, output_x, output_y - node_radius,
                 alpha=alpha)

    draw_node(ax, input_x, input_y, radius=node_radius, color='black')
    for hx in hidden_x:
        draw_node(ax, hx, hidden_y, radius=node_radius, color='black')
    draw_node(ax, output_x, output_y, radius=node_radius, color='black')

    ax.text(input_x, input_y - 0.25, '$t$', fontsize=14, ha='center', va='top',
            fontweight='bold')
    ax.text(output_x, output_y + 0.25, r'$\hat{\sigma}(t)$', fontsize=14,
            ha='center', va='bottom', fontweight='bold')

    ax.text(-2.3, input_y, 'Input\n(1)', fontsize=8, ha='center', va='center',
            color=COLORS['gray'])
    ax.text(-2.3, hidden_y, 'Hidden\n(10)', fontsize=8, ha='center', va='center',
            color=COLORS['gray'])
    ax.text(-2.3, output_y, 'Output\n(1)', fontsize=8, ha='center', va='center',
            color=COLORS['gray'])

    ax.text(0, 3.4, 'PIKAN [1, 10, 1]', fontsize=12, ha='center', va='bottom',
            fontweight='bold')

    ax.text(0, -0.4, 'B-spline activation functions on edges (learnable)',
            fontsize=8, ha='center', va='top', style='italic', color=COLORS['gray'])

    plt.tight_layout()
    save_figure(fig, 'Figure6_PIKAN_architecture', output_dir)
    plt.close(fig)

    return fig


def generate_combined_architecture_figure(output_dir):
    """Figure 1: PIKAN (a) and PEKAN (b) side by side."""
    print("\n[Combined] PIKAN & PEKAN architecture comparison...")

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
    # seeded so the exported schematic is identical between runs
    rng = np.random.default_rng(0)

    ax1 = fig.add_axes([0.02, 0.05, 0.48, 0.88])  # PIKAN
    ax2 = fig.add_axes([0.55, 0.05, 0.43, 0.88])  # PEKAN

    ax1.set_xlim(-2.1, 2.1)
    ax1.set_ylim(-0.45, 3.75)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # schematic only: 6 of the 10 hidden nodes
    n_hidden = 6

    input_y = 0
    hidden_y = 1.45
    output_y = 3.05
    input_x = 0
    hidden_width = 3.2
    hidden_x = np.linspace(-hidden_width/2, hidden_width/2, n_hidden)
    output_x = 0

    node_radius = 0.07
    box_width = 0.20
    box_height = 0.16

    func_types = ['spline', 'parabola', 'sin', 'exp', 'spline', 'parabola', 'sin', 'exp']

    for i, hx in enumerate(hidden_x):
        mid_x = (input_x + hx) / 2
        mid_y = (input_y + hidden_y) / 2
        alpha = 0.6 + 0.4 * rng.random()

        draw_edge(ax1, input_x, input_y + node_radius, mid_x, mid_y - box_height/2 - 0.02, alpha=alpha)
        draw_activation_box(ax1, mid_x, mid_y, box_width, box_height,
                           func_type=func_types[i], alpha=alpha)
        draw_edge(ax1, mid_x, mid_y + box_height/2 + 0.02, hx, hidden_y - node_radius, alpha=alpha)

    for i, hx in enumerate(hidden_x):
        mid_x = (hx + output_x) / 2
        mid_y = (hidden_y + output_y) / 2
        alpha = 0.6 + 0.4 * rng.random()

        draw_edge(ax1, hx, hidden_y + node_radius, mid_x, mid_y - box_height/2 - 0.02, alpha=alpha)
        draw_activation_box(ax1, mid_x, mid_y, box_width, box_height,
                           func_type=func_types[(i+2) % len(func_types)], alpha=alpha)
        draw_edge(ax1, mid_x, mid_y + box_height/2 + 0.02, output_x, output_y - node_radius, alpha=alpha)

    draw_node(ax1, input_x, input_y, radius=node_radius)
    for hx in hidden_x:
        draw_node(ax1, hx, hidden_y, radius=node_radius)
    draw_node(ax1, output_x, output_y, radius=node_radius)

    ax1.text(input_x, input_y - 0.2, '$t$', fontsize=12, ha='center', va='top', fontweight='bold')
    ax1.text(output_x, output_y + 0.22, r'$\hat{\sigma}$', fontsize=12, ha='center', va='bottom', fontweight='bold')
    ax1.text(0, 3.55, '(a) PIKAN', fontsize=11, ha='center', va='bottom', fontweight='bold')
    ax1.text(0, -0.33, 'Learnable B-spline activations', fontsize=7, ha='center', va='top',
             style='italic', color=COLORS['gray'])

    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.45, 4.05)
    ax2.set_aspect('equal')
    ax2.axis('off')

    node_x = 0
    layer_y = [0, 1.0, 2.0, 3.0]

    node_radius = 0.08
    box_width = 0.30
    box_height = 0.22

    for y in layer_y:
        draw_node(ax2, node_x, y, radius=node_radius)

    for i in range(3):
        mid_y = (layer_y[i] + layer_y[i+1]) / 2
        draw_edge(ax2, node_x, layer_y[i] + node_radius, node_x, mid_y - box_height/2 - 0.03)

        if i == 0:
            func = 'power'
        elif i == 1:
            func = 'mittag_leffler'
        else:
            func = 'linear'
        color = COLORS['black']  # black-and-white panel

        rect = FancyBboxPatch((node_x - box_width/2, mid_y - box_height/2), box_width, box_height,
                              boxstyle="square,pad=0",
                              facecolor='white', edgecolor=color,
                              linewidth=1.2, zorder=8)
        ax2.add_patch(rect)

        t = np.linspace(-0.4, 0.4, 50)
        if func == 'power':
            curve = 0.35 * np.sign(t) * np.abs(t / 0.4) ** 0.5
        elif func == 'mittag_leffler':
            curve = 0.35 * np.exp(-np.abs(t) * 3)
        else:
            curve = 0.8 * t

        plot_x = node_x + t * box_width / 0.8
        plot_y = mid_y + curve * box_height / 0.8
        ax2.plot(plot_x, plot_y, color=color, linewidth=1.2, zorder=9)

        draw_edge(ax2, node_x, mid_y + box_height/2 + 0.03, node_x, layer_y[i+1] - node_radius)

    label_offset = 0.25
    ax2.text(node_x + label_offset, layer_y[0], '$t$', fontsize=12, ha='left', va='center', fontweight='bold')
    ax2.text(node_x + label_offset, layer_y[1], r'$\xi$', fontsize=11, ha='left', va='center')
    ax2.text(node_x + label_offset, layer_y[2], '$y$', fontsize=11, ha='left', va='center')
    ax2.text(node_x + label_offset, layer_y[3], r'$\sigma$', fontsize=12, ha='left', va='center', fontweight='bold')

    formula_offset = -0.4
    ax2.text(formula_offset, 0.5, r'$-(t/\tau)^\alpha$', fontsize=7, ha='right', va='center', color=COLORS['black'])
    ax2.text(formula_offset, 1.5, r'$E_\alpha(\xi)$', fontsize=7, ha='right', va='center', color=COLORS['black'])
    ax2.text(formula_offset, 2.5, r'$E_\infty+(E_0-E_\infty)y$', fontsize=6, ha='right', va='center', color=COLORS['black'])

    ax2.text(0, 3.75, '(b) PEKAN', fontsize=11, ha='center', va='bottom', fontweight='bold')
    ax2.text(0, 3.52, r'$\tau, \alpha, E_0, E_\infty$', fontsize=8, ha='center', va='bottom', color=COLORS['gray'])
    ax2.text(0, -0.33, 'Physics-encoded activations', fontsize=7, ha='center', va='top',
             style='italic', color=COLORS['gray'])

    save_figure(fig, 'Figure1_network_architectures', output_dir)
    plt.close(fig)

    return fig


def main():
    out = _HERE.parents[2] / "figures_out"
    out.mkdir(exist_ok=True)
    output_dir = str(out)

    print(f"  Output directory: {output_dir}")

    setup_style()

    generate_figure5_pekan_kan_style(output_dir)
    generate_figure6_pikan_kan_style(output_dir)
    generate_combined_architecture_figure(output_dir)


if __name__ == "__main__":
    main()
