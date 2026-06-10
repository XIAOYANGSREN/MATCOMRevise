"""Journal-style matplotlib helpers (Arial, 89/183 mm columns, 300 dpi)."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

NATURE_COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#D55E00',
    'purple': '#CC78BC',
    'cyan': '#56B4E9',
    'yellow': '#ECE133',
    'gray': '#949494',
    'black': '#000000',
}

NATURE_PALETTE = [
    '#0173B2',
    '#D55E00',
    '#029E73',
    '#DE8F05',
    '#CC78BC',
    '#56B4E9',
    '#949494',
]


def setup_nature_style():
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['legend.fontsize'] = 7

    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['xtick.minor.width'] = 0.3
    plt.rcParams['ytick.minor.width'] = 0.3

    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 1.5
    plt.rcParams['ytick.minor.size'] = 1.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'best'

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.05

    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = False


class NaturePlotStyle:
    """Usage:
        style = NaturePlotStyle()
        fig, ax = style.create_figure(width='single')
        ax.plot(x, y, **style.get_line_style('blue'))
        style.save_figure(fig, 'name', output_dir)
    """

    WIDTHS = {
        'single': 89 / 25.4,
        'medium': 120 / 25.4,
        'double': 183 / 25.4,
    }

    def __init__(self):
        setup_nature_style()
        self.colors = NATURE_COLORS
        self.palette = NATURE_PALETTE
        self.color_index = 0

    def create_figure(self, width='single', height_ratio=0.75,
                      nrows=1, ncols=1, **kwargs):
        fig_width = self.WIDTHS.get(width, self.WIDTHS['single'])
        fig_height = fig_width * height_ratio

        if nrows > 1 or ncols > 1:
            fig_height = fig_width * height_ratio * nrows / ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(fig_width, fig_height),
                                 **kwargs)
        return fig, axes

    def get_color(self, name=None):
        if name is not None:
            return self.colors.get(name, self.colors['blue'])
        color = self.palette[self.color_index % len(self.palette)]
        self.color_index += 1
        return color

    def reset_color_cycle(self):
        self.color_index = 0

    def get_line_style(self, color='blue', linestyle='-', linewidth=1.0,
                       marker=None, markersize=4):
        style = {
            'color': self.colors.get(color, color),
            'linestyle': linestyle,
            'linewidth': linewidth,
        }
        if marker is not None:
            style['marker'] = marker
            style['markersize'] = markersize
            style['markerfacecolor'] = 'white'
            style['markeredgecolor'] = style['color']
            style['markeredgewidth'] = 0.5
        return style

    def get_scatter_style(self, color='blue', size=20, marker='o', alpha=0.7):
        return {
            'c': self.colors.get(color, color),
            's': size,
            'marker': marker,
            'alpha': alpha,
            'edgecolors': 'white',
            'linewidths': 0.3,
        }

    def format_axis(self, ax, xlabel='', ylabel='', title='',
                    xlim=None, ylim=None, legend=True, grid=False):
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontweight='bold', pad=8)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend()
        if grid:
            ax.grid(True, alpha=0.3, linewidth=0.3)

    def add_panel_label(self, ax, label, x=-0.15, y=1.05):
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='bottom', ha='right')

    def save_figure(self, fig, filename, output_dir='./results',
                    formats=['pdf', 'png'], dpi=300):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            filepath = output_path / f"{filename}.{fmt}"
            if fmt == 'png':
                fig.savefig(filepath, dpi=dpi, facecolor='white',
                            bbox_inches='tight', pad_inches=0.05)
            else:
                fig.savefig(filepath, facecolor='white',
                            bbox_inches='tight', pad_inches=0.05)
            print(f"  saved: {filepath}")


def create_error_comparison_plot(style, t, exact, predictions, output_dir,
                                 filename='error_comparison'):
    """Three-panel template: solutions, absolute error, relative error."""
    fig, axes = style.create_figure(width='double', height_ratio=0.4,
                                    nrows=1, ncols=3)

    ax1 = axes[0]
    ax1.plot(t, exact, **style.get_line_style('black', linewidth=1.5),
             label='Exact')
    for i, (name, pred) in enumerate(predictions.items()):
        ax1.plot(t, pred, **style.get_line_style(list(NATURE_COLORS.keys())[i]),
                 label=name, linestyle='--')
    style.format_axis(ax1, xlabel='$t$', ylabel='$u(t)$',
                      title='Solution Comparison', legend=True)
    style.add_panel_label(ax1, 'a')

    ax2 = axes[1]
    for i, (name, pred) in enumerate(predictions.items()):
        error = np.abs(pred - exact)
        ax2.semilogy(t, error, **style.get_line_style(list(NATURE_COLORS.keys())[i]),
                     label=name)
    style.format_axis(ax2, xlabel='$t$', ylabel='Absolute Error',
                      title='Error Distribution', legend=True)
    style.add_panel_label(ax2, 'b')

    ax3 = axes[2]
    for i, (name, pred) in enumerate(predictions.items()):
        rel_error = np.abs(pred - exact) / (np.abs(exact) + 1e-10) * 100
        ax3.plot(t, rel_error, **style.get_line_style(list(NATURE_COLORS.keys())[i]),
                 label=name)
    ax3.axhline(y=1, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    style.format_axis(ax3, xlabel='$t$', ylabel='Relative Error (%)',
                      title='Relative Error', legend=True)
    style.add_panel_label(ax3, 'c')

    plt.tight_layout()
    style.save_figure(fig, filename, output_dir)

    return fig
