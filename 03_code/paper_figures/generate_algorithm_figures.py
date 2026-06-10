"""Pseudocode figures for the PIKAN and PEKAN training algorithms."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'mathtext.fontset': 'cm',
        'savefig.dpi': 600,
        'savefig.pad_inches': 0.1,
        'figure.facecolor': 'white',
    })

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


def draw_algorithm_three_line(ax, title, lines, line_height=0.038):
    """Booktabs-style pseudocode: thick top rule, thin rule under the title,
    thick bottom rule drawn after the content."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.axhline(y=0.97, xmin=0.02, xmax=0.98, color='black', linewidth=2.0)
    ax.axhline(y=0.89, xmin=0.02, xmax=0.98, color='black', linewidth=0.8)

    ax.text(0.5, 0.94, title, transform=ax.transAxes, fontsize=11,
           ha='center', va='center', fontweight='bold')

    y = 0.86
    base_x = 0.03
    indent_width = 0.025
    
    for text, style, indent in lines:
        if style == 'space':
            y -= line_height * 0.5
            continue
        
        x = base_x + indent * indent_width
        
        if style == 'require':
            ax.text(x, y, 'Require:', fontweight='bold', fontsize=9,
                   transform=ax.transAxes, ha='left', va='top')
            ax.text(x + 0.10, y, text, fontsize=9,
                   transform=ax.transAxes, ha='left', va='top')
            y -= line_height * 1.4
            continue
        elif style == 'ensure':
            ax.text(x, y, 'Ensure:', fontweight='bold', fontsize=9,
                   transform=ax.transAxes, ha='left', va='top')
            ax.text(x + 0.085, y, text, fontsize=9,
                   transform=ax.transAxes, ha='left', va='top')
        elif style in ['for', 'endfor', 'return']:
            ax.text(x, y, text, fontsize=9, fontweight='bold',
                   transform=ax.transAxes, ha='left', va='top')
        else:
            ax.text(x, y, text, fontsize=9,
                   transform=ax.transAxes, ha='left', va='top')
        
        y -= line_height

    bottom_line_y = y - line_height * 0.8
    ax.axhline(y=bottom_line_y, xmin=0.02, xmax=0.98, color='black', linewidth=2.0)


def generate_figure6_pikan_algorithm(output_dir):
    print("\n[Figure 2] PIKAN training algorithm...")

    fig, ax = plt.subplots(figsize=(5.6, 5.0))

    title = "Algorithm 1  PIKAN Training Algorithm"
    
    lines = [
        (r"Time grid $\{t_n\}_{n=0}^{N}$, parameters $(\tau, \alpha, E_0, E_\infty)$, strain $\varepsilon(t)$", 'require', 0),
        (r"Trained KAN parameters $\theta^*$", 'ensure', 0),
        ("", 'space', 0),
        (r"1:  Compute normalized input: $t_{\mathrm{in},n} \leftarrow 2t_n/T - 1$", 'line', 0),
        (r"2:  Precompute L1 kernel: $\mathbf{w} \leftarrow \mathrm{CaputoL1Kernel}(N, \alpha)$", 'line', 0),
        (r"3:  Precompute fractional derivative: $\{D^\alpha\varepsilon(t_n)\}$", 'line', 0),
        (r"4:  Initialize KAN network parameters $\theta$", 'line', 0),
        ("", 'space', 0),
        (r"5:  for epoch = 1 to $N_{\mathrm{Adam}}$ do                          $\triangleright$ Stage 1: Adam", 'for', 0),
        (r"6:      Forward: $\hat{\sigma}_n \leftarrow \mathrm{KAN}_\theta(t_{\mathrm{in},n})$", 'line', 1),
        (r"7:      Fractional derivative: $D^\alpha\hat{\sigma} \leftarrow \mathrm{Conv1D}(\Delta\hat{\sigma}, \mathbf{w})$", 'line', 1),
        (r"8:      Residual: $r_n \leftarrow \hat{\sigma}_n + \tau^\alpha D^\alpha\hat{\sigma}_n - (E_\infty\varepsilon_n + E_0\tau^\alpha D^\alpha\varepsilon_n)$", 'line', 1),
        (r"9:      Loss: $\mathcal{L} \leftarrow \frac{1}{N}\sum r_n^2 + \lambda(\hat{\sigma}_0 - E_0\varepsilon_0)^2$", 'line', 1),
        (r"10:    Update: $\theta \leftarrow \mathrm{Adam}(\theta, \nabla_\theta\mathcal{L})$", 'line', 1),
        (r"11: end for", 'endfor', 0),
        ("", 'space', 0),
        (r"12: for epoch = 1 to $N_{\mathrm{LBFGS}}$ do                       $\triangleright$ Stage 2: L-BFGS", 'for', 0),
        (r"13:    Compute $\mathcal{L}$ and $\nabla_\theta\mathcal{L}$ via closure", 'line', 1),
        (r"14:    Update: $\theta \leftarrow \mathrm{LBFGS}(\theta, \mathcal{L}, \nabla_\theta\mathcal{L})$", 'line', 1),
        (r"15: end for", 'endfor', 0),
        ("", 'space', 0),
        (r"16: return $\theta^* \leftarrow \theta$", 'return', 0),
    ]
    
    draw_algorithm_three_line(ax, title, lines)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save_figure(fig, 'Figure2_PIKAN_algorithm', output_dir)
    plt.close(fig)


def generate_figure7_pekan_algorithm(output_dir):
    print("\n[Figure 3] PEKAN training algorithm...")

    fig, ax = plt.subplots(figsize=(5.6, 7.0))

    title = "Algorithm 2  PEKAN Training Algorithm"
    
    lines = [
        (r"Observed data $\{(t_n, \sigma_n^{\mathrm{obs}})\}_{n=1}^{N}$, initial guess $\theta_0$", 'require', 0),
        (r"Inverted parameters $\theta^* = (\tau^*, \alpha^*, E_0^*, E_\infty^*)$", 'ensure', 0),
        ("", 'space', 0),
        (r"1:  Transform to unconstrained variables:", 'line', 0),
        (r"      $\tilde{\tau} \leftarrow \ln(\tau_0)$,  $\tilde{\alpha} \leftarrow \mathrm{logit}(\alpha_0)$,  $\tilde{E}_\infty \leftarrow \ln(E_{\infty,0})$,  $\widetilde{\Delta E} \leftarrow \ln(E_{0,0} - E_{\infty,0})$", 'line', 1),
        (r"2:  Initialize PEKAN with physics-encoded layers", 'line', 0),
        ("", 'space', 0),
        (r"3:  for epoch = 1 to $N_1$ do                                    $\triangleright$ Stage 1: Adam ($\eta$=0.05)", 'for', 0),
        (r"4:      Forward: $\hat{\sigma}_n \leftarrow \mathrm{PEKAN}(t_n; \theta(\tilde{\theta}))$", 'line', 1),
        (r"5:      Loss: $\mathcal{L} \leftarrow \frac{1}{N}\sum(\hat{\sigma}_n - \sigma_n^{\mathrm{obs}})^2$", 'line', 1),
        (r"6:      Update: $\tilde{\theta} \leftarrow \mathrm{Adam}(\tilde{\theta}, \nabla_{\tilde{\theta}}\mathcal{L}, \eta=0.05)$", 'line', 1),
        (r"7:  end for", 'endfor', 0),
        ("", 'space', 0),
        (r"8:  for epoch = 1 to $N_2$ do                                   $\triangleright$ Stage 2: Adam ($\eta$=0.005)", 'for', 0),
        (r"9:      Forward and loss (same as above)", 'line', 1),
        (r"10:    Update: $\tilde{\theta} \leftarrow \mathrm{Adam}(\tilde{\theta}, \nabla_{\tilde{\theta}}\mathcal{L}, \eta=0.005)$", 'line', 1),
        (r"11: end for", 'endfor', 0),
        ("", 'space', 0),
        (r"12: for epoch = 1 to $N_3$ do                                    $\triangleright$ Stage 3: L-BFGS", 'for', 0),
        (r"13:    Compute $\mathcal{L}$ and $\nabla_{\tilde{\theta}}\mathcal{L}$ via closure", 'line', 1),
        (r"14:    Update: $\tilde{\theta} \leftarrow \mathrm{LBFGS}(\tilde{\theta}, \mathcal{L}, \nabla_{\tilde{\theta}}\mathcal{L})$", 'line', 1),
        (r"15: end for", 'endfor', 0),
        ("", 'space', 0),
        (r"16: Convert: $\theta^* \leftarrow (\exp(\tilde{\tau}), \mathrm{sigmoid}(\tilde{\alpha}), \exp(\tilde{E}_\infty)+\exp(\widetilde{\Delta E}), \exp(\tilde{E}_\infty))$", 'line', 0),
        (r"17: return $\theta^*$", 'return', 0),
    ]
    
    draw_algorithm_three_line(ax, title, lines, line_height=0.032)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save_figure(fig, 'Figure3_PEKAN_algorithm', output_dir)
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parents[2] / 'figures_out'
    output_dir.mkdir(exist_ok=True)
    print(f"  Output directory: {output_dir}")

    setup_style()

    generate_figure6_pikan_algorithm(output_dir)
    generate_figure7_pekan_algorithm(output_dir)


if __name__ == "__main__":
    main()
