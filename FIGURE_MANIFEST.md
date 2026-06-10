# Figure manifest

Each manuscript figure and the script that produces it. Scripts write the
manuscript-named file into `figures_out/` at the package root (the data
figures also keep a copy in their own `results/` folder).

Note: the image FILENAMES are historical labels and no longer match the rendered
figure numbers (e.g. manuscript Figure 7 uses the file
`Figure8_noise_robustness.png`). `\includegraphics` keys on the filename, while the
printed number follows the order of `\begin{figure}` in the `.tex`.

| Fig. | Manuscript file | Produced by |
|---|---|---|
| 1 | `Figure1_network_architectures.png` | `03_code/paper_figures/generate_kan_architecture_figures.py` |
| 2 | `Figure2_PIKAN_algorithm.png` | `03_code/paper_figures/generate_algorithm_figures.py` |
| 3 | `Figure3_PEKAN_algorithm.png` | `03_code/paper_figures/generate_algorithm_figures.py` |
| 4 | `Figure4_PIKAN_relaxation.png` | `03_code/paper_figures/generate_manuscript_figures.py` (example_01 result) |
| 5 | `Figure5_PIKAN_general_loading.png` | `03_code/paper_figures/generate_manuscript_figures.py` (example_02 result) |
| 6 | `Figure6_PEKAN_inversion.png` | `03_code/paper_figures/generate_manuscript_figures.py` (example_03 result) |
| 7 | `Figure8_noise_robustness.png` | `04_new_experiments/noise_robustness/noise_sweep.py` |
| 8 | `Figure9_sensitivity.png` | `04_new_experiments/sensitivity/plot_kan_sensitivity.py` |
| 9 | `Figure10_real_data.png` | `04_new_experiments/real_relaxation/fit_real_data.py` |
| 10 | `Figure11_creep.png` | `04_new_experiments/creep/make_creep_figure.py` |
| 11 | `Figure13_prony_mismatch.png` | `04_new_experiments/prony_mismatch/prony_mismatch.py` |
| 12 | `Figure12_arbitrary.png` | `04_new_experiments/arbitrary_loading/arbitrary_loading_inversion.py` |
| 13 | `Figure13_performance_comparison.png` | `03_code/paper_figures/generate_figure4_5_revised.py` (example_03 result) |

Figures 4-6 and 13 require the example results first
(`python 03_code/src/run_all_examples.py`, which writes `results_paper/` at the
package root). The architecture/algorithm scripts (Figures 1-3) are
self-contained schematics. The experiment scripts under `04_new_experiments/`
read only the data folders under `data/`.
