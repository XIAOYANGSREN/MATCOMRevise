# MATCOMRevise

Reviewer-facing code package for the revised manuscript:

**A Kolmogorov-Arnold Network-Based Learning Framework for Fractional-Order Viscoelastic Constitutive Models: Forward Solving and Parameter Inversion**

This repository contains the executable source code, environment files, reproduction instructions, and lightweight tabular outputs used to support the revised manuscript submitted to *Mathematics and Computers in Simulation*.

## Repository scope

The repository mirrors the code-oriented portion of the reproducibility package:

- `03_code/`: core scripts for the three main examples and manuscript figure generation.
- `04_new_experiments/`: revision experiments, including convergence, robustness, sensitivity, timing, real-data fitting, Prony mismatch, creep, and arbitrary-loading tests.
- `pykan/`: vendored pykan implementation used by the experiments, with its own license.
- `results_paper/`: lightweight submitted outputs where suitable for GitHub.
- `data/`: data-access notes for the experimental raw files.
- `FIGURE_MANIFEST.md`: figure-by-figure mapping to the producing script.
- `requirements.txt`: pinned Python environment used for submitted runs.

Large raw `.xls` experimental data files and binary run artifacts are provided in the supplementary archive submitted with the manuscript. After acceptance, the final reviewed code/data snapshot will be archived in a long-term repository such as Zenodo, Figshare, or Mendeley Data and assigned a permanent DOI.

## Environment

Submitted runs used Python 3.12 with PyTorch 2.7.1+cu118, NumPy, SciPy, Matplotlib, Pandas, and related scientific Python packages. Install from:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU accelerates training. Verification and many post-processing scripts can also run on CPU.

## Basic reproduction

```bash
git clone https://github.com/XIAOYANGSREN/MATCOMRevise.git
cd MATCOMRevise
python -m venv .venv
pip install -r requirements.txt

cd 03_code
python src/verify_math.py
python src/run_all_examples.py
```

Individual main examples can be run with:

```bash
python src/run_all_examples.py --example 1
python src/run_all_examples.py --example 2
python src/run_all_examples.py --example 3
```

Revision experiments are in `04_new_experiments/`. Each experiment writes outputs to its local `results/` directory. See `FIGURE_MANIFEST.md` for figure-to-script mapping.

## Data availability

The source code and scripts required to reproduce the numerical results are available in this repository. The full review package, including large raw data files and binary run artifacts, is provided with the manuscript. The final accepted version will be archived in a permanent repository with a DOI.
