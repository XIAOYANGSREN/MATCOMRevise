"""Sensitivity figure from the sweep CSVs: RMSE vs N_t, G, H,
averaged over seeds, with the L1 discretisation floor marked.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE.parents[1] / "figures_out"
L1_FLOOR = 0.587  # Pa, Example 1 RMSE at N_t=120


def load(sweep, xcol):
    rmse = defaultdict(list)
    with open(RESULTS / f"kan_sensitivity_{sweep}.csv", newline="") as f:
        for r in csv.DictReader(f):
            rmse[float(r[xcol])].append(float(r["rmse"]))
    xs = sorted(rmse)
    return xs, [sum(rmse[x]) / len(rmse[x]) for x in xs]


fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

xs, ys = load("n_points", "n_points")
ax[0].loglog(xs, ys, "o-", color="#1f77b4")
ax[0].set_xlabel(r"$N_t$ (time-grid points)")
ax[0].set_ylabel("RMSE (Pa)")
ax[0].set_title("(a) time-grid resolution")

xs, ys = load("grid", "grid")
ax[1].plot(xs, ys, "s-", color="#2ca02c")
ax[1].axhline(L1_FLOOR, ls="--", color="0.5", label=f"L1 floor {L1_FLOOR:.3f} Pa")
ax[1].set_xlabel(r"$G$ (B-spline grid count)")
ax[1].set_ylabel("RMSE (Pa)")
ax[1].set_ylim(0.40, 0.75)
ax[1].set_title("(b) B-spline grid count")
ax[1].legend(fontsize=8)

xs, ys = load("width", "width")
ax[2].plot(xs, ys, "^-", color="#d62728")
ax[2].axhline(L1_FLOOR, ls="--", color="0.5", label=f"L1 floor {L1_FLOOR:.3f} Pa")
ax[2].set_xlabel(r"$H$ (hidden width)")
ax[2].set_ylabel("RMSE (Pa)")
ax[2].set_ylim(0.40, 0.75)
ax[2].set_title("(c) hidden width")
ax[2].legend(fontsize=8)

for a in ax:
    a.grid(alpha=0.3, which="both")
fig.tight_layout()

FIGS.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(RESULTS / f"sensitivity.{ext}", dpi=300, bbox_inches="tight")
    # historical filename; printed as Figure 8 in the manuscript
    fig.savefig(FIGS / f"Figure9_sensitivity.{ext}", dpi=300, bbox_inches="tight")
print(f"saved {FIGS / 'Figure9_sensitivity.png'}")
