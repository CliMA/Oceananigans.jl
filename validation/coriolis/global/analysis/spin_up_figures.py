"""Runs from rest: band variances of surface v, barotropic streamfunction range and mean surface speed every 10 days, and the
zonal 2Δ part of surface v at the last day common to all runs. Usage: python3 spin_up_figures.py <output prefix> [float type]"""
import sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bands import bands, band_variance
from readjld import read, iterations, coordinates

prefix = sys.argv[1]
float_type = sys.argv[2] if len(sys.argv) > 2 else "Float64"
colors = {"FourPoint": "k", "CD": "tab:orange", "FixedSE": "tab:red", "FixedNW": "tab:blue", "ShearSigned": "tab:green"}
cases = {name: f"{name}_{float_type}" for name in colors if iterations(f"{name}_{float_type}")}
available = {name: [i for i in iterations(case) if i > 0] for name, case in cases.items()}
last = min(its[-1] for its in available.values())

λ, φ = coordinates()
longitude, latitude = λ[:, 0], φ[0, :]
order = np.argsort(longitude)
map_rows = (latitude > -70) & (latitude < 60)

fig, axes = plt.subplots(2, 4, figsize=(22, 9))
titles = {"8-22S": "8–22°S, zonal 3–5Δ", "8-22N": "8–22°N, zonal 3–5Δ", "30-50S": "30–50°S, near-Nyquist 2–3Δ",
          "30-50N": "30–50°N, near-Nyquist 2–3Δ", "30-50N mesoscale": "30–50°N, mesoscale 5–20Δ",
          "55-65S mesoscale": "55–65°S, mesoscale 5–20Δ"}
for ax, band in zip(axes.flat, bands):
    for name, case in cases.items():
        its = [i for i in available[name] if i <= last]
        ax.plot([i / 24 for i in its], [band_variance(read(case, "v", i), band) for i in its], color=colors[name], lw=2, label=name)
    ax.set_title(titles[band]); ax.set_xlabel("day"); ax.set_ylabel("surface v variance (cm²/s²)")
axes[0, 0].legend(fontsize=9)
for name, case in cases.items():
    its = [i for i in available[name] if i <= last]
    ψ = [read(case, "ψ", i) for i in its]
    axes[1, 2].plot([i / 24 for i in its], [(np.nanmax(x) - np.nanmin(x)) / 1e6 for x in ψ], color=colors[name], lw=2, label=name)
    speeds = [read(case, "speed", i) for i in its]
    axes[1, 3].plot([i / 24 for i in its], [100 * np.mean(s[s > 0]) for s in speeds], color=colors[name], lw=2, label=name)
axes[1, 2].set_title("barotropic streamfunction range"); axes[1, 2].set_ylabel("Sv"); axes[1, 2].set_xlabel("day")
axes[1, 3].set_title("mean surface speed"); axes[1, 3].set_ylabel("cm/s"); axes[1, 3].set_xlabel("day")
fig.suptitle(f"Spin-up from rest, 1° tripolar, 4 levels: days 10–{last // 24}", fontsize=14)
fig.tight_layout()
fig.savefig(f"{prefix}_series.png", dpi=85)

fig, axes = plt.subplots(1, len(cases), figsize=(8 * len(cases), 4.5), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
for ax, (name, case) in zip(axes.flat, cases.items()):
    v = read(case, "v", last)
    wet = (v != 0) & (np.roll(v, 1, 0) != 0) & (np.roll(v, -1, 0) != 0)
    stripes = np.where(wet, (2 * v - np.roll(v, 1, 0) - np.roll(v, -1, 0)) / 4, np.nan)
    ax.set_facecolor("0.8")
    image = ax.pcolormesh(longitude[order], latitude[map_rows], 100 * stripes[order][:, map_rows].T, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto")
    ax.set_title(name)
fig.colorbar(image, ax=axes, shrink=0.8, label="zonal 2Δ part of surface v (cm/s)")
fig.suptitle(f"Spin-up from rest, day {last // 24}: grid-scale zonal stripes", fontsize=14)
fig.savefig(f"{prefix}_stripes.png", dpi=85)
print(f"last common day {last // 24}")
