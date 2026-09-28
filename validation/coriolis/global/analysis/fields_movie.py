"""Surface speed, SST and surface v of the runs from rest every 30 days, up to the last day common to all runs.
Usage: python3 fields_movie.py <output.mp4> [float type]"""
import sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from readjld import read, iterations, coordinates

output = sys.argv[1]
float_type = sys.argv[2] if len(sys.argv) > 2 else "Float64"
titles = {"FourPoint": "four-point average", "CD": "C-D", "FixedSE": "oriented, fixed SE", "FixedNW": "oriented, fixed NW",
          "ShearSigned": "oriented, shear-signed"}
cases = {name: f"{name}_{float_type}" for name in titles if iterations(f"{name}_{float_type}")}
frames = sorted(i for i in set.intersection(*[set(iterations(case)) for case in cases.values()]) if i % 720 == 0 and i > 0)

λ, φ = coordinates()
longitude, latitude = λ[:, 0], φ[0, :]
order = np.argsort(longitude)
rows = (latitude > -70) & (latitude < 60)
fields = {"speed": ("surface speed (m s$^{-1}$)", "magma", 0, 0.8),
          "T": ("SST (°C)", "RdYlBu_r", 0, 30),
          "v": ("surface meridional velocity (m s$^{-1}$)", "RdBu_r", -0.2, 0.2)}

def field(name, iteration, case):
    wet = read(case, "v", iteration) != 0
    return np.where(wet, read(case, name, iteration), np.nan)[order][:, rows].T

fig, axes = plt.subplots(3, len(cases), figsize=(5.5 * len(cases), 10.5), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
meshes = {}
for r, (name, (label, cmap, lowest, highest)) in enumerate(fields.items()):
    for c, (scheme, case) in enumerate(cases.items()):
        ax = axes[r, c]
        ax.set_facecolor("0.8")
        meshes[(name, case)] = ax.pcolormesh(longitude[order], latitude[rows], field(name, frames[0], case), cmap=cmap,
                                             vmin=lowest, vmax=highest, shading="auto")
        r == 0 and ax.set_title(titles[scheme], fontsize=14)
        c == 0 and ax.set_ylabel("latitude", fontsize=11)
        r == 2 and ax.set_xlabel("longitude", fontsize=11)
    fig.colorbar(meshes[(name, case)], ax=axes[r, :], shrink=0.9, pad=0.01).set_label(label, fontsize=11)
suptitle = fig.suptitle("", fontsize=16)

def update(n):
    iteration = frames[n]
    for (name, case), mesh in meshes.items():
        mesh.set_array(field(name, iteration, case).ravel())
    suptitle.set_text(f"Spin-up from rest, day {iteration // 24} (year {iteration / 24 / 365:.1f})")
    return list(meshes.values())

FuncAnimation(fig, update, frames=len(frames)).save(output, writer=FFMpegWriter(fps=6, bitrate=8000), dpi=80)
update(len(frames) - 1)
fig.savefig(output.replace(".mp4", "_last.png"), dpi=70)
print(f"{len(frames)} frames, days {frames[0] // 24} to {frames[-1] // 24}")
