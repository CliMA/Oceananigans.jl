import os
import ffmpeg
import cmocean
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

plt.ioff()

ds = xr.open_dataset("convecting_plankton.nc")

xC = ds["xC"]
xF = ds["xF"]
zC = ds["zC"]
zF = ds["zF"]

N = ds.time.size
for n in range(N):
    print(f"Plotting vertically stretched plankton frame {n}/{N}...")

    fig, (ax_w, ax_p) = plt.subplots(figsize=(15, 8), ncols=2, dpi=200)
    plt.subplots_adjust(hspace=0.25)
    fig.suptitle(f"Vertically stretched convecting plankton: time = {ds.time.values[n]/3600:.2f} hours")

    ds.w.isel(time=n).plot(ax=ax_w, cmap=cmocean.cm.balance, vmin=-0.015, vmax=0.015, add_colorbar=False)
    ax_w.set_title("w(x, z)")
    ax_w.set_xlabel("x (m)")
    ax_w.set_ylabel("z (m)")

    ax_w.grid(True, which="both", axis="both", linestyle="-", color="k", linewidth=1.0, alpha=0.5)
    ax_w.set_xticks(xF, minor=False)
    ax_w.set_yticks(zC, minor=False)
    ax_w.xaxis.set_ticklabels([])
    ax_w.yaxis.set_ticklabels([])
    ax_w.set_xlim(xF[0], xF[-1])
    ax_w.set_ylim(zF[0], zF[-1])
    ax_w.set_aspect("equal")

    ds.plankton.isel(time=n).plot(ax=ax_p, cmap=cmocean.cm.matter, vmin=0.95, vmax=1.1, add_colorbar=False)
    ax_p.set_title("plankton(x, z)")
    ax_p.set_xlabel("x (m)")
    ax_p.set_ylabel("z (m)")

    ax_p.grid(True, which="both", axis="both", linestyle="-", color="k", linewidth=1.0, alpha=0.5)
    ax_p.set_xticks(xF, minor=False)
    ax_p.set_yticks(zF, minor=False)
    ax_p.xaxis.set_ticklabels([])
    ax_p.yaxis.set_ticklabels([])
    ax_p.set_xlim(xF[0], xF[-1])
    ax_p.set_ylim(zF[0], zF[-1])
    ax_p.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"vertically_stretched_plankton_{n:05d}.png")
    plt.close(fig)


(
    ffmpeg
    .input(f"vertically_stretched_plankton_%05d.png", framerate=15)
    .output(f"vertically_stretched_plankton.mp4", crf=15, pix_fmt="yuv420p")
    .overwrite_output()
    .run()
)

png_files = [file for file in os.listdir(os.getcwd()) if file.endswith(".png")]
print(f"Deleting {len(png_files)} leftover png files...")
[os.remove(file) for file in png_files]
