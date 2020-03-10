import joblib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmocean
import ffmpeg

#####
##### Data from tables 1 and 2 of Ghia et al. (1982).
#####

j_Ghia = [1,      8,      9,      10,     14,     23,     37,     59,     65,     80,     95,     110,    123,    124,    125,    126,    129]
y_Ghia = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]

v_Ghia = {
    100: [0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0000],
    400: [0.0000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477,  0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.0000]
}

k_Ghia = [1,      9,      10,     11,     13,     21,     30,     31,     65,     104,    111,    117,    122,    123,    124,    125,    129]
z_Ghia = [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000]

w_Ghia = {
    100: [0.0000, 0.09233, 0.10091, 0.1089, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.0000]
}

def plot_lid_driven_cavity_vorticity(Re, n):
    ds = xr.open_dataset(f"lid_driven_cavity_Re{Re}.nc")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), dpi=200)
    plt.subplots_adjust(hspace=0.25)
    fig.suptitle(f"Lid-driven cavity, Re = {Re}, t = {ds.time[n].values:.2f}", fontsize=16)

    ax_v, ax_V = axes[0, 0], axes[0, 1]
    ax_w, ax_W = axes[1, 0], axes[1, 1]

    v_line = ds.v.isel(time=n, yF=64)
    ax_v.plot(y_Ghia, v_Ghia[Re], label="Ghia et al. (1982)", color="tab:blue", linestyle="", marker="o", fillstyle="none")
    ax_v.plot(ds.zC, v_line.values.flatten(), label="Oceananigans.jl", color="tab:blue")
    ax_v.legend(loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.2), ncol=2, frameon=False)
    ax_v.set_xlabel("z")
    ax_v.set_ylabel("v")
    ax_v.set_xlim([0, 1])

    w_line = ds.w.isel(time=n, zF=64)
    ax_w.plot(z_Ghia, w_Ghia[Re], label="Ghia et al. (1982)", color="tab:orange", linestyle="", marker="o", fillstyle="none")
    ax_w.plot(ds.yC, w_line.values.flatten(), label="Oceananigans.jl", color="tab:orange")
    ax_w.legend(loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.2), ncol=2, frameon=False)
    ax_w.set_xlabel("y")
    ax_w.set_ylabel("w")
    ax_w.set_xlim([0, 1])

    v = ds.v.isel(time=n).squeeze()
    img_v = v.plot.pcolormesh(ax=ax_V, vmin=-1, vmax=1, cmap=cmocean.cm.balance, add_colorbar=False)
    fig.colorbar(img_v, ax=ax_V, extend="both")
    ax_V.axvline(x=0.5, color="tab:blue", alpha=0.5)
    ax_V.set_title("v-velocity")
    ax_V.set_xlabel("y")
    ax_V.set_ylabel("z")
    ax_V.set_aspect("equal")

    w = ds.w.isel(time=n).squeeze()
    img_w = w.plot.pcolormesh(ax=ax_W, vmin=-1, vmax=1, cmap=cmocean.cm.balance, add_colorbar=False)
    fig.colorbar(img_w, ax=ax_W, extend="both")
    ax_W.axhline(y=0.5, color="tab:orange", alpha=0.5)
    ax_W.set_title("w-velocity")
    ax_W.set_xlabel("y")
    ax_W.set_ylabel("z")
    ax_W.set_aspect("equal")

    print(f"Saving frame {n}/{ds.time.size-1}...")
    plt.savefig(f"lid_driven_cavity_Re{Re}_{n:05d}.png")
    plt.close("all")

Re = 100
ds = xr.open_dataset(f"lid_driven_cavity_Re{Re}.nc")
plot_lid_driven_cavity_vorticity(100, ds.time.size-1)
