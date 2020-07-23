import joblib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmocean
import ffmpeg

from numpy import nan

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
    100: [0.0000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.0000],
    400: [0.0000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663, -0.12146, 0.0000]
}

y_ζ_Ghia = [0.0000, 0.0625, 0.1250, 0.1875, 0.2500, 0.3125, 0.3750, 0.4375, 0.5000, 0.5625, 0.6250, 0.6875, 0.7500, 0.8125, 0.8750, 0.9375, 1.0000]

ζ_Ghia = {
    100: [nan, 40.0110, 22.5378, 16.2862, 12.7844, 10.4199, 8.69628, 7.43218, 6.57451, 6.13973, 6.18946, 6.82674, 8.22110, 10.7414, 15.6591, 30.7923, nan],
    400: [nan, 53.6863, 34.6351, 26.5825, 21.0985, 16.8900, 13.7040, 11.4537, 10.0545, 9.38889, 9.34599, 9.88979, 11.2018, 13.9068, 19.6859, 35.0773, nan]
}

def plot_lid_driven_cavity_frame(Re, n):
    ds = xr.open_dataset(f"lid_driven_cavity_Re{Re}.nc")
    Ny = ds.yC.size
    Nz = ds.zC.size

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 16), dpi=200)
    plt.subplots_adjust(hspace=0.25)
    fig.suptitle(f"Lid-driven cavity, Re = {Re}, t = {ds.time[n].values:.2f}", fontsize=16)

    ax_v_line, ax_v_mesh = axes[0, 0], axes[0, 1]
    ax_w_line, ax_w_mesh = axes[1, 0], axes[1, 1]
    ax_ζ_line, ax_ζ_mesh = axes[2, 0], axes[2, 1]

    v_line = ds.v.isel(time=n, yF=Ny//2)
    ax_v_line.plot(y_Ghia, v_Ghia[Re], label="Ghia et al. (1982)", color="tab:blue", linestyle="", marker="o", fillstyle="none")
    ax_v_line.plot(ds.zC, v_line.values.flatten(), label="Oceananigans.jl", color="tab:blue")
    ax_v_line.legend(loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.2), ncol=2, frameon=False)
    ax_v_line.set_xlabel("z")
    ax_v_line.set_ylabel("v")
    ax_v_line.set_xlim([0, 1])

    w_line = ds.w.isel(time=n, zF=Nz//2)
    ax_w_line.plot(z_Ghia, w_Ghia[Re], label="Ghia et al. (1982)", color="tab:orange", linestyle="", marker="o", fillstyle="none")
    ax_w_line.plot(ds.yC, w_line.values.flatten(), label="Oceananigans.jl", color="tab:orange")
    ax_w_line.legend(loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.2), ncol=2, frameon=False)
    ax_w_line.set_xlabel("y")
    ax_w_line.set_ylabel("w")
    ax_w_line.set_xlim([0, 1])

    v = ds.v.isel(time=n).squeeze()
    img_v = v.plot.pcolormesh(ax=ax_v_mesh, vmin=-1, vmax=1, cmap=cmocean.cm.balance, add_colorbar=False)
    fig.colorbar(img_v, ax=ax_v_mesh, extend="both")
    ax_v_mesh.axvline(x=0.5, color="tab:blue", alpha=0.5)
    ax_v_mesh.set_title("v-velocity")
    ax_v_mesh.set_xlabel("y")
    ax_v_mesh.set_ylabel("z")
    ax_v_mesh.set_aspect("equal")

    w = ds.w.isel(time=n).squeeze()
    img_w = w.plot.pcolormesh(ax=ax_w_mesh, vmin=-1, vmax=1, cmap=cmocean.cm.balance, add_colorbar=False)
    fig.colorbar(img_w, ax=ax_w_mesh, extend="both")
    ax_w_mesh.axhline(y=0.5, color="tab:orange", alpha=0.5)
    ax_w_mesh.set_title("w-velocity")
    ax_w_mesh.set_xlabel("y")
    ax_w_mesh.set_ylabel("z")
    ax_w_mesh.set_aspect("equal")

    ζ_line = -ds.ζ.isel(time=n, zF=Nz-1)
    ax_ζ_line.plot(y_ζ_Ghia, ζ_Ghia[Re], label="Ghia et al. (1982)", color="tab:red", linestyle="", marker="o", fillstyle="none")
    ax_ζ_line.plot(ds.yF, ζ_line.values.flatten(), label="Oceananigans.jl", color="tab:red")
    ax_ζ_line.legend(loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.2), ncol=2, frameon=False)
    ax_ζ_line.set_xlabel("y")
    ax_ζ_line.set_ylabel("vorticity $\zeta$")
    ax_ζ_line.set_xlim([0, 1])
    ax_ζ_line.set_ylim(bottom=0)

    ζ = ds.ζ.isel(yF=slice(0, -2), zF=slice(0, -2), time=n).squeeze()
    img_ζ = ζ.plot.pcolormesh(ax=ax_ζ_mesh, cmap=cmocean.cm.curl, extend="both", add_colorbar=False,
                              norm=colors.SymLogNorm(base=10, linthresh=1e-2, vmin=-1e2, vmax=1e2))
    fig.colorbar(img_ζ, ax=ax_ζ_mesh, extend="both")
    ax_ζ_mesh.axhline(y=ds.zF[Nz-1], color="tab:red", alpha=0.5)
    ax_ζ_mesh.set_title("vorticity")
    ax_ζ_mesh.set_xlabel("y")
    ax_ζ_mesh.set_ylabel("z")
    ax_ζ_mesh.set_aspect("equal")

    print(f"Saving lid-driven cavity Re={Re} frame {n}/{ds.time.size-1}...")
    plt.savefig(f"lid_driven_cavity_Re{Re}_{n:05d}.png")
    plt.close("all")

Re = 400
ds = xr.open_dataset(f"lid_driven_cavity_Re{Re}.nc")

joblib.Parallel(n_jobs=-1)(
    joblib.delayed(plot_lid_driven_cavity_frame)(Re, n)
    for n in range(ds.time.size)
)

(
    ffmpeg
    .input(f"lid_driven_cavity_Re{Re}_%05d.png", framerate=30)
    .output(f"lid_driven_cavity_Re{Re}.mp4", crf=15, pix_fmt='yuv420p')
    .overwrite_output()
    .run()
)

