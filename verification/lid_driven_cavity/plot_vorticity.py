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

j = [1,   8,      9,      10,     14,     23,     37,     59,     65,  80,     95,     110,    123,    124,    125,    126,    129]
y = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]

V = {
    100: [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0],
    400: [0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477,  0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.0]
}

def plot_lid_driven_cavity_vorticity(Re, n):
    ds = xr.open_dataset(f"lid_driven_cavity_Re{Re}.nc")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.suptitle(f"Lid-driven cavity, Re = {Re}, t = {ds.time[n].values:.2f}", fontsize=16)

    # v = ds.v.isel(time=n).squeeze()
    # v.plot.pcolormesh(ax=ax, vmin=-1, vmax=1, cmap=cmocean.cm.balance, extend="both")

    # ζ = ds.ζ.isel(time=n).squeeze()
    # ζ.plot.pcolormesh(ax=ax, cmap=cmocean.cm.curl, extend="both",
    #                   norm=colors.SymLogNorm(linthresh=1e-2, vmin=-1e2, vmax=1e2))

    # ax.set_title("")
    # ax.set_xlabel("y")
    # ax.set_ylabel("z")
    # ax.set_aspect("equal")

    plt.plot(y, V[Re], linestyle="", marker="o")

    v = ds.v.isel(time=n, yF=64)
    plt.plot(ds.zC, v.values.flatten())

    print(f"Saving frame {n}/{ds.time.size-1}...")
    plt.savefig(f"lid_driven_cavity_Re{Re}_{n:05d}.png")
    plt.close("all")

ds = xr.open_dataset(f"lid_driven_cavity_Re{Re}.nc")
plot_lid_driven_cavity_vorticity(100, ds.time.size-1)
