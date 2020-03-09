import joblib
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmocean
import ffmpeg

def plot_lid_driven_cavity_vorticity(Re, n):
    ds = xr.open_dataset(f"lid_driven_cavity_Re{Re}.nc")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.suptitle(f"Lid-driven cavity, Re = {Re}, t = {ds.time[n].values:.2f}", fontsize=16)

    # v = ds.v.isel(time=n).squeeze()
    # v.plot.pcolormesh(ax=ax, vmin=-1, vmax=1, cmap=cmocean.cm.balance, extend="both")

    ζ = ds.ζ.isel(time=n).squeeze()
    ζ.plot.pcolormesh(ax=ax, cmap=cmocean.cm.curl, extend="both",
                      norm=colors.SymLogNorm(linthresh=1e-2, vmin=-1e2, vmax=1e2))

    ax.set_title("")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal")

    print(f"Saving frame {n}/{ds.time.size-1}...")
    plt.savefig(f"lid_driven_cavity_Re{Re}_{n:05d}.png")
    plt.close("all")

ds = xr.open_dataset(f"lid_driven_cavity_Re100.nc")
plot_lid_driven_cavity_vorticity(100, ds.time.size-1)
