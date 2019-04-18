import matplotlib

matplotlib.use("Agg")

import os
import numpy as np
from numpy import ones, meshgrid, linspace
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.switch_backend("Agg")

from joblib import Parallel, delayed

def plot_contourf3d_from_netcdf(nc_filepath, png_filepath, var, dt, vmin, vmax, n_contours, cmap="inferno", var_offset=0, time=0):
    # Enforcing style in here so that it's applied to all workers launched by joblib.
    plt.style.use("dark_background")

    contour_spacing = (vmax - vmin) / n_contours

    nc_output = xr.open_dataset(nc_filepath)

    # TODO: Save time in NetCDF output. For now use the keyword argument.
    # time = nc_output["time"]

    Nx = nc_output["xC"].size
    Ny = nc_output["yC"].size
    Nz = nc_output["zC"].size

    Lx = Nx * (nc_output["xF"].values[1] - nc_output["xF"].values[0])
    Ly = Ny * (nc_output["yF"].values[1] - nc_output["yF"].values[0])
    Lz = Nz * (nc_output["zF"].values[1] - nc_output["zF"].values[0])

    xC = nc_output["xC"]
    yC = nc_output["yC"]
    zC = nc_output["zC"]

    data = nc_output[var]

    fig = plt.figure(figsize=(16, 9))
    # ax = fig.gca(projection="3d")
    ax = Axes3D(fig)  # This is the proper call. See: https://stackoverflow.com/questions/3810865/matplotlib-unknown-projection-3d-error

    data_z_slice = data.sel(yC=slice(0, Lx//2), zC=zC[0]).values - var_offset
    data_x_slice = data.sel(xC=xC[-1], yC=slice(0, Lx//2)).values - var_offset
    data_y_slice = data.sel(yC=yC.values[Ny//2]).values - var_offset

    XC_z, YC_z = meshgrid(linspace(0, Lx, Nx),      linspace(0, Ly/2, Ny//2))
    YC_x, ZC_x = meshgrid(linspace(0, Ly/2, Ny//2), linspace(0, -Lz, Nz))
    XC_y, ZC_y = meshgrid(linspace(0, Lx, Nx),      linspace(0, -Lz, Nz))

    x_offset, y_offset, z_offset = Lx, Ly/2, 0

    cf1 = ax.contourf(XC_z, YC_z, data_z_slice, zdir="z", offset=z_offset, levels=np.arange(vmin, vmax, contour_spacing), cmap=cmap)
    cf2 = ax.contourf(data_x_slice, YC_x, ZC_x, zdir="x", offset=x_offset, levels=np.arange(vmin, vmax, contour_spacing), cmap=cmap)
    cf3 = ax.contourf(XC_y, data_y_slice, ZC_y, zdir="y", offset=y_offset, levels=np.arange(vmin, vmax, contour_spacing), cmap=cmap)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    Z_offset = z_offset*ones((Ny//2, Nx))
    X_offset = x_offset*ones((Nz, Ny//2))
    Y_offset = y_offset*ones((Nz, Nx))

    clb = fig.colorbar(cf3, ticks=[19.0, 19.2, 19.4, 19.6, 19.8, 20.0], shrink=0.9)
    clb.ax.set_title(r"T (°C)")

    ax.set_xlim3d(0, Lx)
    ax.set_ylim3d(0, Ly)
    ax.set_zlim3d(-Lz, 0)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.view_init(elev=30, azim=45)

    ax.set_title("t = {:07d} s ({:03.2f} days)".format(int(time), time / (3600*24)), y=1.05)

    ax.set_xticks(linspace(0, Lx, num=5))
    ax.set_yticks(linspace(0, Ly, num=5))
    ax.set_zticks(linspace(0, -Lz, num=5))

    # plt.show()

    plt.savefig(png_filepath, dpi=300, format="png", transparent=False)
    print("Saving: {:s}".format(png_filepath))

    plt.close("all")

if __name__ == "__main__":
    # Plot a single frame from one NetCDF file.
    # plot_contourf3d_from_netcdf(nc_filepath="convection000072000.nc", png_filepath="convection000072000.png",
    #                             var="T", dt=0.1, vmin=18.5, vmax=20, n_contours=100, var_offset=273.15)

    def input_filepath(i):
        return os.path.join("wind_stress_N256_tau0.1_Q-75_dTdz0.01_k0.0001_dt0.25_days4", "wind_stress_N256_tau0.1_Q-75_dTdz0.01_k0.0001_dt0.25_days4_" + str(i) + ".nc")

    def output_filepath(i):
        return os.path.join("frames", "wind_stress_" + str(i).zfill(4) + ".png")

    # Plot many frames from many NetCDF files in parallel.
    freq = 20  # Output frequency in iterations.
    dt = 0.25
    Parallel(n_jobs=-1)(
        delayed(plot_contourf3d_from_netcdf)(nc_filepath=input_filepath(n), png_filepath=output_filepath(n), var="T",
                                            dt=dt, vmin=19, vmax=20.05, n_contours=400, cmap="prism", var_offset=273.15, time=n*dt*freq)
        for n in [0, 10, 100, 400, 1000] # np.arange(0, 1000)
    )
