import matplotlib

matplotlib.use("Agg")

import numpy as np
from numpy import ones, meshgrid, linspace
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.switch_backend("Agg")

from joblib import Parallel, delayed

def plot_pcolormesh3d_from_netcdf(nc_filepath, png_filepath, var, dt, vmin, vmax, var_offset=0, time=0):
    # Enforcing style in here so that it's applied to all workers launched by joblib.
    plt.style.use("dark_background")

    nc_output = xr.open_dataset(nc_filepath)

    # TODO: Save time in NetCDF output. For now use the keyword argument.
    # time = nc_output["time"]

    Nx = nc_output["xC"].size
    Ny = nc_output["yC"].size
    Nz = nc_output["zC"].size

    Lx = nc_output["xF"].values[-1]
    Ly = nc_output["yF"].values[-1]
    Lz = nc_output["zF"].values[-1]

    xC = nc_output["xC"]
    yC = nc_output["yC"]
    zC = nc_output["zC"]

    data = nc_output[var]

    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig)

    data_z_slice = data.sel(yC=slice(0, Lx//2), zC=zC[0]).values - var_offset
    data_x_slice = data.sel(xC=xC[-1], yC=slice(0, Lx//2)).values - var_offset
    data_y_slice = data.sel(yC=yC.values[Ny//2]).values - var_offset

    XC_z, YC_z = meshgrid(linspace(0, Lx, Nx),      linspace(0, Ly/2, Ny//2))
    YC_x, ZC_x = meshgrid(linspace(0, Ly/2, Ny//2), linspace(0, -Lz, Nz))
    XC_y, ZC_y = meshgrid(linspace(0, Lx, Nx),      linspace(0, -Lz, Nz))

    x_offset, y_offset, z_offset = Lx, Ly/2, 0

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    Z_offset = z_offset*ones((Ny//2, Nx))
    X_offset = x_offset*ones((Nz, Ny//2))
    Y_offset = y_offset*ones((Nz, Nx))

    print(XC_z.shape)
    print(YC_z.shape)
    print(Z_offset.shape)
    print(data_z_slice.shape)

    cf1 = ax.plot_surface(XC_z, YC_z, Z_offset, rstride=1, cstride=1, linewidth=0, alpha=1, vmin=vmin, vmax=vmax,
                          antialiased=False, shade=False, facecolors=plt.cm.jet(norm(data_z_slice)))
    cf2 = ax.plot_surface(X_offset, YC_x, ZC_x, rstride=1, cstride=1, linewidth=0, alpha=1, vmin=vmin, vmax=vmax,
                          antialiased=False, shade=False, facecolors=plt.cm.jet(norm(data_x_slice)))
    cf3 = ax.plot_surface(XC_y, Y_offset, ZC_y, rstride=1, cstride=1, linewidth=0, alpha=1, vmin=vmin, vmax=vmax,
                          antialiased=False, shade=False, facecolors=plt.cm.jet(norm(data_y_slice)))

    # Can't get the colorbar to work with Axes3D and plot_surface...
    # clb = fig.colorbar(cf3, ticks=np.linspace(vmin, vmax, num=6))
    # clb.ax.set_title(r"T (C)")

    ax.set_xlim3d(0, Lx)
    ax.set_ylim3d(0, Ly)
    ax.set_zlim3d(-Lz, 0)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.view_init(elev=30, azim=45)

    ax.set_title("t = {:07d} s ({:03.2f} days)".format(int(time), time / (3600*24)), y=1.05)

    ax.set_xticks(np.linspace(0, Lx, num=5))
    ax.set_yticks(np.linspace(0, Ly, num=5))
    ax.set_zticks(np.linspace(0, -Lz, num=5))

    # plt.show()

    plt.savefig(png_filepath, dpi=300, format="png", transparent=False)
    print("Saving: {:s}".format(png_filepath))

    plt.close("all")

if __name__ == "__main__":
    # Plot a single frame from one NetCDF file.
    plot_pcolormesh3d_from_netcdf(nc_filepath="convection000072000.nc", png_filepath="convection000072000.png",
                                  var="T", dt=0.1, vmin=18.5, vmax=20, var_offset=273.15)

    def input_filepath(i):
        return "convection" + str(i).zfill(9) + ".nc"

    def output_filepath(i):
        return "convection" + str(i).zfill(9) + ".png"

    # Plot many frames from many NetCDF files in parallel.
    # freq = 3600  # Output frequency in iterations.
    # Parallel(n_jobs=8)(
    #     delayed(plot_pcolormesh3d_from_netcdf(nc_filepath=input_filepath(i), png_filepath=output_filepath(i),
    #                                           var="T", dt=0.1, vmin=19, vmax=20, var_offset=273.15))
    #     for n in np.arange(0, 1000)
    # )
