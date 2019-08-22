import matplotlib

matplotlib.use("Agg")

import os
import numpy as np
from numpy import ones, meshgrid, linspace, square, mean
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

plt.switch_backend("Agg")

from joblib import Parallel, delayed

def plot_contourf3d_from_netcdf(nc_filepath, png_filepath, var, dt, vmin, vmax, n_contours, cmap="inferno", var_offset=0, time=0):
    # For some reason I need to do the import here so it shows up on all joblib workers.
    from mpl_toolkits.mplot3d import Axes3D

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

    ax = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=3, projection="3d")
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.25)

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

    T_profile = nc_output["T"].mean(dim=["xC", "yC"])

    ax2 = plt.subplot2grid((3, 4), (0, 3))
    ax2.plot(T_profile.sel(zC=slice(0, -25)) - 273.15, zC.sel(zC=slice(0, -25)))
    ax2.set_title(r"$\overline{T}(z)$ [°C]")
    ax2.set_ylabel("z (m)")
    ax2.set_xlim(19.75, 20)
    ax2.set_ylim(-25, 0)

    u, v, w = nc_output["u"], nc_output["v"], nc_output["w"]
    HTKE_profile = 0.5 * mean(square(u.values) + square(v.values), axis=(1, 2))
    VTKE_profile = 0.5 * square(w).mean(dim=["xC", "yC"])

    ax3 = plt.subplot2grid((3, 4), (1, 3))
    ax3.plot(HTKE_profile[0:Nz//4], zC.sel(zC=slice(0, -25)).values, color="tab:orange", label=r"$(u^2 + v^2)/2$")
    ax3.plot(10 * VTKE_profile.sel(zF=slice(0, -25)), zC.sel(zC=slice(0, -25)), color="tab:green", label=r"$10 \times w^2/2$")
    ax3.set_title("Turbulent kinetic energy [m$^2$/s$^2$]")
    ax3.set_ylabel("z (m)")
    ax3.set_xlim(-0.001, 0.02)
    ax3.set_ylim(-25, 0)
    ax3.legend(loc="lower right")

    alpha = 2.07e-4
    g = 9.80665
    T = nc_output["T"]
    buoyancy_flux_profile = mean(alpha * g * w.values * T.values, axis=(1, 2))

    ax4 = plt.subplot2grid((3, 4), (2, 3))
    ax4.plot(buoyancy_flux_profile[0:Nz//4], zC.sel(zC=slice(0, -25)).values, color="tab:red")
    ax4.set_title(r"Buoyancy flux $\alpha g \overline{w' T'}$")
    ax4.set_ylabel("z (m)")
    ax4.set_xlim(-5e-8, 5e-8)
    ax4.set_ylim(-25, 0)

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
    Parallel(n_jobs=32)(
        delayed(plot_contourf3d_from_netcdf)(nc_filepath=input_filepath(n), png_filepath=output_filepath(n), var="T",
                                            dt=dt, vmin=19, vmax=20.05, n_contours=400, cmap="prism", var_offset=273.15, time=n*dt*freq)
        for n in np.arange(0, 2000)
    )
