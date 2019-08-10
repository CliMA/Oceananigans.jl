import os

import h5py
from joblib import Parallel, delayed
from numpy import ones, meshgrid, linspace, arange, square, mean

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def fn(i):
    return "eddying_channel_" + str(i) + ".jld2"

def fout(i):
    return "channel_plot_" + str(i) + ".png"

def plot_frame(i):
    from mpl_toolkits.mplot3d import Axes3D
    plt.style.use("dark_background")

    vmin, vmax = 7.95, 12.6
    n_contours = 50
    contour_spacing = (vmax - vmin) / n_contours
    cmap = "inferno"

    Lx, Ly, Lz = 250e3, 500e3, 1000  # 160×512×1 km
    Nx, Ny, Nz = 256, 512, 128
    Δx, Δy, Δz = Lx/Nx, Ly/Ny, Lz/Nz

    xC = linspace(0, Lx, num=Nx)
    yC = linspace(0, Ly, num=Ny)
    zC = linspace(0, -Lz, num=Nz)

    print("[{:d}] Reading data...".format(i))

    f = h5py.File(fn(i), "r")
    t = f["t"].value
    data = f["T"].value[0:Nz+1, 1:Ny+1, 1:Nx+1]

    print("[{:d}] Plotting...".format(i))

    fig = plt.figure(figsize=(16, 9))

    ax = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=3, projection="3d")
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.25)

    data_z_slice = data[0, :, :]
    data_x_slice = data[:, :, -1]
    data_y_slice = data[:, 0, :]

    XC_z, YC_z = meshgrid(linspace(0, Lx, Nx), linspace(0, Ly, Ny))
    YC_x, ZC_x = meshgrid(linspace(0, Ly, Ny), linspace(0, -Lz, Nz))
    XC_y, ZC_y = meshgrid(linspace(0, Lx, Nx), linspace(0, -Lz, Nz))

    x_offset, y_offset, z_offset = Lx/1000, 0, 0

    cf1 = ax.contourf(XC_z / 1000, YC_z / 1000, data_z_slice, zdir="z", offset=z_offset, levels=arange(vmin, vmax, contour_spacing), cmap=cmap)
    cf2 = ax.contourf(data_x_slice, YC_x / 1000, ZC_x, zdir="x", offset=x_offset, levels=arange(vmin, vmax, contour_spacing), cmap=cmap)
    cf3 = ax.contourf(XC_y / 1000, data_y_slice, ZC_y, zdir="y", offset=y_offset, levels=arange(vmin, vmax, contour_spacing), cmap=cmap)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    Z_offset = z_offset*ones((Ny, Nx))
    X_offset = x_offset*ones((Nz, Ny))
    Y_offset = y_offset*ones((Nz, Nx))


    clbax = fig.add_axes([0.64, 0.48, 0.015, 0.35])
    clb = fig.colorbar(cf3, ticks=[8, 9, 10, 11, 12], cax=clbax)  #, shrink=0.9)
    clb.ax.set_title(r"T (°C)")

    ax.set_xlim3d(-125, 125 + Lx / 1000)
    ax.set_ylim3d(0, Ly / 1000)
    ax.set_zlim3d(-2*Lz, 0)

    ax.set_title("{:02.2f} days".format(t / (3600*24)), x=0.27, y=0.94)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (m)")

    ax.view_init(elev=30, azim=-50)
    ax.set_axis_off()

    print("[{:d}] Saving plot...".format(i))

    fig.savefig(fout(i//40), transparent=False, format="png", dpi=500)

    plt.close("all")

if __name__ == "__main__":
    Parallel(n_jobs=50)(
        delayed(plot_frame)(i) for i in range(40, 415000, 40)
    )
