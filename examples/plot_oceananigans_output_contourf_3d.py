import matplotlib

matplotlib.use('Agg')

import numpy as np
from numpy import ones, meshgrid, linspace
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.switch_backend('Agg')

from joblib import Parallel, delayed

data_dir = '/nobackup1b/users/alir/Oceananigans/'

def process_iter(iter):
    # Enforcing style in here so that it's applied to all workers launched by joblib.
    plt.style.use('dark_background')

    dt = 0.1
    output_freq = 36000
    vmin, vmax = 19, 20

    time_str = str(dt*output_freq * iter).zfill(12)
    fname = "deep_convection_3d_T_" + time_str + ".dat"

    nc_output = xr.open_dataset("convection000072000.nc")

    Nx, Ny, Nz = nc_output["T"].values.shape
    Lx = nc_output["xF"].values[-1]
    Ly = nc_output["yF"].values[-1]
    Lz = nc_output["zF"].values[-1]

    xC = nc_output["xC"]
    yC = nc_output["yC"]
    zC = nc_output["zC"]

    data = nc_output["T"]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca(projection='3d')

    theta_z_slice = data.sel(yC=slice(0, Lx//2), zC=zC[0]).values - 273.15
    theta_x_slice = data.sel(xC=xC[-1], yC=slice(0, Lx//2)).values - 273.15
    theta_y_slice = data.sel(yC=yC.values[Ny//2]).values - 273.15

    XC_z, YC_z = meshgrid(linspace(0, Lx, Nx), linspace(0, Ly/2, Ny//2))
    YC_x, ZC_x = meshgrid(linspace(0, Ly/2, Ny//2), linspace(0, -Lz, Nz))
    XC_y, ZC_y = meshgrid(linspace(0, Lx, Nx), linspace(0, -Lz, Nz))

    x_offset, y_offset, z_offset = Lx, Ly/2, 0

    cf1 = ax.contourf(XC_z, YC_z, theta_z_slice, zdir='z', offset=z_offset, levels=np.arange(19, 20, 0.01), cmap='jet')
    cf2 = ax.contourf(theta_x_slice, YC_x, ZC_x, zdir='x', offset=x_offset, levels=np.arange(19, 20, 0.01), cmap='jet')
    cf3 = ax.contourf(XC_y, theta_y_slice, ZC_y, zdir='y', offset=y_offset, levels=np.arange(19, 20, 0.01), cmap='jet')

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    Z_offset = z_offset*np.ones((Ny//2, Nx))
    X_offset = x_offset*np.ones((Nz, Ny//2))
    Y_offset = y_offset*np.ones((Nz, Nx))

    clb = fig.colorbar(cf3, ticks=np.linspace(vmin, vmax, num=6))
    clb.ax.set_title(r"T (C)")

    ax.set_xlim3d(0, Lx)
    ax.set_ylim3d(0, Ly)
    ax.set_zlim3d(-Lz, 0)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    ax.view_init(elev=30, azim=45)

    ax.set_title("t = {:07d} s ({:03.2f} days)".format(int(dt*output_freq*iter), dt*output_freq*iter / (3600*24)), y=1.05)

    ax.set_xticks(np.linspace(0, Lx, num=5))
    ax.set_yticks(np.linspace(0, Ly, num=5))
    ax.set_zticks(np.linspace(0, -Lz, num=5))

    # plt.show()

    filename = 'surface_temp_3d_' + str(iter).zfill(5) + '.png'
    plt.savefig(filename, dpi=300, format='png', transparent=False)
    print('Saving: {:s}'.format(filename))

    plt.close('all')

# Parallel(n_jobs=8)(delayed(process_iter)(n) for n in np.arange(0, 423))
process_iter(2)
