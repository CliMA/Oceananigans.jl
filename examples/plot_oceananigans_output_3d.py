import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.switch_backend('Agg')

from joblib import Parallel, delayed

data_dir = '/nobackup1b/users/alir/Oceananigans/'

def process_iter(iter):
    Nx, Ny, Nz = 400, 400, 200
    b = 8
    dt = 10
    output_freq = 10

    time_str = str(dt*output_freq * iter).zfill(12)
    fname = "deep_convection_3d_T_" + time_str + ".dat"

    with open(fname, "rb") as f:
        f.seek(0, os.SEEK_SET)
        data = np.fromfile(f, dtype="<f8", count=Nx*Ny*Nz)
        data = np.reshape(data, [Nx, Ny, Nz], order='F')


    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = Axes3D(fig)

    theta_z_slice = np.flipud(np.rot90(data[:, 0:200, 1])) - 273.15
    theta_x_slice = np.flipud(np.rot90(data[-1, 0:200, :])) - 273.15
    theta_y_slice = np.flipud(np.rot90(data[:, 200, :])) - 273.15

    # print(theta_z_slice.shape)
    # print(theta_x_slice.shape)
    # print(theta_y_slice.shape)

    XC_z, YC_z = np.meshgrid(np.linspace(0, 2000, 400), np.linspace(0, 1000, 200))
    YC_x, ZC_x = np.meshgrid(np.linspace(0, 1000, 200), np.linspace(0, -1000, 200))
    XC_y, ZC_y = np.meshgrid(np.linspace(0, 2000, 400), np.linspace(0, -1000, 200))

    # cf1 = ax.contourf(XC_z, YC_z, theta_z_slice.values, zdir='z', offset=z_offset, levels=np.arange(19.89, 20.01, 0.001), cmap='jet')
    # cf2 = ax.contourf(theta_x_slice.values, YC_x, ZC_x, zdir='x', offset=x_offset, levels=np.arange(19.89, 20.01, 0.001), cmap='jet')
    # cf3 = ax.contourf(XC_y, theta_y_slice.values, ZC_y, zdir='y', offset=y_offset, levels=np.arange(19.89, 20.01, 0.001), cmap='jet')

    norm = matplotlib.colors.Normalize(vmin=19.89, vmax=20.01)

    x_offset, y_offset, z_offset = 2000, 1000, 0

    Z_offset = z_offset*np.ones((200, 400))
    X_offset = x_offset*np.ones((200, 200))
    Y_offset = y_offset*np.ones((200, 400))

    # print(XC_z.shape)
    # print(YC_z.shape)
    # print(Z_offset.shape)

    cf1 = ax.plot_surface(XC_z, YC_z, Z_offset, rstride=1, cstride=1, linewidth=0, alpha=1, vmin=19.89, vmax=20.01,
                          antialiased=False, shade=False, facecolors=plt.cm.jet(norm(theta_z_slice)))
    cf2 = ax.plot_surface(X_offset, YC_x, ZC_x, rstride=1, cstride=1, linewidth=0, alpha=1, vmin=19.89, vmax=20.01,
                          antialiased=False, shade=False, facecolors=plt.cm.jet(norm(theta_x_slice)))
    cf3 = ax.plot_surface(XC_y, Y_offset, ZC_y, rstride=1, cstride=1, linewidth=0, alpha=1, vmin=19.89, vmax=20.01,
                          antialiased=False, shade=False, facecolors=plt.cm.jet(norm(theta_y_slice)))

    # clb = fig.colorbar(cf1, ticks=[19.90, 19.92, 19.94, 19.96, 19.98, 20.00])
    # clb.ax.set_title(r'Temperature (C)')

    ax.set_xlim3d(0, 2000)
    ax.set_ylim3d(0, 2000)
    ax.set_zlim3d(-1000, 0)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    ax.view_init(elev=30, azim=45)

    ax.set_title('t = {:07d} s ({:03.2f} days)'.format(200*iter, 200*iter / (3600*24)))

    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_yticks([0, 500, 1000, 1500, 2000])
    ax.set_zticks([0, -200, -400, -600, -800, -1000])

    # plt.show()

    filename = 'surface_temp_3d_' + str(iter).zfill(5) + '.png'
    plt.savefig(filename, dpi=300, format='png', transparent=False)
    print('Saving: {:s}'.format(filename))

    plt.close('all')

Parallel(n_jobs=8)(delayed(process_iter)(n) for n in np.arange(0, 423))
