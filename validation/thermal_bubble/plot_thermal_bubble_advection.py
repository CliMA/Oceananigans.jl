import os
import joblib
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import ffmpeg

# schemes = ("CenteredSecondOrder", "CenteredFourthOrder", "WENO")
schemes = ("WENO",)
Ns = (32, 128)

def plot_tracer_advection_2d_frame(scheme, N, n):
    dir = f"thermal_bubble_{scheme}_N{N:d}"
    filename = dir + ".nc"
    ds = xr.open_dataset(filename)
    
    u = ds.u.isel(time=n).squeeze()
    w = ds.w.isel(time=n).squeeze()
    T = ds.T.isel(time=n).squeeze()
    
    print(f"{dir} frame {n}/{len(ds.time)}")
    
    fig, ax = plt.subplots(figsize=(9, 9))
    T.plot.pcolormesh(vmin=20.00, vmax=20.01, cmap=cmocean.cm.balance, extend="both")
    ax.set_title(f"{scheme} N={N:d} ")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal")

    plt.savefig(f"{dir}/{dir}_T_{n:03d}.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 9))
    u.plot.pcolormesh(cmap=cmocean.cm.balance, extend="both")
    ax.set_title(f"{scheme} N={N:d} ")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal")

    plt.savefig(f"{dir}/{dir}_u_{n:03d}.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 9))
    w.plot.pcolormesh(cmap=cmocean.cm.balance, extend="both")
    ax.set_title(f"{scheme} N={N:d} ")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal")

    plt.savefig(f"{dir}/{dir}_w_{n:03d}.png")
    plt.close()

for scheme in schemes:
    for N in Ns:
        dir = f"thermal_bubble_{scheme}_N{N:d}"
        if not os.path.exists(dir):
            os.mkdir(dir)

        filename = dir + ".nc"
        ds = xr.open_dataset(filename)

        joblib.Parallel(n_jobs=-1)(
            joblib.delayed(plot_tracer_advection_2d_frame)(scheme, N, n)
            for n in range(ds.time.size)
        )

        for var in ("T", "u", "w"):
            (
                ffmpeg
                .input(f"{dir}/{dir}_{var}_%03d.png", framerate=30)
                .output(f"{dir}/{dir}_{var}.mp4", pix_fmt='yuv420p')
                .overwrite_output()
                .run()
            )
