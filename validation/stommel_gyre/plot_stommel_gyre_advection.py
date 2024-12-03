import os
import joblib
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import ffmpeg

shapes = ("Gaussian", "Square")
schemes = ("CenteredSecondOrder", "CenteredFourthOrder", "WENO")
Ns = (32, 256)
CFLs = (0.05, 0.30)

def plot_tracer_advection_2d_frame(shape, scheme, N, CFL, n):
    dir = f"stommel_gyre_{shape}_{scheme}_N{N:d}_CFL{CFL:.2f}"
    filename = dir + ".nc"
    ds = xr.open_dataset(filename)
    c = ds.c.isel(time=n).squeeze()
    
    print(f"{dir} frame {n}/{len(ds.time)}")
    
    fig, ax = plt.subplots(figsize=(9, 9))
    c.plot.pcolormesh(vmin=-1, vmax=1, cmap=cmocean.cm.balance, extend="both")
    ax.set_title(f"{shape} {scheme} N={N:d} CFL={CFL:.2f}")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal")

    plt.savefig(f"{dir}/{dir}_c_{n:03d}.png")
    plt.close()

for shape in shapes:
    for scheme in schemes:
        for N in Ns:
            for CFL in CFLs:
                dir = f"stommel_gyre_{shape}_{scheme}_N{N:d}_CFL{CFL:.2f}"
                if not os.path.exists(dir):
                    os.mkdir(dir)

                filename = dir + ".nc"
                ds = xr.open_dataset(filename)

                joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(plot_tracer_advection_2d_frame)(shape, scheme, N, CFL, n)
                    for n in range(ds.time.size)
                )

                (
                    ffmpeg
                    .input(f"{dir}/{dir}_c_%03d.png", framerate=30)
                    .output(f"{dir}/{dir}_c.mp4", crf=15, pix_fmt='yuv420p')
                    .overwrite_output()
                    .run()
                )
