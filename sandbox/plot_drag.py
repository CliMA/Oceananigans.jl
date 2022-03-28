import numpy as np
import xarray as xr
import pynanigans as pn



#++++ Options
plot = True
parallel = True
directions = ["x", "y", "z"]
velocities = ["u", "v", "w"]
#----


#++++ Define plotting function
def plotfunc(ds, fig, tt, framedim, y_dir = "z", vels=["u", "v"]):
    ds0 = ds.isel(time=tt)
    axes = fig.subplots(ncols=2, sharey=True)

    # Time series
    ds0[f"{vels[0]}_ddrag"].pnplot(ax=axes[0], y=y_dir, label="u (no IBM w/ drag)")
    ds0[f"{vels[0]}_immsd"].pnplot(ax=axes[0], y=y_dir, label="u (w/ IBM)", ls="--")

    ds0[f"{vels[1]}_ddrag"].pnplot(ax=axes[1], y=y_dir, label="u (no IBM w/ drag)")
    ds0[f"{vels[1]}_immsd"].pnplot(ax=axes[1], y=y_dir, label="u (w/ IBM)", ls="--")

    for i, ax in enumerate(axes):
        ax.set_xlim(0, None)
        ax.legend()
        ax.set_title(vels[i])
    return None, None
#----

for bounded_dir in directions:

    #++++ Get directions right
    per_directions = [ direc for direc in directions if direc is not bounded_dir ]
    per_velocities = [ veloc for direc, veloc in zip(directions, velocities) if direc is not bounded_dir ]
    #----
    
    #++++ Open dataset
    vid_ddrag = xr.load_dataset(f"{bounded_dir}_control_drag_model.nc", decode_times=False, engine="netcdf4")
    vid_immsd = xr.load_dataset(f"{bounded_dir}_immersed_model.nc", decode_times=False, engine="netcdf4")

    vels_ddrag = { f"{vel}_ddrag" : vid_ddrag[vel] for vel in per_velocities }
    vels_immsd = { f"{vel}_immsd" : vid_immsd[vel] for vel in per_velocities }
    ds = xr.Dataset(vels_ddrag | vels_immsd).squeeze()

    if parallel:
        ds = ds.chunk(dict(time=1))
    #----
    
    #++++
    if plot:
        from matplotlib import pyplot as plt
        from xmovie import Movie
        from dask.diagnostics import ProgressBar
        plt.rcParams['figure.constrained_layout.use'] = True
    
    
        mov = Movie(ds, plotfunc, input_check=False, 
                    y_dir=bounded_dir,
                    vels=per_velocities)

        with ProgressBar():
            mov.save(f"{bounded_dir}_drag.mp4", 
                     parallel=parallel, parallel_compute_kwargs=dict(), 
                     overwrite_existing=True,
                     )
    #----
