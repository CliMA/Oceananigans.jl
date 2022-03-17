import numpy as np
import xarray as xr
import pynanigans as pn



#++++ Options
plot = True
parallel = False
#----


#++++ Define plotting function
def plotfunc(ds, fig, tt, *args, **kwargs):
    ds0 = ds.isel(time=tt)
    ax1 = fig.subplots(ncols=1,)

    # Time series
    ds0.u_cdrag.pnplot(ax=ax1, y='z', label="u (no IBM w/ drag)")
    ds0.u_noslip.pnplot(ax=ax1, y='z', label="u (no IBM w/ no slip)")
    ds0.u_imsd.pnplot(ax=ax1, y='z', label="u (w/ IBM)")
    ax1.set_xlim(0, None)
    ax1.legend()

    return None, None
#----


#++++ Open dataset
vid_cdrag = xr.load_dataset("control_drag_model.nc", decode_times=False, engine="netcdf4").squeeze()
vid_noslip = xr.load_dataset("control_noslip_model.nc", decode_times=False, engine="netcdf4").squeeze()
vid_imsd = xr.load_dataset("immersed_model.nc", decode_times=False, engine="netcdf4").squeeze()
ds = xr.Dataset(dict(u_cdrag=vid_cdrag.u, u_noslip=vid_noslip.u, u_imsd=vid_imsd.u))
if parallel:
    ds = ds.chunk(dict(time=1))
#----

#++++
if plot:
    from matplotlib import pyplot as plt
    from xmovie import Movie
    from dask.diagnostics import ProgressBar
    plt.rcParams['figure.constrained_layout.use'] = True


    mov = Movie(ds, plotfunc, input_check=False)
    with ProgressBar():
        mov.save(f"u_drag.mp4", 
                 parallel=parallel, parallel_compute_kwargs=dict(), 
                 overwrite_existing=True)
#----
