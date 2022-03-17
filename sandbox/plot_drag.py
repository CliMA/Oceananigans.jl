import numpy as np
import xarray as xr


#++++ Preamble
filenames = ["immersed_stokes_first_problem_drag", "not_immersed_stokes_first_problem_drag"] 
#----

#++++ Options
plot = True
dmap = "RdBu_r"
smap = "viridis"
#----

for fname in filenames:
    #++++ Open dataset
    out = xr.load_dataset(fname, decode_times=False).chunk(dict(time=1)).squeeze()
    #----


    variables = ['v_tot', 'ω_y',]

    if plot:
        from matplotlib import pyplot as plt
        from xmovie import Movie
        from dask.diagnostics import ProgressBar

        for var in variables:
            vmin = vmins[var]
            vmax = vmaxs[var]
            mov = Movie(out[var], vmin=vmin, vmax=vmax, cmap=cmap)
            with ProgressBar():
                mov.save(f"{var}.mp4", 
                         parallel=True, parallel_compute_kwargs=dict(), 
                         overwrite_existing=True)

