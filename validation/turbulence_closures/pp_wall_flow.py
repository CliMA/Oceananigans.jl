import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
ds_smaglilly = xr.load_dataset("wall_flow_SmagorinskyLilly.nc", decode_times=False)
ds_scainvmag = xr.load_dataset("wall_flow_ScaleInvariantSmagorinsky.nc", decode_times=False)

H = 1
u_star = 1
κ = 0.4
z0 = 1e-4*H
û = (u_star / κ) * np.log(ds_smaglilly.zC / z0)

û.plot(x="zC")
ds_smaglilly.U.isel(time=-1).plot(x="zC", ls="--", xscale="log", label="Smag-Lilly")
ds_scainvmag.U.isel(time=-1).plot(x="zC", ls="--", xscale="log", label="ScaleInvSmag")
plt.ylim([10, 20])
plt.xlim([1e-3, 4e-1])
