using Statistics: mean
using Oceananigans.Fields: compute!

struct WindowedSpatialAverage{F, S, D}
          field :: F
   field_slicer :: S
           dims :: D
end

"""
    WindowedSpatialAverage(field; dims, field_slicer=FieldSlicer())

Builds a `WindowedSpatialAverage` of `field` that returns the average of `field` over a 

slice defined by `field_sicer`.


Example
=======

```jldoctest
using Oceananigans
using Oceananigans.Diagnostics: WindowedSpatialAverage, FieldSlicer
using Oceananigans.OutputWriters: AveragedTimeInterval, NetCDFOutputWriter

grid = RegularRectilinearGrid(size=(4, 6, 4), extent=(1,1,1))
model = IncompressibleModel(grid=grid)

set!(model.velocities.u, 1)

slicer = FieldSlicer(j=3:6, k=1)

U_wsa = WindowedSpatialAverage(model.velocities.u; dims=(1, 2), field_slicer=slicer)

simulation = Simulation(model, Δt=10, stop_iteration=10)
simulation.output_writers[:simple_output] = NetCDFOutputWriter(model, (U_wsa=U_wsa,), 
                                                               schedule = 10,
                                                               filepath = "windowed_spatial_average_jldoctest.nc")
```
"""
WindowedSpatialAverage(field; dims, field_slicer=FieldSlicer()) = WindowedSpatialAverage(field, field_slicer, dims)

function (wsa::WindowedSpatialAverage)(model)
    compute!(wsa.field)
    window = slice_parent(wsa.field_slicer, wsa.field)
    return dropdims(mean(window, dims=wsa.dims), dims=wsa.dims)
end
