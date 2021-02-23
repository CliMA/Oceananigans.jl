import Oceananigans.OutputWriters: slice_parent
using Statistics: mean

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

```julia
using Oceananigans.Diagnostics: WindowedSpatialAverage

slicer = FieldSlicer(j=Ny÷2+1:Ny)

U_wsa = WindowedSpatialAverage(u; dims=(1, 2), field_slicer=slicer)

simulation.output_writers[:simple_output] = NetCDFOutputWriter(model, (U_wsa=U_wsa,), 
                                                               schedule = AveragedTimeInterval(10seconds),
                                                               filepath = "file.nc")
```
"""

WindowedSpatialAverage(field; dims, field_slicer=FieldSlicer()) = WindowedSpatialAverage(field, field_slicer, dims)

function (wsa::WindowedSpatialAverage)(model)
    compute!(wsa.field)
    window = slice_parent(wsa.field_slicer, wsa.field)
    return dropdims(mean(window, dims=wsa.dims), dims=wsa.dims)
end


# The function below makes sure that the correct dimensions are automatically applied to
# the NetCDF output using NetCDFOutputWriter
using NCDatasets: defVar
using Oceananigans.Fields: reduced_location
import Oceananigans.OutputWriters: xdim, ydim, zdim, define_output_variable!
function define_output_variable!(dataset, 
                                 wtsa::Union{WindowedSpatialAverage, WindowedTimeAverage{<:WindowedSpatialAverage}}, 
                                 name, array_type, compression, attributes, dimensions)
    wsa = wtsa isa WindowedTimeAverage ? wtsa.operand : wtsa
    LX, LY, LZ = reduced_location(location(wsa.field), dims=wsa.dims)

    output_dims = tuple(xdim(LX)..., ydim(LY)..., zdim(LZ)...)
    defVar(dataset, name, eltype(array_type), (output_dims..., "time"),
           compression=compression, attrib=attributes)
    return nothing
end

