module Fields

export Face, Center
export AbstractField, Field, Average, Integral, Reduction, field
export CenterField, XFaceField, YFaceField, ZFaceField
export BackgroundField
export interior, data, xnode, ynode, znode, location
export set!, compute!, @compute, regrid!
export VelocityFields, TracerFields, tracernames, PressureFields, TendencyFields
export interpolate, FieldSlicer

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.BoundaryConditions

import Oceananigans: short_show

include("abstract_field.jl")
include("zero_field.jl")
include("function_field.jl")
include("field.jl")
include("field_reductions.jl")
include("regridding_fields.jl")
include("field_tuples.jl")
include("background_fields.jl")
include("interpolate.jl")
include("field_slicer.jl")
include("show_fields.jl")
include("broadcasting_abstract_fields.jl")

"""
    field(loc, a, grid)

Build a field from `a` at `loc` and on `grid`.
"""
function field(loc, a::Array, grid)
    f = Field(loc, grid)
    f .= a
    return f
end

field(loc, a::Function, grid) = FunctionField(loc, a, grid)

function field(loc, f::Field, grid)
    loc === location(f) && grid === f.grid && return f
    error("Cannot construct field at $loc and on $grid from $f")
end

include("set!.jl")

end # module
