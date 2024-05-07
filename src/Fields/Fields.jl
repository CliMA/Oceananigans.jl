module Fields

export Face, Center
export AbstractField, Field, Average, Integral, Reduction, Accumulation, field
export CenterField, XFaceField, YFaceField, ZFaceField
export BackgroundField
export interior, data, xnode, ynode, znode, location
export set!, compute!, @compute, regrid!
export VelocityFields, TracerFields, TendencyFields, tracernames
export interpolate

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.BoundaryConditions
using Oceananigans.Utils

import Oceananigans.Architectures: on_architecture

include("abstract_field.jl")
include("constant_field.jl")
include("function_field.jl")
include("field_boundary_buffers.jl")
include("field.jl")
include("reductions.jl")
include("regridding_fields.jl")
include("field_tuples.jl")
include("background_fields.jl")
include("interpolate.jl")
include("show_fields.jl")
include("broadcasting_abstract_fields.jl")

"""
    field(loc, a, grid)

Build a field from array `a` at `loc` and on `grid`.
"""
@inline function field(loc, a::AbstractArray, grid)
    f = Field(loc, grid)
    a = on_architecture(architecture(grid), a)
    try
        copyto!(parent(f), a)
    catch
        f .= a
    end
    return f
end

@inline field(loc, a::Function, grid) = FunctionField(loc, a, grid)
@inline field(loc, a::Number, grid) = ConstantField(a)

@inline function field(loc, f::Field, grid)
    loc === location(f) && grid === f.grid && return f
    error("Cannot construct field at $loc and on $grid from $f")
end

include("set!.jl")

end # module
