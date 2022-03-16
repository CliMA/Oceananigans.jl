module Fields

export Face, Center
export AbstractField, Field, Average, Integral, Reduction, field
export CenterField, XFaceField, YFaceField, ZFaceField
export BackgroundField
export interior, data, xnode, ynode, znode, location
export set!, compute!, @compute, regrid!
export VelocityFields, TracerFields, tracernames, PressureFields, TendencyFields
export interpolate

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.BoundaryConditions

include("abstract_field.jl")
include("constant_field.jl")
include("function_field.jl")
include("field_boundary_buffers.jl")
include("field.jl")
include("field_reductions.jl")
include("regridding_fields.jl")
include("field_tuples.jl")
include("background_fields.jl")
include("interpolate.jl")
include("show_fields.jl")
include("broadcasting_abstract_fields.jl")

"""
    field(loc, a, grid)

Build a field from `a` at `loc` and on `grid`.
"""
function field(loc, a::AbstractArray, grid)
    f = Field(loc, grid)
    a = arch_array(architecture(grid), a)
    try
        copyto!(parent(f), a)
    catch
        f .= a
    end
    return f
end

field(loc, a::Function, grid) = FunctionField(loc, a, grid)

function field(loc, f::Field, grid)
    loc === location(f) && grid === f.grid && return f
    error("Cannot construct field at $loc and on $grid from $f")
end

include("set!.jl")

end # module
