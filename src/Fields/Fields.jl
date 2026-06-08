module Fields

export Face, Center, location
export AbstractField, Field, Reduction, Accumulation, field
export CenterField, XFaceField, YFaceField, ZFaceField
export interior, data, xnode, ynode, znode
export set!, compute!, @compute, regrid!
export VelocityFields, TracerFields, tracernames
export interpolate

using OffsetArrays: OffsetArray
using Adapt: Adapt, adapt

using Oceananigans: Oceananigans, instantiated_location, location
using Oceananigans.Architectures: Architectures, child_architecture, on_architecture
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.Grids: Grids, AbstractGrid, Bounded, Center, Face, LatitudeLongitudeGrid,
    RectilinearGrid, new_data, interior_indices, total_size, topology, nodes, xnodes,
    ynodes, znodes, node, xnode, ynode, znode
using Oceananigans.Utils: KernelParameters, launch!, prettysummary, interpolator

"Return the location `(LX, LY, LZ)` of an `AbstractField{LX, LY, LZ}`."
@inline Oceananigans.location(a) = (Nothing, Nothing, Nothing) # used in AbstractOperations for location inference
@inline Oceananigans.location(a, i) = location(a)[i]
@inline function Oceananigans.instantiated_location(a)
    LX, LY, LZ = location(a)
    return (LX(), LY(), LZ())
end

include("abstract_field.jl")
include("constant_field.jl")
include("function_field.jl")
include("field.jl")
include("field_indices.jl")
include("scans.jl")
include("regridding_fields.jl")
include("field_tuples.jl")
include("interpolate.jl")
include("show_fields.jl")
include("broadcasting_abstract_fields.jl")

"""
    field(loc, a, grid)

Build a field from array `a` at `loc` and on `grid`.
"""
@inline function field(loc, a::AbstractArray, grid)
    loc = instantiate(loc)
    f = Field(loc, grid)
    a = on_architecture(architecture(grid), a)
    try
        set!(f, a)
    catch
        copyto!(parent(f), parent(a))
    end
    return f
end

# Build a field off of the current data
@inline function field(loc, a::OffsetArray, grid)
    loc = instantiate(loc)
    a = on_architecture(architecture(grid), a)
    return Field(loc, grid; data=a)
end

@inline field(loc, a::Function, grid) = FunctionField(loc, a, grid)
@inline field(loc, a::Number, grid) = ConstantField(a)
@inline field(loc, a::ZeroField, grid) = a
@inline field(loc, a::ConstantField, grid) = a

@inline function field(loc, f::Field, grid)
    loc = instantiate(loc)
    loc === instantiated_location(f) && grid === f.grid && return f

    msg = """
    Cannot reconstruct field, originally located at ($(instantiated_location(f))), at $loc.

    Destination grid:
    $grid

    Source grid:
    $(f.grid)
    """

    return throw(ArgumentError(msg))
end

include("set!.jl")

end # module
