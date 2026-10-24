using Oceananigans.Fields: AbstractField, Field, fill_halo_regions!, interior

#####
##### TopLoad: the weight of a floating solid top (e.g. an ice shelf)
#####

struct TopLoad{B, R}
    buoyancy :: B
    reference_tracers :: R
end

"""
$(TYPEDSIGNATURES)

Return a `top_load` for a [`GridFittedBottom`](@ref) or [`PartialCellBottom`](@ref) with an immersed
top that is computed by [`top_load_potential`](@ref) from `buoyancy` and `reference_tracers` when the
`ImmersedBoundaryGrid` is built. A fluid whose stratification equals `reference_tracers` is then at rest.

Example
=======

```jldoctest
julia> using Oceananigans

julia> underlying_grid = RectilinearGrid(size=(1, 4, 10), x=(0, 1), y=(0, 4), z=(-100, 0), topology=(Periodic, Bounded, Bounded));

julia> top_height(x, y) = y < 2 ? -40 : 0;

julia> top_load = TopLoad(BuoyancyTracer(), (; b = (x, y, z) -> 1e-5 * z));

julia> grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-100; top_height, top_load));

julia> grid.immersed_boundary.top_load[1, 1:4, 1]
4-element Vector{Float64}:
 0.008
 0.008
 0.0
 0.0
```
"""
TopLoad(buoyancy, reference_tracers::NamedTuple) = TopLoad{typeof(buoyancy), typeof(reference_tracers)}(buoyancy, reference_tracers)

Base.summary(load::TopLoad) = string("TopLoad(", summary(load.buoyancy), ", reference_tracers=", keys(load.reference_tracers), ")")

validate_top_load(top_height, top_load) = nothing
validate_top_load(::Nothing, ::Nothing) = nothing
validate_top_load(::Nothing, top_load) = throw(ArgumentError("A top_load requires a top_height."))

materialize_top_load(grid, unloaded_ib, ::Nothing) = nothing

function materialize_top_load(grid, unloaded_ib, top_load)
    top_load_field = Field{Center, Center, Nothing}(grid)
    set_top_load!(top_load_field, top_load)
    fill_halo_regions!(top_load_field)
    return top_load_field.data
end

set_top_load!(top_load_field, top_load) = set_bottom_height!(top_load_field, top_load)
set_top_load!(top_load_field, top_load::AbstractField) = copyto!(interior(top_load_field), interior(top_load))
