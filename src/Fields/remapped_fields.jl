using Oceananigans.Grids: default_indices, new_data, validate_indices
using Oceananigans.Utils: getregion

#####
##### Fields remapped onto another grid: a `remapping` precomputed once interpolates a source field, or the
##### components of a horizontal vector, onto the nodes of a destination field with `remap!`. Grids extend
##### `remapping` and `remap!` with their own weights and kernels; by default the nodes of the destination are
##### interpolated pointwise, with `interpolate!`.
#####

"""
    PointwiseInterpolation()

The remapping that evaluates the source field at each node of the destination with [`interpolate!`](@ref),
which needs no precomputation. It is the remapping between grids whose coordinates coincide, like two
`RectilinearGrid`s or two `LatitudeLongitudeGrid`s.
"""
struct PointwiseInterpolation end

Base.summary(::PointwiseInterpolation) = "PointwiseInterpolation"

"""
    remapping(destination_grid, destination_location, source)

Return the object that [`remap!`](@ref) uses to remap `source` onto the nodes at `destination_location`
of `destination_grid`, or the vector with components `source = (u, v)` onto their geographic components.
The remapping holds whatever is precomputed once for the pair of grids, such as interpolation stencils
and weights; by default it is a [`PointwiseInterpolation`](@ref), which holds nothing.
"""
remapping(destination_grid, destination_location, source::AbstractField) = PointwiseInterpolation()

remapping(destination_grid, destination_location, components::Tuple) =
    throw(ArgumentError("Remapping a vector from $(summary(first(components).grid)) is not supported: " *
                        "remap its components as scalars if the grids share their basis."))

remapping(destination::AbstractField, source) = remapping(destination.grid, instantiated_location(destination), source)

"""
    remap!(destination, source, remapping = remapping(destination, source))
    remap!((uᵍ, vᵍ), (u, v), remapping = remapping(uᵍ, (u, v)))

Remap the scalar `source` onto the nodes of `destination`, or the horizontal vector with components
`(u, v)` onto its geographic components `(uᵍ, vᵍ)`, which share a location, with `remapping`.
Only the nodes within the `indices` of the destination are remapped; the source and destination
share their vertical nodes. Fields on grids with rotated bases, such as a `YinYangGrid`, must be
remapped as vectors if they are the components of one.
"""
function remap!(destination::AbstractField, source, remapping = remapping(destination, source))
    remap!(destination, source, remapping, nothing)
    return destination
end

function remap!(geographic::Tuple{AbstractField, AbstractField}, components::Tuple,
                remapping = remapping(first(geographic), components))
    remap!(geographic[1], components, remapping, Val(1))
    remap!(geographic[2], components, remapping, Val(2))
    return geographic
end

# The scalar remapping (`component::Nothing`), or the eastward (`Val(1)`) or northward (`Val(2)`)
# component of a vector; extended by grids with their own remappings
remap!(destination, source, ::PointwiseInterpolation, ::Nothing) = interpolate!(destination, source)

#####
##### Remapped fields
#####

"""
    Remap(source, remapping, component)

The operand of a remapped field: `source` is remapped with `remapping`, as a scalar if `component` is
`nothing`, or as the eastward (`Val(1)`) or northward (`Val(2)`) component of the vector `source = (u, v)`.
"""
struct Remap{S, R, C}
    source :: S
    remapping :: R
    component :: C
end

remapping_name(remapping) = summary(remapping)

Base.summary(remap::Remap{<:Tuple}) = string("Remap of the ", remap.component isa Val{1} ? "eastward" : "northward",
                                            " component of a vector with ", remapping_name(remap.remapping))
Base.summary(remap::Remap) = string("Remap of ", summary(remap.source), " with ", remapping_name(remap.remapping))

const RemappedField = Field{<:Any, <:Any, <:Any, <:Remap}

# The indices of a field remapped from `source`: the vertical window of the source, which the grids
# share, and the horizontal `indices` of the destination, whose nodes are the source's only by coincidence
function remapped_indices(source, destination_indices)
    source_indices = getregion(indices(source), 1)
    windowed = !(source_indices[1] isa Colon) || !(source_indices[2] isa Colon)
    windowed && throw(ArgumentError("Cannot remap $(summary(source)) with indices $source_indices: " *
                                    "remapped fields may be windowed only in the vertical."))
    return (destination_indices[1], destination_indices[2],
            compute_index_intersection(destination_indices[3], source_indices[3]))
end

"""
    remapped_field(source, destination_grid; indices = (:, :, :))

Return a computed field that holds `source`, a field or an operation on another grid, remapped at its own
location onto `destination_grid` with the [`remapping`](@ref) of the pair of grids, or the pair of
geographic components at cell centers of the horizontal vector `source = (u, v)`. The field is windowed
to `indices` on `destination_grid`, within the vertical window of `source`, and is recomputed from `source`
by `compute!`, so it can be reduced, composed into operations, and written by output writers like any
other field.

Example
=======

A field on a `YinYangGrid` is remapped onto a `LatitudeLongitudeGrid` with the spacing of its panels:

```jldoctest
using Oceananigans

grid = YinYangGrid(size=(48, 16, 1), z=(-1, 0))
c = CenterField(grid)
set!(c, (λ, φ, z) -> sind(φ))

cᵍ = remapped_field(c, LatitudeLongitudeGrid(size=(64, 32, 1), longitude=(-180, 180), latitude=(-90, 90), z=(-1, 0)))

# output
64×32×1 Field{Center, Center, Center} on LatitudeLongitudeGrid on CPU
├── grid: 64×32×1 LatitudeLongitudeGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×1 halo
├── boundary conditions: Nothing
├── operand: Remap of 54×22×1 Field{Center, Center, Center} on YinYangGrid on CPU with YinYangRemapping
├── status: time=0.0
└── data: 70×38×3 OffsetArray(::Array{Float64, 3}, -2:67, -2:35, 0:2) with eltype Float64 with indices -2:67×-2:35×0:2
    └── max=0.998793, min=-0.998793, mean=-1.56125e-17
```
"""
function remapped_field(source::AbstractField, destination_grid; indices = default_indices(3))
    validate_remapping_destination(destination_grid)
    loc = instantiated_location(source)
    indices = remapped_indices(source, validate_indices(indices, loc, destination_grid))
    method = remapping(destination_grid, loc, source)
    return remapped_field(source, method, destination_grid, loc, indices, nothing)
end

function remapped_field(components::Tuple{AbstractField, AbstractField}, destination_grid; indices = default_indices(3))
    validate_remapping_destination(destination_grid)
    u, v = components
    loc = (Center(), Center(), instantiated_location(u)[3])
    indices = remapped_indices(u, validate_indices(indices, loc, destination_grid))
    indices == remapped_indices(v, indices) ||
        throw(ArgumentError("The components of a remapped vector must share their vertical window."))
    method = remapping(destination_grid, loc, components)
    return map(component -> remapped_field(components, method, destination_grid, loc, indices, component), (Val(1), Val(2)))
end

# Extended by grids that cannot be remapped onto
validate_remapping_destination(destination_grid) = nothing

remapped_field(source::Scan, destination_grid; kw...) = remapped_field(Field(source), destination_grid; kw...)

function remapped_field(source, remapping, grid, loc, indices, component)
    indices = validate_indices(indices, loc, grid)
    data = new_data(eltype(grid), grid, loc, indices)
    operand = Remap(source, remapping, component)
    return compute!(Field(loc, grid, data, nothing, indices, operand, FieldStatus()))
end

function compute!(field::RemappedField, time = nothing)
    remap = field.operand
    compute_at!(remap.source, time)
    remap!(field, remap.source, remap.remapping, remap.component)
    set_status!(field.status, time)
    return field
end
