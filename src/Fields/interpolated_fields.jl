using KernelAbstractions: @index, @kernel
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: _node, default_indices, interior_z_indices, new_data, topology, validate_indices
using Oceananigans.Utils: KernelParameters, getregion, interpolator, launch!, _interpolate

#####
##### Fields interpolated onto another grid: an interpolation method, precomputed once for a pair of grids,
##### interpolates a source field or the components of a horizontal vector onto the nodes of a destination
##### field. Grids extend `interpolation_method` and `interpolate_with!` with their own stencils and kernels;
##### by default the nodes of the destination are interpolated pointwise, in the horizontal, at the vertical
##### index that the grids share. Conservative regridding is `regrid!`.
#####

"""
    PointwiseInterpolation()

Interpolate the source field horizontally onto each node of the destination, at the vertical index
that the source and destination share, with no precomputation. This is the interpolation method between
grids whose coordinates coincide, like two `RectilinearGrid`s or two `LatitudeLongitudeGrid`s.
"""
struct PointwiseInterpolation end

Base.summary(::PointwiseInterpolation) = "PointwiseInterpolation"

"""
    interpolation_method(destination_grid, destination_location, source)

Return the interpolation method that interpolates `source` onto the nodes at `destination_location` of
`destination_grid`, or the vector with components `source = (u, v)` onto its geographic components.
The method holds whatever is precomputed once for the pair of grids, such as stencils and weights;
by default it is a [`PointwiseInterpolation`](@ref), which holds nothing.
"""
interpolation_method(destination_grid, destination_location, source::AbstractField) = PointwiseInterpolation()

interpolation_method(destination_grid, destination_location, components::Tuple) =
    throw(ArgumentError("Interpolating a vector from $(summary(first(components).grid)) is not supported: " *
                        "interpolate its components as scalars if the grids share their basis."))

interpolation_method(destination::AbstractField, source) =
    interpolation_method(destination.grid, instantiated_location(destination), source)

"""
    interpolate!((uᵍ, vᵍ), (u, v), method = interpolation_method(uᵍ, (u, v)))

Interpolate the horizontal vector with components `(u, v)` onto its geographic components `(uᵍ, vᵍ)`,
which share a location. Fields on grids with rotated bases, such as a `YinYangGrid`, must be
interpolated as vectors if they are the components of one.
"""
function interpolate!(geographic::Tuple{AbstractField, AbstractField}, components::Tuple,
                      method = interpolation_method(first(geographic), components))
    interpolate_with!(geographic[1], components, method, Val(1))
    interpolate_with!(geographic[2], components, method, Val(2))
    return geographic
end

# The scalar interpolation (`component::Nothing`), or the eastward (`Val(1)`) or northward (`Val(2)`)
# component of a vector; extended by grids with their own interpolations. The pointwise interpolation interpolates
# the source horizontally at the vertical index of each destination node, rather than with `interpolate!`,
# whose vertical stencil reaches beyond the window of a source windowed in the vertical.
function interpolate_with!(destination, source, ::PointwiseInterpolation, ::Nothing)
    destination_grid = destination.grid
    source_grid = source.grid
    map(validate_flat_interpolation, topology(destination_grid)[1:2], topology(source_grid)[1:2], (1, 2))

    Base.size(destination_grid, 3) == Base.size(source_grid, 3) ||
        throw(ArgumentError("The destination and source grids have different vertical sizes."))
    ks = interior_z_indices(destination)
    ks ⊆ interior_z_indices(source) ||
        throw(ArgumentError("The vertical indices $ks of the destination are not held by the source."))

    params = KernelParameters(interior_indices(destination))
    launch!(architecture(destination), destination_grid, params, _interpolate_horizontally!,
            destination, destination_grid, instantiated_location(destination),
            source, source_grid, instantiated_location(source))

    return nothing
end

@kernel function _interpolate_horizontally!(destination, destination_grid, destination_location,
                                            source, source_grid, source_location)
    i, j, k = @index(Global, NTuple)

    # The horizontal node of the destination, omitting its `Nothing` and `Flat` directions like `_interpolate!`
    horizontal_node = _node(i, j, k, destination_grid, destination_location[1], destination_location[2], nothing)
    horizontal_node = flatten_node(horizontal_node...)

    fidx = FractionalIndices(horizontal_node, source_grid, source_location[1], source_location[2], nothing)
    ix = interpolator(fidx.i)
    iy = interpolator(fidx.j)
    @inbounds destination[i, j, k] = _interpolate(source, ix, iy, (k, k, 0))
end

#####
##### Interpolated fields
#####

"""
    Interpolate(source, method, component)

The operand of an interpolated field: `source` is interpolated with `method`, as a scalar if `component` is
`nothing`, or as the eastward (`Val(1)`) or northward (`Val(2)`) component of the vector `source = (u, v)`.
"""
struct Interpolate{S, M, C}
    source :: S
    method :: M
    component :: C
end

interpolation_name(method) = summary(method)

function Base.summary(operand::Interpolate{<:Tuple})
    direction = operand.component isa Val{1} ? "eastward" : "northward"
    return string("Interpolation of the ", direction, " component of a vector with ", interpolation_name(operand.method))
end

Base.summary(operand::Interpolate) =
    string("Interpolation of ", summary(operand.source), " with ", interpolation_name(operand.method))

const InterpolatedField = Field{<:Any, <:Any, <:Any, <:Interpolate}

# The indices of a field interpolated from `source`: the vertical window of the source, which the grids
# share, and the horizontal `indices` of the destination, whose nodes are the source's only by coincidence
function interpolated_indices(source, destination_indices)
    source_indices = getregion(indices(source), 1)
    windowed = !(source_indices[1] isa Colon) || !(source_indices[2] isa Colon)
    windowed && throw(ArgumentError("Cannot interpolate $(summary(source)) with indices $source_indices: " *
                                    "interpolated fields may be windowed only in the vertical."))
    return (destination_indices[1], destination_indices[2],
            compute_index_intersection(destination_indices[3], source_indices[3]))
end

"""
    interpolated_field(source, destination_grid; indices = (:, :, :))

Return a computed field that holds `source`, a field or an operation on another grid, interpolated at its own
location onto `destination_grid` with the [`interpolation_method`](@ref) of the pair of grids, or the pair of
geographic components at cell centers of the horizontal vector `source = (u, v)`. The field is windowed
to `indices` on `destination_grid`, within the vertical window of `source`, and is recomputed from `source`
by `compute!`, so it can be reduced, composed into operations, and written by output writers like any
other field.

Example
=======

A field on a `YinYangGrid` is interpolated onto a `LatitudeLongitudeGrid` with the spacing of its panels:

```jldoctest
using Oceananigans

grid = YinYangGrid(size=(48, 16, 1), z=(-1, 0))
c = CenterField(grid)
set!(c, (λ, φ, z) -> cosd(φ))

cᵍ = interpolated_field(c, LatitudeLongitudeGrid(size=(64, 32, 1), longitude=(-180, 180), latitude=(-90, 90), z=(-1, 0)))

# output
64×32×1 Field{Center, Center, Center} on LatitudeLongitudeGrid on CPU
├── grid: 64×32×1 LatitudeLongitudeGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×1 halo
├── boundary conditions: Nothing
├── operand: Interpolation of 54×22×1 Field{Center, Center, Center} on YinYangGrid on CPU with YinYangGeographicInterpolation
├── status: time=0.0
└── data: 70×38×3 StaticStridesOffsetArray(::Array{Float64, 3}, -2:67, -2:35, 0:2) with eltype Float64 with indices -2:67×-2:35×0:2
    └── max=0.998795, min=0.0585736, mean=0.63742
```
"""
function interpolated_field(source::AbstractField, destination_grid; indices = default_indices(3))
    validate_interpolation_destination(destination_grid)
    loc = instantiated_location(source)
    indices = interpolated_indices(source, validate_indices(indices, loc, destination_grid))
    method = interpolation_method(destination_grid, loc, source)
    return interpolated_field(source, method, destination_grid, loc, indices, nothing)
end

function interpolated_field(components::Tuple{AbstractField, AbstractField}, destination_grid; indices = default_indices(3))
    validate_interpolation_destination(destination_grid)
    u, v = components
    loc = (Center(), Center(), instantiated_location(u)[3])
    indices = interpolated_indices(u, validate_indices(indices, loc, destination_grid))
    indices == interpolated_indices(v, indices) ||
        throw(ArgumentError("The components of a interpolated vector must share their vertical window."))
    method = interpolation_method(destination_grid, loc, components)
    return map(component -> interpolated_field(components, method, destination_grid, loc, indices, component), (Val(1), Val(2)))
end

# Extended by grids that cannot be interpolated onto
validate_interpolation_destination(destination_grid) = nothing

interpolated_field(source::Scan, destination_grid; kw...) = interpolated_field(Field(source), destination_grid; kw...)

function interpolated_field(source, method, grid, loc, indices, component)
    indices = validate_indices(indices, loc, grid)
    data = new_data(eltype(grid), grid, loc, indices)
    operand = Interpolate(source, method, component)
    return compute!(Field(loc, grid, data, nothing, indices, operand, FieldStatus()))
end

function compute!(field::InterpolatedField, time = nothing)
    operand = field.operand
    compute_at!(operand.source, time)
    interpolate_with!(field, operand.source, operand.method, operand.component)
    set_status!(field.status, time)
    return field
end
