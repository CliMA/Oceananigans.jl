using Oceananigans: instantiated_location
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!, fill_bottom_and_top_halo!
using Oceananigans.Utils: launch!, KernelParameters, Reference, Iterate, getregion, @apply_regionally
using Oceananigans.Grids: immersed_peripheral_node, immersed_inactive_node
using Oceananigans.ImmersedBoundaries: ImmersedBoundaries, ImmersedBoundaryGrid, GridFittedBoundary
using KernelAbstractions.Extras.LoopInfo: @unroll

# The halos of a mask would be interpolated as a scalar, which a `Bool` cannot hold: only bottoms are immersed
ImmersedBoundaries.materialize_immersed_boundary(::YinYangGrid, ::GridFittedBoundary) =
    throw(ArgumentError("GridFittedBoundary is not supported on a YinYangGrid; use GridFittedBottom or PartialCellBottom."))

#####
##### Halo filling between the Yin and Yang panels
#####
##### The donor stencils read only interior nodes, so the two panels can be filled in either order.
##### The vertical halos are filled first, so that the interpolation also fills those of the halo columns.
#####

without_boundary_conditions(bcs) = isnothing(bcs)
without_boundary_conditions(bcs::MultiRegionObject) = all(isnothing, bcs.regional_objects)

"""
    fill_halo_regions!(field::YinYangField, args...; kwargs...)

Fill the vertical halos of `field` from its bottom and top boundary conditions, which receive `args`,
then its horizontal halos by interpolation from the other panel. `field` is interpolated as a scalar:
the components of a horizontal vector must be filled together, as `fill_halo_regions!((u, v))`.
Interpolated vector components on immersed boundaries are zeroed; vertically integrated
components are zeroed where the entire column is immersed.
Scalar stencils that cross immersed cells use normalized positive weights from wet donors.
Fully wet scalar stencils retain the chosen interpolation order.
Fields windowed or reduced in `x` or `y` get only their vertical halos.
"""
function BoundaryConditions.fill_halo_regions!(field::YinYangField, args::Vararg{Any, N}; kwargs...) where N
    without_boundary_conditions(field.boundary_conditions) && return nothing

    grid = field.grid
    ℓx, ℓy, _ = instantiated_location(field)
    table = Reference(interpolation_table(yin_yang_interpolation(grid), ℓx, ℓy))
    multiregion_field = scalar_halo_source(field, grid)
    regions = Iterate(1:2)

    @apply_regionally fill_yin_yang_vertical_halo!(grid, field, args...)
    @apply_regionally fill_yin_yang_scalar_halo!(grid, field, multiregion_field, regions, table)

    return nothing
end

scalar_halo_source(field, grid) = Reference(field.data.regional_objects)
scalar_halo_source(field, grid::ImmersedMultiRegionGrid) =
    Reference((; data=field.data.regional_objects, grids=(getregion(grid, 1), getregion(grid, 2))))

const HorizontallyReducedYinYangField = Union{YinYangField{<:Nothing}, YinYangField{<:Any, <:Nothing}}

function BoundaryConditions.fill_halo_regions!(field::HorizontallyReducedYinYangField, args::Vararg{Any, N}; kwargs...) where N
    without_boundary_conditions(field.boundary_conditions) && return nothing
    @apply_regionally fill_yin_yang_vertical_halo!(field.grid, field, args...)
    return nothing
end

fill_yin_yang_vertical_halo!(grid, field, args::Vararg{Any, N}) where N =
    fill_bottom_and_top_halo!(field.data, field.boundary_conditions, field.indices, instantiated_location(field), grid, args...)

function fill_yin_yang_scalar_halo!(grid, field, multiregion_field, region, table)
    donor = multiregion_field[3 - region]
    launch_yin_yang_interpolation!(grid, field, table, _interpolate_yin_yang_scalar_halo!, field, donor, table)
    return nothing
end

function fill_yin_yang_scalar_halo!(grid::ImmersedBoundaryGrid, field, source, region, table)
    donor = source.data[3 - region]
    donor_grid = source.grids[3 - region]
    loc = instantiated_location(field)
    launch_yin_yang_interpolation!(grid, field, table, _interpolate_immersed_yin_yang_scalar_halo!,
                                   field, donor, donor_grid, loc, table)
    return nothing
end

function launch_yin_yang_interpolation!(grid, field, table, kernel!, args::Vararg{Any, N}) where N
    field.indices isa Tuple{Colon, Colon, Any} || return nothing # horizontal windows have no seam halos

    ks = axes(field.data, 3)
    kernel_parameters = KernelParameters((length(table), length(ks)), (0, first(ks) - 1))
    launch!(architecture(grid), grid, kernel_parameters, kernel!, args...)

    return nothing
end

@kernel function _interpolate_yin_yang_scalar_halo!(c, donor, table)
    n, k = @index(Global, NTuple)

    @inbounds begin
        i = table.receivers.i[n]
        j = table.receivers.j[n]
        c[i, j, k] = interpolate_donor(donor, table, n, k)
    end
end

@inline scalar_mask_index(k, ::Center, grid) = clamp(k, 1, grid.Nz)
@inline scalar_mask_index(k, ::Face, grid) = clamp(k, 1, grid.Nz + 1)

@inline function immersed_scalar_donor(i, j, k, grid, ℓx, ℓy, ℓz)
    k′ = scalar_mask_index(k, ℓz, grid)
    return immersed_inactive_node(i, j, k′, grid, ℓx, ℓy, ℓz)
end

@inline function immersed_scalar_donor(i, j, k, grid, ℓx, ℓy, ::Nothing)
    immersed = true
    for k in 1:grid.Nz
        immersed &= immersed_inactive_node(i, j, k, grid, ℓx, ℓy, Center())
    end
    return immersed
end

@kernel function _interpolate_immersed_yin_yang_scalar_halo!(c, donor, donor_grid, loc, table)
    n, k = @index(Global, NTuple)
    @inbounds begin
        i = table.receivers.i[n]
        j = table.receivers.j[n]
        c[i, j, k] = interpolate_wet_donor(donor, donor_grid, loc, table, n, k)
    end
end

@inline function interpolate_wet_donor(donor, grid, loc, table::YinYangInterpolationTable{S}, n, k) where S
    value = interpolate_donor(donor, table, n, k)
    wet_value = zero(value)
    wet_weight = zero(value)
    all_wet = true
    @inbounds begin
        i₀, j₀ = table.donors.i[n], table.donors.j[n]
        @unroll for b in 1:S
            @unroll for a in 1:S
                i, j = i₀ + a - 1, j₀ + b - 1
                immersed = immersed_scalar_donor(i, j, k, grid, loc...)
                weight = ifelse(immersed, zero(value), max(0, table.weights.x[n, a] * table.weights.y[n, b]))
                wet_value += weight * ifelse(immersed, zero(value), donor[i, j, k])
                wet_weight += weight
                all_wet &= !immersed
            end
        end
    end
    # A convex wet-only stencil preserves constants without amplifying masked coastal data.
    coastal_value = wet_value / ifelse(wet_weight > 0, wet_weight, one(wet_weight))
    return ifelse(all_wet, value, coastal_value)
end

@inline function interpolate_donor(donor, table::YinYangInterpolationTable{S}, n, k) where S
    @inbounds begin
        i₀ = table.donors.i[n]
        j₀ = table.donors.j[n]
        wx = table.weights.x
        wy = table.weights.y

        value = zero(eltype(donor))
        @unroll for b in 1:S
            row = zero(eltype(donor))
            @unroll for a in 1:S
                row += wx[n, a] * donor[i₀ + a - 1, j₀ + b - 1, k]
            end
            value += wy[n, b] * row
        end
    end

    return value
end

#####
##### Horizontal vectors
#####

const YinYangXComponent = YinYangField{<:Face, <:Center}
const YinYangYComponent = YinYangField{<:Center, <:Face}

BoundaryConditions.fill_halo_regions!(fields::Tuple{YinYangField, Vararg{YinYangField}}, args::Vararg{Any, N}; kwargs...) where N =
    foreach(field -> fill_halo_regions!(field, args...; kwargs...), fields)

BoundaryConditions.fill_halo_regions!((u, v)::Tuple{YinYangXComponent, YinYangYComponent}, args::Vararg{Any, N}; kwargs...) where N =
    fill_halo_regions!(u, v, args...; kwargs...)

"""
    fill_halo_regions!(u::YinYangField{Face, Center}, v::YinYangField{Center, Face}, args...; signed = true, kwargs...)
    fill_halo_regions!(u::YinYangField{Face, Face}, v::YinYangField{Face, Face}, args...; signed = true, kwargs...)

Fill the halos of the horizontal vector `(u, v)`: both components of the other panel are interpolated to each
halo node and rotated into this panel's basis. With `signed = false`, `u` and `v` are filled as scalars.
"""
function BoundaryConditions.fill_halo_regions!(u::YinYangXComponent, v::YinYangYComponent,
                                               args::Vararg{Any, N}; signed = true, kwargs...) where N
    interpolation = yin_yang_interpolation(u.grid)
    tables = (interpolation.face_center,
              interpolation.face_center_from_center_face,
              interpolation.center_face_from_face_center,
              interpolation.center_face)

    return fill_yin_yang_vector_halos!(u, v, tables, signed, args...; kwargs...)
end

function BoundaryConditions.fill_halo_regions!(u::YinYangField{<:Face, <:Face}, v::YinYangField{<:Face, <:Face},
                                               args::Vararg{Any, N}; signed = true, kwargs...) where N
    table = yin_yang_interpolation(u.grid).face_face
    return fill_yin_yang_vector_halos!(u, v, (table, table, table, table), signed, args...; kwargs...)
end

function fill_yin_yang_vector_halos!(u, v, tables, signed, args::Vararg{Any, N}; kwargs...) where N
    if !signed
        fill_halo_regions!(u, args...; kwargs...)
        fill_halo_regions!(v, args...; kwargs...)
        return nothing
    end

    without_boundary_conditions(u.boundary_conditions) && without_boundary_conditions(v.boundary_conditions) && return nothing

    grid = u.grid
    multiregion_u = Reference(u.data.regional_objects)
    multiregion_v = Reference(v.data.regional_objects)
    regions = Iterate(1:2)

    @apply_regionally fill_yin_yang_vertical_halo!(grid, u, args...)
    @apply_regionally fill_yin_yang_vertical_halo!(grid, v, args...)
    @apply_regionally fill_yin_yang_vector_halo!(grid, u, v, multiregion_u, multiregion_v, regions, Reference(tables))

    return nothing
end

function fill_yin_yang_vector_halo!(grid, u, v, multiregion_u, multiregion_v, region, tables)
    uᵈ = multiregion_u[3 - region]
    vᵈ = multiregion_v[3 - region]
    locu = instantiated_location(u)
    locv = instantiated_location(v)
    launch_yin_yang_interpolation!(grid, u, tables[1], _interpolate_yin_yang_vector_halo!, u, v, uᵈ, vᵈ, grid, locu, locv, tables...)
    return nothing
end

@inline maskable_receiver_index(i, ::Face, H) = max(i, 2 - H)
@inline maskable_receiver_index(i, loc, H) = i

@inline function immersed_vector_receiver(i, j, k, grid, ℓx, ℓy, ℓz)
    # The outermost lower halo face has no stored cell mask on its lower side.
    i′ = maskable_receiver_index(i, ℓx, grid.Hx)
    j′ = maskable_receiver_index(j, ℓy, grid.Hy)
    k′ = maskable_receiver_index(k, ℓz, grid.Hz)
    available = (i == i′) & (j == j′) & (k == k′)
    return available & immersed_peripheral_node(i′, j′, k′, grid, ℓx, ℓy, ℓz)
end

@inline function immersed_vector_receiver(i, j, k, grid, ℓx, ℓy, ::Nothing)
    immersed = true
    for k in 1:grid.Nz
        immersed &= immersed_vector_receiver(i, j, k, grid, ℓx, ℓy, Center())
    end
    return immersed
end

# `uu` and `uv` interpolate the donor `u` and `v` to the receivers of `u`, and `vu` and `vv` to those of `v`.
# All four tables list the same receivers in the same order.
@kernel function _interpolate_yin_yang_vector_halo!(u, v, uᵈ, vᵈ, grid, locu, locv, uu, uv, vu, vv)
    n, k = @index(Global, NTuple)

    @inbounds begin
        ũ = interpolate_donor(uᵈ, uu, n, k)
        ṽ = interpolate_donor(vᵈ, uv, n, k)
        i = uu.receivers.i[n]
        j = uu.receivers.j[n]
        value = ũ * uu.rotation.cosψ[n] - ṽ * uu.rotation.sinψ[n]
        u[i, j, k] = ifelse(immersed_vector_receiver(i, j, k, grid, locu...), zero(value), value)

        ũ = interpolate_donor(uᵈ, vu, n, k)
        ṽ = interpolate_donor(vᵈ, vv, n, k)
        i = vv.receivers.i[n]
        j = vv.receivers.j[n]
        value = ũ * vv.rotation.sinψ[n] + ṽ * vv.rotation.cosψ[n]
        v[i, j, k] = ifelse(immersed_vector_receiver(i, j, k, grid, locv...), zero(value), value)
    end
end
