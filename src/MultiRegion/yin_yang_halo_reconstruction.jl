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
##### Halo filling between the Yin and Yang panels. The donor stencils read only interior nodes, so the panels
##### can be filled in either order. The vertical halos are filled first, so that the interpolation also fills
##### those of the halo columns.
#####

without_boundary_conditions(bcs) = isnothing(bcs)
without_boundary_conditions(bcs::MultiRegionObject) = all(isnothing, bcs.regional_objects)

# Fields windowed or reduced in x or y have no seam halos
const HorizontallyReducedYinYangField = Union{YinYangField{<:Nothing}, YinYangField{<:Any, <:Nothing}}
seam_halos(field) = getregion(field.indices, 1) isa Tuple{Colon, Colon, Any} && !(field isa HorizontallyReducedYinYangField)

horizontal_location(field) = instantiated_location(field)[1:2]

fill_yin_yang_vertical_halo!(grid, field, args::Vararg{Any, N}) where N =
    fill_bottom_and_top_halo!(field.data, field.boundary_conditions, field.indices, instantiated_location(field), grid, args...)

"""
    fill_halo_regions!(field::YinYangField, args...; kwargs...)

Fill the vertical halos of `field` from its bottom and top boundary conditions, which receive `args`,
then its horizontal halos by interpolation from the other panel. `field` is interpolated as a scalar:
the components of a horizontal vector must be filled together, as `fill_halo_regions!((u, v))`.
Stencils that cross immersed cells use normalized positive weights from the wet donors.
"""
function BoundaryConditions.fill_halo_regions!(field::YinYangField, args::Vararg{Any, N}; kwargs...) where N
    without_boundary_conditions(field.boundary_conditions) && return nothing
    grid = field.grid
    @apply_regionally fill_yin_yang_vertical_halo!(grid, field, args...)
    seam_halos(field) || return nothing

    ℓx, ℓy, _ = instantiated_location(field)
    table = Reference(interpolation_table(yin_yang_interpolation(grid), ℓx, ℓy))
    data = Reference(field.data.regional_objects)
    grids = Reference((getregion(grid, 1), getregion(grid, 2)))
    @apply_regionally fill_yin_yang_scalar_halo!(grid, field, data, grids, Iterate(1:2), table)
    return nothing
end

fill_yin_yang_scalar_halo!(grid, field, data, grids, region, table) =
    launch_yin_yang_interpolation!(field, axes(field.data, 3), table, _interpolate_yin_yang_scalar!,
                                   field, data[3 - region], grids[3 - region], instantiated_location(field), table)

# Interpolate the receivers of `table` at the vertical indices `ks` of `field`
function launch_yin_yang_interpolation!(field, ks, table, kernel!, args::Vararg{Any, N}) where N
    length(table) == 0 && return nothing
    kernel_parameters = KernelParameters((length(table), length(ks)), (0, first(ks) - 1))
    launch!(architecture(field), field.grid, kernel_parameters, kernel!, args...)
    return nothing
end

@kernel function _interpolate_yin_yang_scalar!(c, donor, donor_grid, loc, table)
    n, k = @index(Global, NTuple)
    @inbounds begin
        i = table.receivers.i[n]
        j = table.receivers.j[n]
        c[i, j, k] = interpolate_donor(donor, donor_grid, loc, table, n, k)
    end
end

@inline function interpolate_donor(donor, table::YinYangInterpolationTable{S}, n, k) where S
    @inbounds begin
        i₀, j₀ = table.donors.i[n], table.donors.j[n]
        wx, wy = table.weights.x, table.weights.y
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

# Every donor of a grid without immersed boundaries is wet
@inline interpolate_donor(donor, grid, loc, table, n, k) = interpolate_donor(donor, table, n, k)

@inline function interpolate_donor(donor, grid::ImmersedBoundaryGrid, loc, table::YinYangInterpolationTable{S}, n, k) where S
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

@inline scalar_mask_index(k, ::Center, grid) = clamp(k, 1, grid.Nz)
@inline scalar_mask_index(k, ::Face, grid) = clamp(k, 1, grid.Nz + 1)

@inline immersed_scalar_donor(i, j, k, grid, ℓx, ℓy, ℓz) =
    immersed_inactive_node(i, j, scalar_mask_index(k, ℓz, grid), grid, ℓx, ℓy, ℓz)

# Vertically integrated nodes are immersed where the entire column is
@inline function column_immersed(immersed_node, i, j, grid, ℓx, ℓy)
    immersed = true
    for k in 1:grid.Nz
        immersed &= immersed_node(i, j, k, grid, ℓx, ℓy, Center())
    end
    return immersed
end

@inline immersed_scalar_donor(i, j, k, grid, ℓx, ℓy, ::Nothing) = column_immersed(immersed_inactive_node, i, j, grid, ℓx, ℓy)

#####
##### Horizontal vectors
#####

const YinYangXComponent = YinYangField{<:Face, <:Center}
const YinYangYComponent = YinYangField{<:Center, <:Face}
const YinYangCornerComponent = YinYangField{<:Face, <:Face}

BoundaryConditions.fill_halo_regions!(fields::Tuple{YinYangField, Vararg{YinYangField}}, args::Vararg{Any, N}; kwargs...) where N =
    foreach(field -> fill_halo_regions!(field, args...; kwargs...), fields)

BoundaryConditions.fill_halo_regions!((u, v)::Tuple{YinYangXComponent, YinYangYComponent}, args::Vararg{Any, N}; kwargs...) where N =
    fill_halo_regions!(u, v, args...; kwargs...)

"""
    fill_halo_regions!(u::YinYangField{Face, Center}, v::YinYangField{Center, Face}, args...; signed = true, kwargs...)
    fill_halo_regions!(u::YinYangField{Face, Face}, v::YinYangField{Face, Face}, args...; signed = true, kwargs...)

Fill the halos of the horizontal vector `(u, v)`: both components of the other panel are interpolated to each
halo node and rotated into this panel's basis. Interpolated components on immersed boundaries are zeroed;
vertically integrated components are zeroed where the entire column is immersed.
With `signed = false`, `u` and `v` are filled as scalars.
"""
BoundaryConditions.fill_halo_regions!(u::YinYangXComponent, v::YinYangYComponent, args::Vararg{Any, N}; kwargs...) where N =
    fill_yin_yang_vector_halos!(u, v, args...; kwargs...)

BoundaryConditions.fill_halo_regions!(u::YinYangCornerComponent, v::YinYangCornerComponent, args::Vararg{Any, N}; kwargs...) where N =
    fill_yin_yang_vector_halos!(u, v, args...; kwargs...)

function fill_yin_yang_vector_halos!(u, v, args::Vararg{Any, N}; signed = true, kwargs...) where N
    if !signed
        fill_halo_regions!(u, args...; kwargs...)
        fill_halo_regions!(v, args...; kwargs...)
        return nothing
    end

    without_boundary_conditions(u.boundary_conditions) && without_boundary_conditions(v.boundary_conditions) && return nothing

    grid = u.grid
    @apply_regionally fill_yin_yang_vertical_halo!(grid, u, args...)
    @apply_regionally fill_yin_yang_vertical_halo!(grid, v, args...)
    seam_halos(u) || return nothing

    # Each component is interpolated from both donor components, at its own location and at the other's
    ℓu, ℓv = horizontal_location(u), horizontal_location(v)
    table(ℓ, ℓ′) = interpolation_table(yin_yang_interpolation(grid), ℓ..., ℓ′...)
    tables = Reference((table(ℓu, ℓu), table(ℓu, ℓv), table(ℓv, ℓu), table(ℓv, ℓv)))
    data = Reference((u.data.regional_objects, v.data.regional_objects))
    @apply_regionally fill_yin_yang_vector_halo!(grid, u, v, data, Iterate(1:2), tables)
    return nothing
end

function fill_yin_yang_vector_halo!(grid, u, v, data, region, (uu, uv, vu, vv))
    uᵈ = data[1][3 - region]
    vᵈ = data[2][3 - region]
    launch_yin_yang_interpolation!(u, axes(u.data, 3), uu, _interpolate_yin_yang_vector_component!,
                                   u, uᵈ, vᵈ, grid, instantiated_location(u), uu, uv, Val(1))
    launch_yin_yang_interpolation!(v, axes(v.data, 3), vv, _interpolate_yin_yang_vector_component!,
                                   v, uᵈ, vᵈ, grid, instantiated_location(v), vu, vv, Val(2))
    return nothing
end

@inline rotated_component(::Val{1}, u, v, cosψ, sinψ) = u * cosψ - v * sinψ
@inline rotated_component(::Val{2}, u, v, cosψ, sinψ) = u * sinψ + v * cosψ

# Component 1 or 2 of the donor vector `(uᵈ, vᵈ)` interpolated with `u_table` and `v_table`, which list the same
# receivers in the same order, and rotated into the receivers' basis
@inline function interpolate_donor_vector(component, uᵈ, vᵈ, u_table, v_table, n, k)
    ũ = interpolate_donor(uᵈ, u_table, n, k)
    ṽ = interpolate_donor(vᵈ, v_table, n, k)
    return @inbounds rotated_component(component, ũ, ṽ, u_table.rotation.cosψ[n], u_table.rotation.sinψ[n])
end

# Receivers on immersed boundaries of `grid` are zeroed; `nothing` leaves every receiver unmasked
@kernel function _interpolate_yin_yang_vector_component!(c, uᵈ, vᵈ, grid, loc, u_table, v_table, component)
    n, k = @index(Global, NTuple)
    @inbounds begin
        i = u_table.receivers.i[n]
        j = u_table.receivers.j[n]
        value = interpolate_donor_vector(component, uᵈ, vᵈ, u_table, v_table, n, k)
        c[i, j, k] = ifelse(immersed_vector_receiver(i, j, k, grid, loc), zero(value), value)
    end
end

@inline maskable_receiver_index(i, ::Face, H) = max(i, 2 - H)
@inline maskable_receiver_index(i, loc, H) = i

@inline immersed_vector_receiver(i, j, k, ::Nothing, loc) = false
@inline immersed_vector_receiver(i, j, k, grid, loc) = immersed_vector_receiver(i, j, k, grid, loc...)

@inline function immersed_vector_receiver(i, j, k, grid, ℓx, ℓy, ℓz)
    # The outermost lower halo face has no stored cell mask on its lower side.
    i′ = maskable_receiver_index(i, ℓx, grid.Hx)
    j′ = maskable_receiver_index(j, ℓy, grid.Hy)
    k′ = maskable_receiver_index(k, ℓz, grid.Hz)
    available = (i == i′) & (j == j′) & (k == k′)
    return available & immersed_peripheral_node(i′, j′, k′, grid, ℓx, ℓy, ℓz)
end

@inline immersed_vector_receiver(i, j, k, grid, ℓx, ℓy, ::Nothing) =
    column_immersed(immersed_vector_receiver, i, j, grid, ℓx, ℓy)
