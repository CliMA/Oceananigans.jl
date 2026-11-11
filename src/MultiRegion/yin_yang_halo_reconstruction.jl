using Oceananigans: instantiated_location
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!, fill_bottom_and_top_halo!
using Oceananigans.Utils: launch!
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
seam_halos(field) = getregion(field.indices, 1) isa Tuple{Colon, Colon, Any} &&
                    !(field isa Union{YinYangField{<:Nothing}, YinYangField{<:Any, <:Nothing}})

horizontal_location(field) = instantiated_location(field)[1:2]

fill_yin_yang_vertical_halos!(field, args::Vararg{Any, N}) where N =
    apply_regionally!(fill_bottom_and_top_halo!, field.data, field.boundary_conditions, field.indices,
                      instantiated_location(field), field.grid, args...)

"""
    fill_halo_regions!(field::YinYangField, args...; kwargs...)

Fill the vertical halos of `field` from its bottom and top boundary conditions, which receive `args`,
then its horizontal halos by interpolation from the other panel. `field` is interpolated as a scalar:
the components of a horizontal vector must be filled together, as `fill_halo_regions!((u, v))`.
Stencils that cross immersed cells use normalized positive weights from the wet donors.
"""
function BoundaryConditions.fill_halo_regions!(field::YinYangField, args::Vararg{Any, N}; kwargs...) where N
    without_boundary_conditions(field.boundary_conditions) && return nothing
    fill_yin_yang_vertical_halos!(field, args...)
    seam_halos(field) || return nothing

    grid = field.grid
    loc = instantiated_location(field)
    # Copies read the interiors of other regions, and interpolations read their copied halos: every device finishes
    # writing before each phase and reading after
    order_region_devices!()
    # Pushed: each piece is launched on its donor's device, so the donor's stencil reads are local; the receiver's
    # table is read in order (coalesced) and each receiver node is one remote write. Every halo node has one donor.
    for r in regions(grid)
        copy_subregion_halos!(field, r, region_halo(getregion(yin_yang_region_halos(grid), r), loc[1], loc[2]).copies)
    end
    order_region_devices!()
    for r in regions(grid)
        data = getregion(field.data, r)
        for (donor, (table,)) in region_halo(getregion(yin_yang_region_halos(grid), r), loc[1], loc[2]).interpolations
            activate_region_device!(donor)
            launch_yin_yang_interpolation!(getregion(grid, donor), axes(data, 3), table, _interpolate_yin_yang_scalar!,
                                           data, getregion(field.data, donor), getregion(grid, donor), loc, table)
        end
    end
    order_region_devices!()
    return nothing
end

# Interpolate the receivers of `table` at the vertical indices `ks` with `kernel!` on `grid`
function launch_yin_yang_interpolation!(grid, ks, table, kernel!, args::Vararg{Any, N}) where N
    length(table) == 0 && return nothing
    kernel_parameters = KernelParameters((length(table), length(ks)), (0, first(ks) - 1))
    launch!(architecture(grid), grid, kernel_parameters, kernel!, args...)
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

# The vertical halos take the masks of the bottom and top cells
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

"""
    fill_halo_regions!((u, v)::Tuple{YinYangField{Face, Center}, YinYangField{Center, Face}}, args...; kwargs...)

Fill the halos of the horizontal vector `(u, v)`: both components of the other panel are interpolated to each
halo node and rotated into this panel's basis. Interpolated components on immersed boundaries are zeroed;
vertically integrated components are zeroed where the entire column is immersed.
"""
function BoundaryConditions.fill_halo_regions!((u, v)::Tuple{YinYangXComponent, YinYangYComponent}, args::Vararg{Any, N};
                                               kwargs...) where N
    without_boundary_conditions(u.boundary_conditions) && return nothing
    fill_yin_yang_vertical_halos!(u, args...)
    fill_yin_yang_vertical_halos!(v, args...)
    seam_halos(u) || return nothing

    grid = u.grid
    ℓu, ℓv = instantiated_location(u), instantiated_location(v)
    order_region_devices!()
    # Pushed from each donor's device (see the scalar method)
    for r in regions(grid)
        halo = getregion(yin_yang_region_halos(grid), r).vector
        copy_subregion_halos!(u, r, halo.u_copies)
        copy_subregion_halos!(v, r, halo.v_copies)
    end
    order_region_devices!()
    # Each component is interpolated from both donor components, at its own location and at the other's
    for r in regions(grid)
        halo = getregion(yin_yang_region_halos(grid), r).vector
        uʳ, vʳ = getregion(u.data, r), getregion(v.data, r)
        panel = getregion(grid, r)
        for (donor, (uu, uv)) in halo.u
            activate_region_device!(donor)
            launch_yin_yang_interpolation!(getregion(grid, donor), axes(uʳ, 3), uu, _interpolate_yin_yang_vector_component!,
                                           uʳ, getregion(u.data, donor), getregion(v.data, donor), panel, ℓu, (uu, uv), Val(1))
        end
        for (donor, (vu, vv)) in halo.v
            activate_region_device!(donor)
            launch_yin_yang_interpolation!(getregion(grid, donor), axes(vʳ, 3), vv, _interpolate_yin_yang_vector_component!,
                                           vʳ, getregion(u.data, donor), getregion(v.data, donor), panel, ℓv, (vu, vv), Val(2))
        end
    end
    order_region_devices!()
    return nothing
end

@inline rotated_component(::Val{1}, u, v, cosψ, sinψ) = u * cosψ - v * sinψ
@inline rotated_component(::Val{2}, u, v, cosψ, sinψ) = u * sinψ + v * cosψ

# Component 1 or 2 of the donor vector `(uᵈ, vᵈ)` interpolated with `u_table` and `v_table`, which list the same
# receivers in the same order, and rotated into the receivers' basis. Receivers on immersed boundaries of `grid`
# are zeroed; `grid = nothing` leaves every receiver unmasked.
@kernel function _interpolate_yin_yang_vector_component!(c, uᵈ, vᵈ, grid, loc, tables, component)
    n, k = @index(Global, NTuple)
    @inbounds begin
        u_table, v_table = tables
        i = u_table.receivers.i[n]
        j = u_table.receivers.j[n]
        ũ = interpolate_donor(uᵈ, u_table, n, k)
        ṽ = interpolate_donor(vᵈ, v_table, n, k)
        value = rotated_component(component, ũ, ṽ, u_table.rotation.cosψ[n], u_table.rotation.sinψ[n])
        c[i, j, k] = ifelse(immersed_vector_receiver(i, j, k, grid, loc), zero(value), value)
    end
end

@inline immersed_vector_receiver(i, j, k, ::Nothing, loc) = false
@inline immersed_vector_receiver(i, j, k, grid, loc) = immersed_vector_receiver(i, j, k, grid, loc...)

@inline function immersed_vector_receiver(i, j, k, grid, ℓx, ℓy, ℓz)
    # The outermost lower halo face has no stored cell mask on its lower side
    i′ = maskable_receiver_index(i, ℓx, grid.Hx)
    j′ = maskable_receiver_index(j, ℓy, grid.Hy)
    available = (i == i′) & (j == j′)
    return available & immersed_peripheral_node(i′, j′, k, grid, ℓx, ℓy, ℓz)
end

@inline maskable_receiver_index(i, ::Face, H) = max(i, 2 - H)
@inline maskable_receiver_index(i, loc, H) = i

@inline immersed_vector_receiver(i, j, k, grid, ℓx, ℓy, ::Nothing) =
    column_immersed(immersed_vector_receiver, i, j, grid, ℓx, ℓy)
