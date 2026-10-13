using Oceananigans: instantiated_location
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!, fill_halo_event!, get_boundary_kernels
using Oceananigans.Utils: launch!, KernelParameters, Reference, Iterate, @apply_regionally
using KernelAbstractions.Extras.LoopInfo: @unroll

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
Fields windowed or reduced in `x` or `y` get only their vertical halos.
"""
function BoundaryConditions.fill_halo_regions!(field::YinYangField, args::Vararg{Any, N}; kwargs...) where N
    without_boundary_conditions(field.boundary_conditions) && return nothing

    grid = field.grid
    ℓx, ℓy, _ = instantiated_location(field)
    table = Reference(interpolation_table(yin_yang_interpolation(grid), ℓx, ℓy))
    multiregion_field = Reference(field.data.regional_objects)
    regions = Iterate(1:2)

    @apply_regionally fill_yin_yang_vertical_halo!(grid, field, args...)
    @apply_regionally fill_yin_yang_scalar_halo!(grid, field, multiregion_field, regions, table)

    return nothing
end

const HorizontallyReducedYinYangField = Union{YinYangField{<:Nothing}, YinYangField{<:Any, <:Nothing}}

function BoundaryConditions.fill_halo_regions!(field::HorizontallyReducedYinYangField, args::Vararg{Any, N}; kwargs...) where N
    without_boundary_conditions(field.boundary_conditions) && return nothing
    @apply_regionally fill_yin_yang_vertical_halo!(field.grid, field, args...)
    return nothing
end

function fill_yin_yang_vertical_halo!(grid, field, args::Vararg{Any, N}) where N
    loc = instantiated_location(field)
    kernels!, bcs = get_boundary_kernels(field.boundary_conditions, field.data, grid, loc, field.indices)
    fill_halo_event!(field.data, kernels!.bottom_and_top, bcs.bottom_and_top, loc, grid, args...)
    return nothing
end

function fill_yin_yang_scalar_halo!(grid, field, multiregion_field, region, table)
    donor = multiregion_field[3 - region]
    launch_yin_yang_interpolation!(grid, field, table, _interpolate_yin_yang_scalar_halo!, field, donor, table)
    return nothing
end

function launch_yin_yang_interpolation!(grid, field, table, kernel!, args::Vararg{Any, N}) where N
    horizontally_windowed(field.indices) && return nothing

    ks = axes(field.data, 3)
    kernel_parameters = KernelParameters((length(table), length(ks)), (0, first(ks) - 1))
    launch!(architecture(grid), grid, kernel_parameters, kernel!, args...)

    return nothing
end

horizontally_windowed(::Tuple{Colon, Colon, Any}) = false
horizontally_windowed(indices) = true

@kernel function _interpolate_yin_yang_scalar_halo!(c, donor, table)
    n, k = @index(Global, NTuple)

    @inbounds begin
        i = table.receivers.i[n]
        j = table.receivers.j[n]
        c[i, j, k] = interpolate_donor(donor, table, n, k)
    end
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
    launch_yin_yang_interpolation!(grid, u, tables[1], _interpolate_yin_yang_vector_halo!, u, v, uᵈ, vᵈ, tables...)
    return nothing
end

# `uu` and `uv` interpolate the donor `u` and `v` to the receivers of `u`, and `vu` and `vv` to those of `v`.
# All four tables list the same receivers in the same order.
@kernel function _interpolate_yin_yang_vector_halo!(u, v, uᵈ, vᵈ, uu, uv, vu, vv)
    n, k = @index(Global, NTuple)

    @inbounds begin
        ũ = interpolate_donor(uᵈ, uu, n, k)
        ṽ = interpolate_donor(vᵈ, uv, n, k)
        i = uu.receivers.i[n]
        j = uu.receivers.j[n]
        u[i, j, k] = ũ * uu.rotation.cosψ[n] - ṽ * uu.rotation.sinψ[n]

        ũ = interpolate_donor(uᵈ, vu, n, k)
        ṽ = interpolate_donor(vᵈ, vv, n, k)
        i = vv.receivers.i[n]
        j = vv.receivers.j[n]
        v[i, j, k] = ũ * vv.rotation.sinψ[n] + ṽ * vv.rotation.cosψ[n]
    end
end
