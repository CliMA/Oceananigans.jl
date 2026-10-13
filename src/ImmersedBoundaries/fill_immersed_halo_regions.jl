using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BoundaryConditions: NFBC, VBC, GBC, FieldBoundaryConditions, getbc
using Oceananigans.Grids: inactive_cell, interior_indices
using Oceananigans.Operators: Δx, Δy, Δz, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ

import Oceananigans.Fields: fill_immersed_halo_regions!, immersed_boundary_condition

#####
##### Open immersed boundaries
#####
##### A `NormalFlow` condition on a face-normal velocity opens the immersed faces on that side.
##### The velocity is set on the open faces, and the immersed nodes behind them are filled
##### like halo nodes from the adjacent fluid node, following each field's own immersed condition.
#####

const IBGField = Field{<:CenterOrFace, <:CenterOrFace, <:CenterOrFace, <:Any, <:IBG}

function fill_immersed_halo_regions!(field::IBGField, args...; fill_normal_flow_bcs=true, kwargs...)
    grid = field.grid
    arch = architecture(grid)
    loc = instantiated_location(field)
    ibc = immersed_boundary_condition(field.boundary_conditions)
    params = KernelParameters(interior_indices(field)...)

    if fill_normal_flow_bcs && has_normal_flow(ibc)
        launch!(arch, grid, params, _fill_immersed_normal_flow!, field, loc, grid, ibc, args)
    end

    open = open_immersed_sides(args...)
    if any(open)
        launch!(arch, grid, params, _fill_immersed_halos!, field, loc, grid, ibc, open, args)
    end

    return nothing
end

@inline immersed_boundary_condition(bcs::FieldBoundaryConditions) = bcs.immersed
@inline immersed_boundary_condition(other) = nothing

@inline is_normal_flow(bc) = false
@inline is_normal_flow(::NFBC) = true

@inline has_normal_flow(ibc) = false
@inline has_normal_flow(ibc::IBC) = is_normal_flow(ibc.west) | is_normal_flow(ibc.east) |
                                    is_normal_flow(ibc.south) | is_normal_flow(ibc.north) |
                                    is_normal_flow(ibc.bottom) | is_normal_flow(ibc.top)

@inline immersed_side(ibc::IBC, side) = getproperty(ibc, side)
@inline immersed_side(ibc, side) = nothing

# The sides of the immersed boundary that are open, read from the face-normal velocities in `model_fields`
@inline open_immersed_sides(args...) = (west=false, east=false, south=false, north=false, bottom=false, top=false)

@inline function open_immersed_sides(clock, model_fields::NamedTuple, args...)
    u_ibc = immersed_boundary_condition(get(model_fields, :u, nothing))
    v_ibc = immersed_boundary_condition(get(model_fields, :v, nothing))
    w_ibc = immersed_boundary_condition(get(model_fields, :w, nothing))

    return (west   = is_normal_flow(immersed_side(u_ibc, :west)),
            east   = is_normal_flow(immersed_side(u_ibc, :east)),
            south  = is_normal_flow(immersed_side(v_ibc, :south)),
            north  = is_normal_flow(immersed_side(v_ibc, :north)),
            bottom = is_normal_flow(immersed_side(w_ibc, :bottom)),
            top    = is_normal_flow(immersed_side(w_ibc, :top)))
end

#####
##### Normal flow through open immersed faces
#####

@kernel function _fill_immersed_normal_flow!(ψ, loc, grid, ibc, args)
    i, j, k = @index(Global, NTuple)
    @inbounds ψ[i, j, k] = immersed_normal_flow(i, j, k, grid, loc, ibc, ψ[i, j, k], args...)
end

@inline normal_flow(bc::NFBC, i, j, k, grid, args...) = getbc(bc, i, j, k, grid, args...)
@inline normal_flow(bc, i, j, k, grid, args...) = zero(grid)

# Faces of the immersed boundary, named after the side of the adjacent fluid cell they bound
@inline west_immersed_face(i, j, k, grid)   = immersed_peripheral_node(i, j, k, grid, f, c, c) & !inactive_cell(i, j, k, grid) & inactive_cell(i-1, j, k, grid)
@inline east_immersed_face(i, j, k, grid)   = immersed_peripheral_node(i, j, k, grid, f, c, c) & !inactive_cell(i-1, j, k, grid) & inactive_cell(i, j, k, grid)
@inline south_immersed_face(i, j, k, grid)  = immersed_peripheral_node(i, j, k, grid, c, f, c) & !inactive_cell(i, j, k, grid) & inactive_cell(i, j-1, k, grid)
@inline north_immersed_face(i, j, k, grid)  = immersed_peripheral_node(i, j, k, grid, c, f, c) & !inactive_cell(i, j-1, k, grid) & inactive_cell(i, j, k, grid)
@inline bottom_immersed_face(i, j, k, grid) = immersed_peripheral_node(i, j, k, grid, c, c, f) & !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)
@inline top_immersed_face(i, j, k, grid)    = immersed_peripheral_node(i, j, k, grid, c, c, f) & !inactive_cell(i, j, k-1, grid) & inactive_cell(i, j, k, grid)

# Conditions are evaluated at the fluid cell adjacent to the face, like every immersed condition
@inline function immersed_normal_flow(i, j, k, grid, ::Tuple{Face, Center, Center}, ibc, ψ, args...)
    ψ = ifelse(west_immersed_face(i, j, k, grid), normal_flow(ibc.west, i, j, k, grid, args...), ψ)
    ψ = ifelse(east_immersed_face(i, j, k, grid), normal_flow(ibc.east, i-1, j, k, grid, args...), ψ)
    return ψ
end

@inline function immersed_normal_flow(i, j, k, grid, ::Tuple{Center, Face, Center}, ibc, ψ, args...)
    ψ = ifelse(south_immersed_face(i, j, k, grid), normal_flow(ibc.south, i, j, k, grid, args...), ψ)
    ψ = ifelse(north_immersed_face(i, j, k, grid), normal_flow(ibc.north, i, j-1, k, grid, args...), ψ)
    return ψ
end

@inline function immersed_normal_flow(i, j, k, grid, ::Tuple{Center, Center, Face}, ibc, ψ, args...)
    ψ = ifelse(bottom_immersed_face(i, j, k, grid), normal_flow(ibc.bottom, i, j, k, grid, args...), ψ)
    ψ = ifelse(top_immersed_face(i, j, k, grid), normal_flow(ibc.top, i, j, k-1, grid, args...), ψ)
    return ψ
end

@inline immersed_normal_flow(i, j, k, grid, loc, ibc, ψ, args...) = ψ

#####
##### Net inflow through the open immersed faces
#####

@inline function immersed_inflow(i, j, k, grid, u, v, w, open)
    Qᵂ = Ax_qᶠᶜᶜ(i,   j,   k,   grid, u) * (open.west   & west_immersed_face(i,   j,   k,   grid))
    Qᴱ = Ax_qᶠᶜᶜ(i+1, j,   k,   grid, u) * (open.east   & east_immersed_face(i+1, j,   k,   grid))
    Qˢ = Ay_qᶜᶠᶜ(i,   j,   k,   grid, v) * (open.south  & south_immersed_face(i,  j,   k,   grid))
    Qᴺ = Ay_qᶜᶠᶜ(i,   j+1, k,   grid, v) * (open.north  & north_immersed_face(i,  j+1, k,   grid))
    Qᴮ = Az_qᶜᶜᶠ(i,   j,   k,   grid, w) * (open.bottom & bottom_immersed_face(i, j,   k,   grid))
    Qᵀ = Az_qᶜᶜᶠ(i,   j,   k+1, grid, w) * (open.top    & top_immersed_face(i,    j,   k+1, grid))
    return Qᵂ - Qᴱ + Qˢ - Qᴺ + Qᴮ - Qᵀ
end

"""
$(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` for the volume inflow into each cell of `velocities` through
its open immersed faces, or `nothing` when the immersed boundary has no open faces.
"""
function immersed_inflow_operation(velocities)
    u, v, w = velocities
    open = open_immersed_sides(nothing, velocities)
    any(open) || return nothing
    return KernelFunctionOperation{Center, Center, Center}(immersed_inflow, u.grid, u, v, w, open)
end

#####
##### Immersed halo nodes behind open faces
#####

@kernel function _fill_immersed_halos!(ψ, loc, grid, ibc, open, args)
    i, j, k = @index(Global, NTuple)
    @inbounds ψ[i, j, k] = immersed_halo_value(i, j, k, grid, loc, ibc, open, ψ, args...)
end

# The halo value extrapolated from the fluid value `ψ` over the signed distance `Δ` to the halo node
@inline halo_value(bc::VBC, ψ, Δ, i, j, k, grid, args...) = 2 * getbc(bc, i, j, k, grid, args...) - ψ
@inline halo_value(bc::GBC, ψ, Δ, i, j, k, grid, args...) = ψ + Δ * getbc(bc, i, j, k, grid, args...)
@inline halo_value(bc, ψ, Δ, i, j, k, grid, args...) = ψ

@inline function immersed_halo_value(i, j, k, grid, loc, ibc, open, ψ, args...)
    ℓx, ℓy, ℓz = loc
    halo = immersed_peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz)
    @inbounds ψᵢ = ψ[i, j, k]
    ψᵢ = x_halo_value(i, j, k, grid, ℓx, ℓy, ℓz, ibc, open, ψ, ψᵢ, halo, args...)
    ψᵢ = y_halo_value(i, j, k, grid, ℓx, ℓy, ℓz, ibc, open, ψ, ψᵢ, halo, args...)
    ψᵢ = z_halo_value(i, j, k, grid, ℓx, ℓy, ℓz, ibc, open, ψ, ψᵢ, halo, args...)
    return ψᵢ
end

# A field is not extrapolated across the faces it lives on
@inline x_halo_value(i, j, k, grid, ::Face, ℓy, ℓz, ibc, open, ψ, ψᵢ, args...) = ψᵢ
@inline y_halo_value(i, j, k, grid, ℓx, ::Face, ℓz, ibc, open, ψ, ψᵢ, args...) = ψᵢ
@inline z_halo_value(i, j, k, grid, ℓx, ℓy, ::Face, ibc, open, ψ, ψᵢ, args...) = ψᵢ

@inline function x_halo_value(i, j, k, grid, ℓx::Center, ℓy, ℓz, ibc, open, ψ, ψᵢ, halo, args...)
    fluid_east = halo & !peripheral_node(i+1, j, k, grid, ℓx, ℓy, ℓz)
    fluid_west = halo & !peripheral_node(i-1, j, k, grid, ℓx, ℓy, ℓz)
    @inbounds ψᴱ = ψ[i+1, j, k]
    @inbounds ψᵂ = ψ[i-1, j, k]
    Δᴱ = Δx(i+1, j, k, grid, f, ℓy, ℓz)
    Δᵂ = Δx(i,   j, k, grid, f, ℓy, ℓz)
    ψᵢ = ifelse(open.west & fluid_east, halo_value(immersed_side(ibc, :west), ψᴱ, -Δᴱ, i+1, j, k, grid, args...), ψᵢ)
    ψᵢ = ifelse(open.east & fluid_west, halo_value(immersed_side(ibc, :east), ψᵂ, +Δᵂ, i-1, j, k, grid, args...), ψᵢ)
    return ψᵢ
end

@inline function y_halo_value(i, j, k, grid, ℓx, ℓy::Center, ℓz, ibc, open, ψ, ψᵢ, halo, args...)
    fluid_north = halo & !peripheral_node(i, j+1, k, grid, ℓx, ℓy, ℓz)
    fluid_south = halo & !peripheral_node(i, j-1, k, grid, ℓx, ℓy, ℓz)
    @inbounds ψᴺ = ψ[i, j+1, k]
    @inbounds ψˢ = ψ[i, j-1, k]
    Δᴺ = Δy(i, j+1, k, grid, ℓx, f, ℓz)
    Δˢ = Δy(i, j,   k, grid, ℓx, f, ℓz)
    ψᵢ = ifelse(open.south & fluid_north, halo_value(immersed_side(ibc, :south), ψᴺ, -Δᴺ, i, j+1, k, grid, args...), ψᵢ)
    ψᵢ = ifelse(open.north & fluid_south, halo_value(immersed_side(ibc, :north), ψˢ, +Δˢ, i, j-1, k, grid, args...), ψᵢ)
    return ψᵢ
end

@inline function z_halo_value(i, j, k, grid, ℓx, ℓy, ℓz::Center, ibc, open, ψ, ψᵢ, halo, args...)
    fluid_above = halo & !peripheral_node(i, j, k+1, grid, ℓx, ℓy, ℓz)
    fluid_below = halo & !peripheral_node(i, j, k-1, grid, ℓx, ℓy, ℓz)
    @inbounds ψᵀ = ψ[i, j, k+1]
    @inbounds ψᴮ = ψ[i, j, k-1]
    Δᵀ = Δz(i, j, k+1, grid, ℓx, ℓy, f)
    Δᴮ = Δz(i, j, k,   grid, ℓx, ℓy, f)
    ψᵢ = ifelse(open.bottom & fluid_above, halo_value(immersed_side(ibc, :bottom), ψᵀ, -Δᵀ, i, j, k+1, grid, args...), ψᵢ)
    ψᵢ = ifelse(open.top & fluid_below,    halo_value(immersed_side(ibc, :top),    ψᴮ, +Δᴮ, i, j, k-1, grid, args...), ψᵢ)
    return ψᵢ
end
