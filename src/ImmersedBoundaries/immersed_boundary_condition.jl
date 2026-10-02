using Oceananigans.BoundaryConditions: BoundaryConditions,
                                       BoundaryCondition,
                                       DefaultBoundaryCondition,
                                       LeftBoundary,
                                       RightBoundary,
                                       regularize_boundary_condition,
                                       VBC, GBC, FBC, NFBC, Flux, ContinuousBoundaryFunction,
                                       needs_implicit_solver

import Oceananigans.BoundaryConditions: regularize_immersed_boundary_condition,
                                        bc_str,
                                        update_boundary_condition!

struct ImmersedBoundaryCondition{W, E, S, N, B, T}
    west :: W
    east :: E
    south :: S
    north :: N
    bottom :: B
    top :: T
end

const IBC = ImmersedBoundaryCondition

bc_str(::IBC) = "ImmersedBoundaryCondition"

Base.summary(ibc::IBC) =
    string(bc_str(ibc), " with ",
           "west=", bc_str(ibc.west), ", ",
           "east=", bc_str(ibc.east), ", ",
           "south=", bc_str(ibc.south), ", ",
           "north=", bc_str(ibc.north), ", ",
           "bottom=", bc_str(ibc.bottom), ", ",
           "top=", bc_str(ibc.top))

Base.show(io::IO, ibc::IBC) =
    print(io, "ImmersedBoundaryCondition:", "\n",
              "├── west: ", summary(ibc.west), "\n",
              "├── east: ", summary(ibc.east), "\n",
              "├── south: ", summary(ibc.south), "\n",
              "├── north: ", summary(ibc.north), "\n",
              "├── bottom: ", summary(ibc.bottom), "\n",
              "└── top: ", summary(ibc.top))

"""
    ImmersedBoundaryCondition(; west=nothing,
                               east=nothing,
                               south=nothing,
                               north=nothing,
                               bottom=nothing,
                               top=nothing)

Return an `ImmersedBoundaryCondition` with conditions on individual cell
`interfaces ∈ (west, east, south, north, bottom, top)` between the fluid
and the immersed boundary.
"""
function ImmersedBoundaryCondition(; west = nothing,
                                     east = nothing,
                                     south = nothing,
                                     north = nothing,
                                     bottom = nothing,
                                     top = nothing)

    return ImmersedBoundaryCondition(west, east, south, north, bottom, top)
end

BoundaryConditions.needs_implicit_solver(ibc::ImmersedBoundaryCondition) = needs_implicit_solver(ibc.bottom) | needs_implicit_solver(ibc.top)

#####
##### Boundary condition "regularization"
#####

const ZFBC = BoundaryCondition{<:Flux, Nothing}
regularize_immersed_boundary_condition(ibc::ZFBC, ibg::IBG, args...) = ibc # keep it

regularize_immersed_boundary_condition(default::DefaultBoundaryCondition, ibg::IBG, loc, field_name, args...) =
    regularize_immersed_boundary_condition(default.boundary_condition, ibg, loc, field_name, args...)

# Convert certain non-immersed boundary conditions to immersed boundary conditions
function regularize_immersed_boundary_condition(ibc::Union{VBC, GBC, FBC}, ibg::IBG, loc, field_name, args...)
    _ibc = ImmersedBoundaryCondition(Tuple(ibc for i=1:6)...)
    regularize_immersed_boundary_condition(_ibc, ibg, loc, field_name, args...)
end

"""
$(TYPEDSIGNATURES)
"""
function regularize_immersed_boundary_condition(bc::IBC, grid, loc, field_name, prognostic_field_names)

    west   = regularize_immersed_side(closed_velocity_normal_flow(bc.west, loc[1], field_name), grid, loc, 1, LeftBoundary,  prognostic_field_names)
    east   = regularize_immersed_side(closed_velocity_normal_flow(bc.east, loc[1], field_name), grid, loc, 1, RightBoundary, prognostic_field_names)
    south  = regularize_immersed_side(closed_velocity_normal_flow(bc.south, loc[2], field_name), grid, loc, 2, LeftBoundary,  prognostic_field_names)
    north  = regularize_immersed_side(closed_velocity_normal_flow(bc.north, loc[2], field_name), grid, loc, 2, RightBoundary, prognostic_field_names)
    bottom = regularize_immersed_side(closed_velocity_normal_flow(bc.bottom, loc[3], field_name), grid, loc, 3, LeftBoundary,  prognostic_field_names)
    top    = regularize_immersed_side(closed_velocity_normal_flow(bc.top, loc[3], field_name), grid, loc, 3, RightBoundary, prognostic_field_names)

    return ImmersedBoundaryCondition(; west, east, south, north, bottom, top)
end

closed_velocity_normal_flow(bc, loc, field_name) = bc

function closed_velocity_normal_flow(bc::NFBC, loc::Face, field_name)
    if field_name in (:u, :v, :w, :uh, :vh) && !nontrivial_normal_flow(bc)
        return nothing
    end
    return bc
end

regularize_immersed_side(::Nothing, grid, loc, dim, Side, names) = nothing

function regularize_immersed_side(bc, grid, loc, dim, Side, names)
    if loc[dim] isa Center
        return regularize_boundary_condition(bc, grid, loc, dim, Side, names)
    end

    bc isa NFBC || return nothing

    if !isnothing(bc.classification.scheme)
        throw(ArgumentError("Face-located immersed conditions must prescribe NormalFlow without a matching scheme."))
    end

    if isnothing(names) && bc.condition isa ContinuousBoundaryFunction && !isempty(bc.condition.field_dependencies)
        return bc
    end

    return regularize_boundary_condition(bc, grid, loc, dim, Side, names)
end

nontrivial_normal_flow(::Nothing) = false
nontrivial_normal_flow(bc) = false
nontrivial_normal_flow(bc::NFBC) = !isnothing(bc.classification.scheme) || !(isnothing(bc.condition) || (bc.condition isa Number && iszero(bc.condition)))
nontrivial_normal_flow(bc::DefaultBoundaryCondition) = nontrivial_normal_flow(bc.boundary_condition)
nontrivial_normal_flow(bc::IBC) = any(nontrivial_normal_flow, (bc.west, bc.east, bc.south, bc.north, bc.bottom, bc.top))

function reject_immersed_normal_flow_velocity_boundary_conditions(boundary_conditions, velocity_names=(:u, :v, :w))
    for name in velocity_names
        if haskey(boundary_conditions, name) && nontrivial_normal_flow(boundary_conditions[name].immersed)
            throw(ArgumentError("Prescribed immersed NormalFlow on prognostic velocity $name is not supported."))
        end
    end
    return nothing
end

Adapt.adapt_structure(to, bc::ImmersedBoundaryCondition) = ImmersedBoundaryCondition(Adapt.adapt(to, bc.west),
                                                                                     Adapt.adapt(to, bc.east),
                                                                                     Adapt.adapt(to, bc.south),
                                                                                     Adapt.adapt(to, bc.north),
                                                                                     Adapt.adapt(to, bc.bottom),
                                                                                     Adapt.adapt(to, bc.top))

update_boundary_condition!(bc::ImmersedBoundaryCondition, args...) = nothing

#####
##### Alternative implementation for immersed flux divergence
#####

#=
# Another idea...
# loc is the field location
# These are evaluated on both sides of a cell (eg left and right)
function immersed_flux_x(i, j, k, ibg, bc, loc, c, closure, K, id, args...)
    qᵂ = west_ib_flux(i, j, k, ibg, bc.west, loc, c, closure, K, id, args...)
    qᴱ = east_ib_flux(i, j, k, ibg, bc.east, loc, c, closure, K, id, args...)

    LX, LY, LZ = loc
    west_boundary = immersed_peripheral_node(flip(LX), LY, LZ, i, j, k, ibg) & !inactive_node(LX, LY, LZ, i, j, k, ibg)
    east_boundary = immersed_peripheral_node(flip(LX), LY, LZ, i, j, k, ibg) & !inactive_node(LX, LY, LZ, i-1, j, k, ibg)

    return ifelse(west_boundary, qᵂ, zero(ibg)) + ifelse(east_boundary, qᴱ, zero(ibg))
end

function immersed_flux_y(i, j, k, ibg, bc, loc, c, closure, K, id, args...)
    qˢ = south_ib_flux(i, j, k, ibg, bc.south, loc, c, closure, K, id, args...)
    qᴺ = north_ib_flux(i, j, k, ibg, bc.north, loc, c, closure, K, id, args...)

    LX, LY, LZ = loc
    south_boundary = immersed_peripheral_node(LX, flip(LY), LZ, i, j, k, ibg) & !inactive_node(LX, LY, LZ, i, j, k, ibg)
    north_boundary = immersed_peripheral_node(LX, flip(LY), LZ, i, j, k, ibg) & !inactive_node(LX, LY, LZ, i, j-1, k, ibg)

    return ifelse(south_boundary, qˢ, zero(ibg)) + ifelse(north_boundary, qᴺ, zero(ibg))
end

function immersed_flux_z(i, j, k, ibg, bc, loc, c, closure, K, id, args...)
    qᴮ = bottom_ib_flux(i, j, k, ibg, bc.bottom, loc, c, closure, K, id, args...)
    qᵀ =    top_ib_flux(i, j, k, ibg, bc.top,    loc, c, closure, K, id, args...)

    LX, LY, LZ = loc
    bottom_boundary = immersed_peripheral_node(LX, LY, flip(LZ), i, j, k, ibg) & !inactive_node(LX, LY, LZ, i, j, k, ibg)
    top_boundary    = immersed_peripheral_node(LX, LY, flip(LZ), i, j, k, ibg) & !inactive_node(LX, LY, LZ, i, j, k-1, ibg)

    return ifelse(bottom_boundary, qᴮ, zero(ibg)) + ifelse(top_boundary, qᵀ, zero(ibg))
end

@inline immersed_∂ⱼ_τ₁ⱼ(i, j, k, ibg::GFIBG, U, u_bc::IBC, closure, K, args...) =
    return 1/Vᶠᶜᶜ(i, j, k, ibg) * (δxᶠᵃᵃ(i, j, k, ibg, Ax_qᶜᶜᶜ, immersed_flux_x, u_bc, (f, c, c), U.u, closure, K, nothing, args...) +
                                   δyᵃᶜᵃ(i, j, k, ibg, Ax_qᶜᶜᶜ, immersed_flux_y, u_bc, (f, c, c), U.u, closure, K, nothing, args...) +
                                   δzᵃᵃᶜ(i, j, k, ibg, Az_qᶠᶜᶠ, immersed_flux_z, u_bc, (f, c, c), U.u, closure, K, nothing, args...))


@inline immersed_∂ⱼ_τ₂ⱼ(i, j, k, ibg::GFIBG, U, v_bc::IBC, closure, K, args...) =
    return 1/Vᶜᶠᶜ(i, j, k, ibg) * (δxᶜᵃᵃ(i, j, k, ibg, Ax_qᶠᶠᶜ, immersed_flux_x, v_bc, (c, c, c), U.v, closure, K, nothing, args...) +
                                   δyᵃᶠᵃ(i, j, k, ibg, Ax_qᶜᶜᶜ, immersed_flux_y, v_bc, (c, f, c), U.v, closure, K, nothing, args...) +
                                   δzᵃᵃᶜ(i, j, k, ibg, Az_qᶜᶠᶠ, immersed_flux_z, v_bc, (c, f, c), U.v, closure, K, nothing, args...))

    immersed_flux_divergence(i, j, k, ibg, v_bc, (c, f, c), U.v, closure, K, nothing, args...)

@inline immersed_∂ⱼ_τ₃ⱼ(i, j, k, ibg::GFIBG, U, w_bc::IBC, closure, K, args...) =
    return 1/Vᶜᶜᶠ(i, j, k, ibg) * (δxᶜᵃᵃ(i, j, k, ibg, Ax_qᶠᶜᶠ, immersed_flux_x, w_bc, (c, c, f), U.w, closure, K, nothing, args...) +
                                   δyᵃᶜᵃ(i, j, k, ibg, Ax_qᶜᶠᶠ, immersed_flux_y, w_bc, (c, c, f), U.w, closure, K, nothing, args...) +
                                   δzᵃᵃᶠ(i, j, k, ibg, Az_qᶜᶜᶜ, immersed_flux_z, w_bc, (c, c, f), U.w, closure, K, nothing, args...))

@inline immersed_∇_dot_qᶜ(i, j, k, ibg::GFIBG, C, c_bc::IBC, closure, K, id, args...) =
    return 1/Vᶜᶜᶜ(i, j, k, ibg) * (δxᶜᵃᵃ(i, j, k, ibg, Ax_qᶠᶜᶜ, immersed_flux_x, c_bc, (c, c, c), U.u, closure, K, id, args...) +
                                   δyᵃᶜᵃ(i, j, k, ibg, Ax_qᶜᶠᶜ, immersed_flux_y, c_bc, (c, c, c), U.u, closure, K, id, args...) +
                                   δzᵃᵃᶜ(i, j, k, ibg, Az_qᶜᶜᶠ, immersed_flux_z, c_bc, (c, c, c), U.u, closure, K, id, args...))

=#
