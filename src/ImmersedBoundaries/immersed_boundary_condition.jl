using Oceananigans.BoundaryConditions: Flux, Value, Gradient, flip, BoundaryCondition, ContinuousBoundaryFunction
using Oceananigans.BoundaryConditions: getbc, regularize_boundary_condition, LeftBoundary, RightBoundary
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, h_diffusivity, z_diffusivity
using Oceananigans.Operators: index_left, index_right, Δx, Δy, Δz, div

import Oceananigans.BoundaryConditions: regularize_immersed_boundary_condition, bc_str

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
    print(io, "ImmersedBoundaryCondition:", '\n',
              "├── west: ", summary(ibc.west), '\n',
              "├── east: ", summary(ibc.east), '\n',
              "├── south: ", summary(ibc.south), '\n',
              "├── north: ", summary(ibc.north), '\n',
              "├── bottom: ", summary(ibc.bottom), '\n',
              "└── top: ", summary(ibc.top))

"""
    ImmersedBoundaryCondition(; interfaces...)

Return an ImmersedBoundaryCondition with conditions on individual
cell `interfaces ∈ (west, east, south, north, bottom, top)`
between the fluid and immersed boundary.
"""
function ImmersedBoundaryCondition(; west = nothing,
                                     east = nothing,
                                     south = nothing,
                                     north = nothing,
                                     bottom = nothing,
                                     top = nothing)

    return ImmersedBoundaryCondition(west, east, south, north, bottom, top)
end

#####
##### Nothing and FluxBoundaryCondition.
#####
##### Very Important Note: For FluxBoundaryCondition,
##### we assume fluxes are directed along the "inward-facing normal".
##### For example, east_immersed_flux = - user_flux.
##### With this convention, positive fluxes _increase_ boundary-adjacent
##### cell values, and negative fluxes _decrease_ them.
#####

for side in (:west, :south, :bottom)
    get_flux = Symbol(side, :_ib_flux)
    @eval begin
        @inline $get_flux(i, j, k, ibg, ::Nothing, args...) = zero(eltype(ibg))
        @inline $get_flux(i, j, k, ibg, bc::FBC, loc, c, closure, K, id, args...) = + getbc(bc, i, j, k, ibg, args...)
    end
end

for side in (:east, :north, :top)
    get_flux = Symbol(side, :_ib_flux)
    @eval begin
        @inline $get_flux(i, j, k, ibg, ::Nothing, args...) = zero(eltype(ibg))
        @inline $get_flux(i, j, k, ibg, bc::FBC, loc, c, closure, K, id, args...) = - getbc(bc, i, j, k, ibg, args...)
    end
end

#####
##### Value boundary condition. It's harder!
#####

# Harder in some ways... ValueBoundaryCondition...
const VBC = BoundaryCondition{Value}
const GBC = BoundaryCondition{Gradient}
const VBCorGBC = Union{VBC, GBC}
const ASD = AbstractScalarDiffusivity

# "Gradient" utility for Value or Gradient boundary conditions
@inline right_gradient(i, j, k, ibg, κ, Δ, bc::GBC, c, clock, fields) = getbc(bc, i, j, k, ibg, clock, fields)
@inline left_gradient(i, j, k, ibg, κ, Δ, bc::GBC, c, clock, fields)  = getbc(bc, i, j, k, ibg, clock, fields)

@inline function right_gradient(i, j, k, ibg, κ, Δ, bc::VBC, c, clock, fields)
    cᵇ = getbc(bc, i, j, k, ibg, clock, fields)
    cⁱʲᵏ = @inbounds c[i, j, k]
    return 2 * (cᵇ - cⁱʲᵏ) / Δ
end

@inline function left_gradient(i, j, k, ibg, κ, Δ, bc::VBC, c, clock, fields)
    cᵇ = getbc(bc, i, j, k, ibg, clock, fields)
    cⁱʲᵏ = @inbounds c[i, j, k]
    return 2 * (cⁱʲᵏ - cᵇ) / Δ
end

# Metric and index gymnastics for the 6 facets of the cube

@inline function west_ib_flux(i, j, k, ibg, bc::VBCorGBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δx(index_left(i, LX), j, k, ibg, LX, LY, LZ)
    κ = h_diffusivity(i, j, k, ibg, flip(LX), LY, LZ, closure, K, id, clock)
    ∇c = left_gradient(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
    return - κ * ∇c
end

@inline function east_ib_flux(i, j, k, ibg, bc::VBCorGBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δx(index_right(i, LX), j, k, ibg, LX, LY, LZ)
    κ = h_diffusivity(i, j, k, ibg, flip(LX), LY, LZ, closure, K, id, clock)
    ∇c = right_gradient(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
    return - κ * ∇c
end

@inline function south_ib_flux(i, j, k, ibg, bc::VBCorGBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δy(i, index_left(j, LY), k, ibg, LX, LY, LZ)
    κ = h_diffusivity(i, j, k, ibg, LX, flip(LY), LZ, closure, K, id, clock)
    ∇c = left_gradient(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
    return - κ * ∇c
end

@inline function north_ib_flux(i, j, k, ibg, bc::VBCorGBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δy(i, index_right(j, LY), k, ibg, LX, LY, LZ)
    κ = h_diffusivity(i, j, k, ibg, LX, flip(LY), LZ, closure, K, id, clock)
    ∇c = right_gradient(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
    return - κ * ∇c
end

@inline function bottom_ib_flux(i, j, k, ibg, bc::VBCorGBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δz(i, j, index_left(k, LZ), ibg, LX, LY, LZ)
    κ = z_diffusivity(i, j, k, ibg, LX, LY, flip(LZ), closure, K, id, clock)
    ∇c = left_gradient(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
    return - κ * ∇c
end

@inline function top_ib_flux(i, j, k, ibg, bc::VBCorGBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δz(i, j, index_right(k, LZ), ibg, LX, LY, LZ)
    κ = z_diffusivity(i, j, k, ibg, LX, LY, flip(LZ), closure, K, id, clock)
    ∇c = right_gradient(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
    return - κ * ∇c
end

#####
##### Immersed flux divergence
#####

@inline function immersed_flux_divergence(i, j, k, ibg::GFIBG, bc, loc, c, closure, K, id, clock, fields)
    # Fetch fluxes across immersed boundary
    q̃ᵂ =   west_ib_flux(i, j, k, ibg, bc.west,   loc, c, closure, K, id, clock, fields)
    q̃ᴱ =   east_ib_flux(i, j, k, ibg, bc.east,   loc, c, closure, K, id, clock, fields)
    q̃ˢ =  south_ib_flux(i, j, k, ibg, bc.south,  loc, c, closure, K, id, clock, fields)
    q̃ᴺ =  north_ib_flux(i, j, k, ibg, bc.north,  loc, c, closure, K, id, clock, fields)
    q̃ᴮ = bottom_ib_flux(i, j, k, ibg, bc.bottom, loc, c, closure, K, id, clock, fields)
    q̃ᵀ =    top_ib_flux(i, j, k, ibg, bc.top,    loc, c, closure, K, id, clock, fields)

    iᵂ, jˢ, kᴮ = index_right.((i, j, k), loc)
    iᴱ, jᴺ, kᵀ = index_left.((i, j, k), loc)
    LX, LY, LZ = loc

    # Impose i) immersed fluxes if we're on an immersed boundary or ii) zero otherwise.
    qᵂ = conditional_flux(iᵂ, j, k, ibg, flip(LX), LY, LZ, q̃ᵂ, zero(eltype(ibg)))
    qᴱ = conditional_flux(iᴱ, j, k, ibg, flip(LX), LY, LZ, q̃ᴱ, zero(eltype(ibg)))
    qˢ = conditional_flux(i, jˢ, k, ibg, LX, flip(LY), LZ, q̃ˢ, zero(eltype(ibg)))
    qᴺ = conditional_flux(i, jᴺ, k, ibg, LX, flip(LY), LZ, q̃ᴺ, zero(eltype(ibg)))
    qᴮ = conditional_flux(i, j, kᴮ, ibg, LX, LY, flip(LZ), q̃ᴮ, zero(eltype(ibg)))
    qᵀ = conditional_flux(i, j, kᵀ, ibg, LX, LY, flip(LZ), q̃ᵀ, zero(eltype(ibg)))

    return div(i, j, k, ibg, loc, qᵂ, qᴱ, qˢ, qᴺ, qᴮ, qᵀ)
end


@inline immersed_∂ⱼ_τ₁ⱼ(i, j, k, ibg::GFIBG, U, u_bc::IBC, closure, K, clock, fields) =
    immersed_flux_divergence(i, j, k, ibg, u_bc, (f, c, c), U.u, closure, K, nothing, clock, fields)

@inline immersed_∂ⱼ_τ₂ⱼ(i, j, k, ibg::GFIBG, U, v_bc::IBC, closure, K, clock, fields) =
    immersed_flux_divergence(i, j, k, ibg, v_bc, (c, f, c), U.v, closure, K, nothing, clock, fields)

@inline immersed_∂ⱼ_τ₃ⱼ(i, j, k, ibg::GFIBG, U, w_bc::IBC, closure, K, clock, fields) =
    immersed_flux_divergence(i, j, k, ibg, w_bc, (c, c, f), U.w, closure, K, nothing, clock, fields)

@inline immersed_∇_dot_qᶜ(i, j, k, ibg::GFIBG, C, c_bc::IBC, closure, K, id, clock, fields) =
    immersed_flux_divergence(i, j, k, ibg, c_bc, (c, c, c), C, closure, K, id, clock, fields)

#####
##### Boundary condition "regularization"
#####

const ZFBC = BoundaryCondition{Flux, Nothing}
regularize_immersed_boundary_condition(ibc::ZFBC, ibg::GFIBG, args...) = ibc # keep it

# Compiler hint
@inline immersed_flux_divergence(i, j, k, ibg::GFIBG, bc::ZFBC, loc, c, closure, K, id, clock, fields) = zero(ibg)

regularize_immersed_boundary_condition(default::DefaultBoundaryCondition, ibg::GFIBG, loc, field_name, args...) =
    regularize_immersed_boundary_condition(default.boundary_condition, ibg, loc, field_name, args...)

# Convert certain non-immersed boundary conditions to immersed boundary conditions
function regularize_immersed_boundary_condition(ibc::Union{VBC, GBC, FBC}, ibg::GFIBG, loc, field_name, args...)
    ibc = ImmersedBoundaryCondition(Tuple(ibc for i=1:6)...)
    regularize_immersed_boundary_condition(ibc, ibg, loc, field_name, args...) 
end

"""
    regularize_immersed_boundary_condition(bc::BoundaryCondition{C, <:ContinuousBoundaryFunction},
                                           topo, loc, dim, I, prognostic_field_names) where C
"""
function regularize_immersed_boundary_condition(bc::IBC, grid, loc, field_name, prognostic_field_names)

    topo = topology(grid)

    west   = loc[1] === Face ? nothing : regularize_boundary_condition(bc.west,   topo, loc, 1, LeftBoundary,  prognostic_field_names)
    east   = loc[1] === Face ? nothing : regularize_boundary_condition(bc.east,   topo, loc, 1, RightBoundary, prognostic_field_names)
    south  = loc[2] === Face ? nothing : regularize_boundary_condition(bc.south,  topo, loc, 2, LeftBoundary,  prognostic_field_names)
    north  = loc[2] === Face ? nothing : regularize_boundary_condition(bc.north,  topo, loc, 2, RightBoundary, prognostic_field_names)
    bottom = loc[3] === Face ? nothing : regularize_boundary_condition(bc.bottom, topo, loc, 3, LeftBoundary,  prognostic_field_names)
    top    = loc[3] === Face ? nothing : regularize_boundary_condition(bc.top,    topo, loc, 3, RightBoundary, prognostic_field_names)

    return ImmersedBoundaryCondition(; west, east, south, north, bottom, top)
end

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


