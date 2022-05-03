using Oceananigans.Grids: idxᴿ, idxᴸ, flip
using Oceananigans.BoundaryConditions: FBC, VBC, GBC, ZFBC, BoundaryCondition, ContinuousBoundaryFunction
using Oceananigans.BoundaryConditions: getbc, regularize_boundary_condition, LeftBoundary, RightBoundary
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, h_diffusivity, z_diffusivity
using Oceananigans.Operators: Δx, Δy, Δz, div

import Oceananigans.BoundaryConditions: west_flux, east_flux, south_flux, north_flux, bottom_flux, top_flux
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
    side_flux = Symbol(side, :_flux)
    @eval begin
        @inline $side_flux(i, j, k, ibg::IBG, bc::FBC, loc, c, closure, K, id, args...) = + getbc(bc, i, j, k, ibg, args...)
        @inline $side_flux(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
    end
end

for side in (:east, :north, :top)
    side_flux = Symbol(side, :_flux)
    @eval begin
        # Note sign convection for fluxes
        @inline $side_flux(i, j, k, ibg::IBG, bc::FBC, loc, c, closure, K, id, args...) = - getbc(bc, i, j, k, ibg, args...)
        @inline $side_flux(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
    end
end

#####
##### Immersed flux divergence
#####

@inline function immersed_flux_divergence(i, j, k, ibg::GFIBG, bc, loc, c, closure, K, id, clock, fields)
    # Fetch fluxes associated with bc::ImmersedBoundaryCondition
    q̃ᵂ =   west_flux(i, j, k, ibg, bc.west,   loc, c, closure, K, id, clock, fields)
    q̃ᴱ =   east_flux(i, j, k, ibg, bc.east,   loc, c, closure, K, id, clock, fields)
    q̃ˢ =  south_flux(i, j, k, ibg, bc.south,  loc, c, closure, K, id, clock, fields)
    q̃ᴺ =  north_flux(i, j, k, ibg, bc.north,  loc, c, closure, K, id, clock, fields)
    q̃ᴮ = bottom_flux(i, j, k, ibg, bc.bottom, loc, c, closure, K, id, clock, fields)
    q̃ᵀ =    top_flux(i, j, k, ibg, bc.top,    loc, c, closure, K, id, clock, fields)

    iᵂ, jˢ, kᴮ = idxᴿ.((i, j, k), loc)
    iᴱ, jᴺ, kᵀ = idxᴸ.((i, j, k), loc)
    LX, LY, LZ = loc

    # Impose i) immersed fluxes if we're on an immersed boundary or ii) zero otherwise.
    qᵂ = conditional_x_flux(iᵂ, j, k, ibg, flip(LX), LY, LZ, q̃ᵂ, zero(ibg))
    qᴱ = conditional_x_flux(iᴱ, j, k, ibg, flip(LX), LY, LZ, q̃ᴱ, zero(ibg))
    qˢ = conditional_y_flux(i, jˢ, k, ibg, LX, flip(LY), LZ, q̃ˢ, zero(ibg))
    qᴺ = conditional_y_flux(i, jᴺ, k, ibg, LX, flip(LY), LZ, q̃ᴺ, zero(ibg))
    qᴮ = conditional_z_flux(i, j, kᴮ, ibg, LX, LY, flip(LZ), q̃ᴮ, zero(ibg))
    qᵀ = conditional_z_flux(i, j, kᵀ, ibg, LX, LY, flip(LZ), q̃ᵀ, zero(ibg))

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

