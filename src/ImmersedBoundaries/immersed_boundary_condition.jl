using Oceananigans.BoundaryConditions: Flux, Value, flip, BoundaryCondition, getbc
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, h_diffusivity, z_diffusivity
using Oceananigans.Operators: index_left, index_right, Δx, Δy, Δz, div

import Oceananigans.BoundaryConditions: regularize_immersed_boundary_condition

struct ImmersedBoundaryCondition{W, E, S, N, B, T}
    west :: W                  
    east :: E
    south :: S   
    north :: N
    bottom :: B
    top :: T
end

"""
    ImmersedBoundaryCondition(; interfaces...)

Return an ImmersedBoundaryCondition with conditions on individual
cell `interfaces ∈ (west, east, south, north, bottom, top)`
between the fluid and immersed boundary.
"""
function ImmersedBoundaryCondition(;
                                   west = nothing,
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
const ASD = AbstractScalarDiffusivity
const IBC = ImmersedBoundaryCondition

# Well, this part is easy anyways
regularize_immersed_boundary_condition(ibc::Union{VBC, FBC}, ibg::GFIBG, loc, field_name, args...) =
    ImmersedBoundaryCondition(Tuple(ibc for i=1:6)...)

# Of course this is just fine!
const ZFBC = BoundaryCondition{Flux, Nothing}
regularize_immersed_boundary_condition(ibc::ZFBC, ibg::GFIBG, args...) = ibc

# Don't blame us, the user gave us this crazy boundary condition!
regularize_immersed_boundary_condition(ibc::IBC, ibg::GFIBG, loc, field_name, args...) = ibc

@inline function left_ib_flux_value_bc(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
    cᵇ = getbc(bc, i, j, k, ibg, clock, fields)
    cⁱʲᵏ = @inbounds c[i, j, k]
    return - 2 * κ * (cⁱʲᵏ - cᵇ) / Δ
end

@inline function right_ib_flux_value_bc(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
    cᵇ = getbc(bc, i, j, k, ibg, clock, fields)
    cⁱʲᵏ = @inbounds c[i, j, k]
    return - 2 * κ * (cᵇ - cⁱʲᵏ) / Δ
end

@inline function west_ib_flux(i, j, k, ibg, bc::VBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δx(index_left(i, LX), j, k, ibg, LX, LY, LZ)
    κ = h_diffusivity(i, j, k, ibg, flip(LX), LY, LZ, closure, K, id, clock)
    return left_ib_flux_value_bc(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
end

@inline function east_ib_flux(i, j, k, ibg, bc::VBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δx(index_right(i, LX), j, k, ibg, LX, LY, LZ)
    κ = h_diffusivity(i, j, k, ibg, flip(LX), LY, LZ, closure, K, id, clock)
    return right_ib_flux_value_bc(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
end

@inline function south_ib_flux(i, j, k, ibg, bc::VBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δy(i, index_left(j, LY), k, ibg, LX, LY, LZ)
    κ = h_diffusivity(i, j, k, ibg, LX, flip(LY), LZ, closure, K, id, clock)
    return left_ib_flux_value_bc(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
end

@inline function north_ib_flux(i, j, k, ibg, bc::VBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δy(i, index_right(j, LY), k, ibg, LX, LY, LZ)
    κ = h_diffusivity(i, j, k, ibg, LX, flip(LY), LZ, closure, K, id, clock)
    return right_ib_flux_value_bc(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
end

@inline function bottom_ib_flux(i, j, k, ibg, bc::VBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δz(i, j, index_left(k, LZ), ibg, LX, LY, LZ)
    κ = z_diffusivity(i, j, k, ibg, LX, LY, flip(LZ), closure, K, id, clock)
    return left_ib_flux_value_bc(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
end

@inline function top_ib_flux(i, j, k, ibg, bc::VBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields)
    Δ = Δz(i, j, index_right(k, LZ), ibg, LX, LY, LZ)
    κ = z_diffusivity(i, j, k, ibg, LX, LY, flip(LZ), closure, K, id, clock)
    return right_ib_flux_value_bc(i, j, k, ibg, κ, Δ, bc, c, clock, fields)
end

#####
##### Immersed flux divergence
#####
##### Note: this may not work with Flat dimensions.
#####

@inline function immersed_flux_divergence(i, j, k, ibg::GFIBG, bc, loc, c, closure, K, id, args...)
    q̃ᵂ =   west_ib_flux(i, j, k, ibg, bc.west,   loc, c, closure, K, id, args...)
    q̃ᴱ =   east_ib_flux(i, j, k, ibg, bc.east,   loc, c, closure, K, id, args...)
    q̃ˢ =  south_ib_flux(i, j, k, ibg, bc.south,  loc, c, closure, K, id, args...)
    q̃ᴺ =  north_ib_flux(i, j, k, ibg, bc.north,  loc, c, closure, K, id, args...)
    q̃ᴮ = bottom_ib_flux(i, j, k, ibg, bc.bottom, loc, c, closure, K, id, args...)
    q̃ᵀ =    top_ib_flux(i, j, k, ibg, bc.top,    loc, c, closure, K, id, args...)

    iᵂ, jˢ, kᴮ = index_right.((i, j, k), loc)
    iᴱ, jᴺ, kᵀ = index_left.((i, j, k), loc)
    LX, LY, LZ = loc

    nc = immersed_peripheral_node # so we don't add fluxes across _non-immersed_ boundaries
    qᵂ = conditional_flux(iᵂ, j, k, ibg, flip(LX), LY, LZ, q̃ᵂ, zero(eltype(ibg)), nc)
    qᴱ = conditional_flux(iᴱ, j, k, ibg, flip(LX), LY, LZ, q̃ᴱ, zero(eltype(ibg)), nc)
    qˢ = conditional_flux(i, jˢ, k, ibg, LX, flip(LY), LZ, q̃ˢ, zero(eltype(ibg)), nc)
    qᴺ = conditional_flux(i, jᴺ, k, ibg, LX, flip(LY), LZ, q̃ᴺ, zero(eltype(ibg)), nc)
    qᴮ = conditional_flux(i, j, kᴮ, ibg, LX, LY, flip(LZ), q̃ᴮ, zero(eltype(ibg)), nc)
    qᵀ = conditional_flux(i, j, kᵀ, ibg, LX, LY, flip(LZ), q̃ᵀ, zero(eltype(ibg)), nc)

    return div(i, j, k, ibg, loc, qᵂ, qᴱ, qˢ, qᴺ, qᴮ, qᵀ)
end

@inline immersed_∂ⱼ_τ₁ⱼ(i, j, k, ibg::GFIBG, U, u_bc::IBC, closure, K, args...) =
    immersed_flux_divergence(i, j, k, ibg, u_bc, (f, c, c), U.u, closure, K, nothing, args...)

@inline immersed_∂ⱼ_τ₂ⱼ(i, j, k, ibg::GFIBG, U, v_bc::IBC, closure, K, args...) =
    immersed_flux_divergence(i, j, k, ibg, v_bc, (c, f, c), U.v, closure, K, nothing, args...)

@inline immersed_∂ⱼ_τ₃ⱼ(i, j, k, ibg::GFIBG, U, w_bc::IBC, closure, K, args...) =
    immersed_flux_divergence(i, j, k, ibg, w_bc, (c, c, f), U.w, closure, K, nothing, args...)

@inline immersed_∇_dot_qᶜ(i, j, k, ibg::GFIBG, C, c_bc::IBC, closure, K, id, args...) =
    immersed_flux_divergence(i, j, k, ibg, c_bc, (c, c, c), C, closure, K, id, args...)

