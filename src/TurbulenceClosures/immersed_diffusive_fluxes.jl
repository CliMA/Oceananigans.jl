using Oceananigans.BoundaryConditions: getbc, left_gradient, right_gradient
using Oceananigans.BoundaryConditions: FBC, ZFBC, PBC, MCBC, DCBC, VBC, GBC, MBC, NFBC, ZBC
using Oceananigans.Operators: index_left, index_right, Δx, Δy, Δz, div

using Oceananigans.Advection: conditional_flux

using Oceananigans.Advection: conditional_flux_ccc,
                              conditional_flux_ffc,
                              conditional_flux_fcf,
                              conditional_flux_cff,
                              conditional_flux_fcc,
                              conditional_flux_cfc,
                              conditional_flux_ccf

using Oceananigans.ImmersedBoundaries: GFIBG, IBC

const IBG = ImmersedBoundaryGrid

#####
##### Immersed Diffusive fluxes
#####

# ccc, ffc, fcf
@inline _viscous_flux_ux(i, j, k, ibg::IBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(ibg), viscous_flux_ux(i, j, k, ibg, args...))
@inline _viscous_flux_uy(i, j, k, ibg::IBG, args...) = conditional_flux_ffc(i, j, k, ibg, zero(ibg), viscous_flux_uy(i, j, k, ibg, args...))
@inline _viscous_flux_uz(i, j, k, ibg::IBG, args...) = conditional_flux_fcf(i, j, k, ibg, zero(ibg), viscous_flux_uz(i, j, k, ibg, args...))

 # ffc, ccc, cff
@inline _viscous_flux_vx(i, j, k, ibg::IBG, args...) = conditional_flux_ffc(i, j, k, ibg, zero(ibg), viscous_flux_vx(i, j, k, ibg, args...))
@inline _viscous_flux_vy(i, j, k, ibg::IBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(ibg), viscous_flux_vy(i, j, k, ibg, args...))
@inline _viscous_flux_vz(i, j, k, ibg::IBG, args...) = conditional_flux_cff(i, j, k, ibg, zero(ibg), viscous_flux_vz(i, j, k, ibg, args...))

 # fcf, cff, ccc
@inline _viscous_flux_wx(i, j, k, ibg::IBG, args...) = conditional_flux_fcf(i, j, k, ibg, zero(ibg), viscous_flux_wx(i, j, k, ibg, args...))
@inline _viscous_flux_wy(i, j, k, ibg::IBG, args...) = conditional_flux_cff(i, j, k, ibg, zero(ibg), viscous_flux_wy(i, j, k, ibg, args...))
@inline _viscous_flux_wz(i, j, k, ibg::IBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(ibg), viscous_flux_wz(i, j, k, ibg, args...))

# fcc, cfc, ccf
@inline _diffusive_flux_x(i, j, k, ibg::IBG, args...) = conditional_flux_fcc(i, j, k, ibg, zero(ibg), diffusive_flux_x(i, j, k, ibg, args...))
@inline _diffusive_flux_y(i, j, k, ibg::IBG, args...) = conditional_flux_cfc(i, j, k, ibg, zero(ibg), diffusive_flux_y(i, j, k, ibg, args...))
@inline _diffusive_flux_z(i, j, k, ibg::IBG, args...) = conditional_flux_ccf(i, j, k, ibg, zero(ibg), diffusive_flux_z(i, j, k, ibg, args...))

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
    side_ib_flux = Symbol(side, :_ib_flux)
    @eval begin
        @inline $side_ib_flux(i, j, k, ibg, ::Nothing, args...) = zero(eltype(ibg))
        @inline $side_ib_flux(i, j, k, ibg, bc::FBC, loc, c, closure, K, id, clock, fields, I) = + getbc(bc, I..., ibg, clock, fields)
    end
end

for side in (:east, :north, :top)
    side_ib_flux = Symbol(side, :_ib_flux)
    @eval begin
        @inline $side_ib_flux(i, j, k, ibg, ::Nothing, args...) = zero(eltype(ibg))
        @inline $side_ib_flux(i, j, k, ibg, bc::FBC, loc, c, closure, K, id, clock, fields, I) = - getbc(bc, I..., ibg, clock, fields)
    end
end

#####
##### Value, Gradient and Mixed conditions: the flux of a scalar diffusivity through the wall, from the gradient the condition
##### sets between the wall and the node half a cell away. `I` are the indices that read the condition: the cell next to an
##### immersed wall, or the boundary plane on a side of the domain.
#####

const VGMBC = Union{VBC, GBC, MBC}
const ASD = AbstractScalarDiffusivity

# Metric and index gymnastics for the 6 facets of the cube

@inline flip(::Type{Face}) = Center
@inline flip(::Type{Center}) = Face

@inline flip(::Face) = Center()
@inline flip(::Center) = Face()

# The flux through a facet whose condition sets the value or the gradient: -κ ∇c, with the diffusivity across the facet
for (side, Δ, diffusivity, index, gradient, d) in ((:west,   Δx, :h_diffusivity, index_left,  left_gradient,  1),
                                                    (:east,   Δx, :h_diffusivity, index_right, right_gradient, 1),
                                                    (:south,  Δy, :h_diffusivity, index_left,  left_gradient,  2),
                                                    (:north,  Δy, :h_diffusivity, index_right, right_gradient, 2),
                                                    (:bottom, Δz, :z_diffusivity, index_left,  left_gradient,  3),
                                                    (:top,    Δz, :z_diffusivity, index_right, right_gradient, 3))
    flux      = Symbol(:_, side, :_ib_flux)
    indices   = Any[:i, :j, :k]
    locations = Any[:LX, :LY, :LZ]
    indices[d]   = :($index($(indices[d]), $(locations[d])))
    locations[d] = :(flip($(locations[d])))

    @eval @inline function $flux(i, j, k, ibg, bc::VGMBC, (LX, LY, LZ), c, closure::ASD, K, id, clock, fields, I)
        Δ  = $Δ(i, j, k, ibg, LX, LY, LZ)
        κ  = $diffusivity($(indices...), ibg, $(locations...), closure, K, id, clock, fields)
        c¹ = @inbounds c[i, j, k]
        ∇c = $gradient(bc, c¹, Δ, I..., ibg, clock, fields)
        return - κ * ∇c
    end
end

sides = [:west, :east, :south, :north, :bottom, :top]

for side in sides
    flux = Symbol(side, "_ib_flux")
    _flux = Symbol("_", flux)

    @eval begin
        @inline $flux(i, j, k, ibg, bc::VGMBC, args...) = $_flux(i, j, k, ibg, bc::VGMBC, args...)
        @inline $_flux(i, j, k, ibg, bc::VGMBC, args...) = zero(ibg) # fallback for non-ASD closures

        @inline $flux(i, j, k, ibg, bc::VGMBC, loc, c, closures::Tuple{<:Any}, Ks, id, clock, fields, I) =
            $_flux(i, j, k, ibg, bc, loc, c, closures[1], Ks[1], id, clock, fields, I)

        @inline $flux(i, j, k, ibg, bc::VGMBC, loc, c, closures::Tuple{<:Any, <:Any}, Ks, id, clock, fields, I) =
            $_flux(i, j, k, ibg, bc, loc, c, closures[1], Ks[1], id, clock, fields, I) +
            $_flux(i, j, k, ibg, bc, loc, c, closures[2], Ks[2], id, clock, fields, I)

        @inline $flux(i, j, k, ibg, bc::VGMBC, loc, c, closures::Tuple{<:Any, <:Any, <:Any}, Ks, id, clock, fields, I) =
            $_flux(i, j, k, ibg, bc, loc, c, closures[1], Ks[1], id, clock, fields, I) +
            $_flux(i, j, k, ibg, bc, loc, c, closures[2], Ks[2], id, clock, fields, I) +
            $_flux(i, j, k, ibg, bc, loc, c, closures[3], Ks[3], id, clock, fields, I)

        @inline $flux(i, j, k, ibg, bc::VGMBC, loc, c, closures::Tuple{<:Any, <:Any, <:Any, <:Any}, Ks, id, clock, fields, I) =
            $_flux(i, j, k, ibg, bc, loc, c, closures[1], Ks[1], id, clock, fields, I) +
            $_flux(i, j, k, ibg, bc, loc, c, closures[2], Ks[2], id, clock, fields, I) +
            $_flux(i, j, k, ibg, bc, loc, c, closures[3], Ks[3], id, clock, fields, I) +
            $_flux(i, j, k, ibg, bc, loc, c, closures[4], Ks[4], id, clock, fields, I)

        @inline $flux(i, j, k, ibg, bc::VGMBC, loc, c, closures::Tuple, Ks, id, clock, fields, I) =
            $_flux(i, j, k, ibg, bc, loc, c, closures[1], Ks[1], id, clock, fields, I) +
             $flux(i, j, k, ibg, bc, loc, c, closures[2:end], Ks[2:end], id, clock, fields, I)
    end
end

#####
##### Immersed flux divergence
#####

@inline immersed_flux_divergence(i, j, k, grid, bc, loc, c, closure, K, id, clock, fields) = zero(grid)

@inline function immersed_flux_divergence(i, j, k, ibg::GFIBG, bc::IBC, loc, c, closure, K, id, clock, fields)
    # Fetch fluxes across immersed boundary
    q̃ᵂ =   west_ib_flux(i, j, k, ibg, bc.west,   loc, c, closure, K, id, clock, fields, (i, j, k))
    q̃ᴱ =   east_ib_flux(i, j, k, ibg, bc.east,   loc, c, closure, K, id, clock, fields, (i, j, k))
    q̃ˢ =  south_ib_flux(i, j, k, ibg, bc.south,  loc, c, closure, K, id, clock, fields, (i, j, k))
    q̃ᴺ =  north_ib_flux(i, j, k, ibg, bc.north,  loc, c, closure, K, id, clock, fields, (i, j, k))
    q̃ᴮ = bottom_ib_flux(i, j, k, ibg, bc.bottom, loc, c, closure, K, id, clock, fields, (i, j, k))
    q̃ᵀ =    top_ib_flux(i, j, k, ibg, bc.top,    loc, c, closure, K, id, clock, fields, (i, j, k))

    iᵂ, jˢ, kᴮ = map(index_left,  (i, j, k), loc) # Broadcast instead of map causes inference failure
    iᴱ, jᴺ, kᵀ = map(index_right, (i, j, k), loc)
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

#####
##### Conditions on the domain sides
#####

# Flux through the face `i`, `j` or `k` on a domain side, positive along the coordinate, zero away from that side;
# Value, Gradient and Mixed conditions take the wall flux of the cell next to the face, read on the boundary plane
for (side, boundary_face, plane, cell) in ((:west,   :(i == 1),           (:j, :k), (:i, :j, :k)),
                                           (:east,   :(i == grid.Nx + 1), (:j, :k), (:(i - 1), :j, :k)),
                                           (:south,  :(j == 1),           (:i, :k), (:i, :j, :k)),
                                           (:north,  :(j == grid.Ny + 1), (:i, :k), (:i, :(j - 1), :k)),
                                           (:bottom, :(k == 1),           (:i, :j), (:i, :j, :k)),
                                           (:top,    :(k == grid.Nz + 1), (:i, :j), (:i, :j, :(k - 1))))

    domain_flux = Symbol(side, :_domain_flux)
    wall_flux   = Symbol(side, :_ib_flux)

    @eval begin
        @inline $domain_flux(i, j, k, grid, bc, args...) = zero(grid)

        @inline $domain_flux(i, j, k, grid, bc::FBC, loc, c, closure, K, id, clock, fields, buoyancy) = ifelse($boundary_face,
                                                                                                               getbc(bc, $(plane...), grid, clock, fields, closure, buoyancy),
                                                                                                               zero(grid))

        @inline $domain_flux(i, j, k, grid, bc::VGMBC, loc, c, closure, K, id, clock, fields, buoyancy) = ifelse($boundary_face,
                                                                                                                 $wall_flux($(cell...), grid, bc, loc, c, closure, K, id, clock, fields, ($(plane...),)),
                                                                                                                 zero(grid))
    end
end

const NotFluxBC = Union{PBC, MCBC, DCBC, NFBC, ZFBC, ZBC, Nothing}
const NoDomainFluxes = NamedTuple{(:west, :east, :south, :north, :bottom, :top, :immersed),
                                  <:Tuple{NotFluxBC, NotFluxBC, NotFluxBC, NotFluxBC, NotFluxBC, NotFluxBC, Any}}

@inline domain_flux_divergence(i, j, k, grid, ::NoDomainFluxes, loc, args...) = zero(grid)

@inline function domain_flux_divergence(i, j, k, grid, bcs, loc, args...)
    iᵂ, jˢ, kᴮ = map(index_left,  (i, j, k), loc)
    iᴱ, jᴺ, kᵀ = map(index_right, (i, j, k), loc)

    qᵂ =   west_domain_flux(iᵂ, j, k, grid, bcs.west,   loc, args...)
    qᴱ =   east_domain_flux(iᴱ, j, k, grid, bcs.east,   loc, args...)
    qˢ =  south_domain_flux(i, jˢ, k, grid, bcs.south,  loc, args...)
    qᴺ =  north_domain_flux(i, jᴺ, k, grid, bcs.north,  loc, args...)
    qᴮ = bottom_domain_flux(i, j, kᴮ, grid, bcs.bottom, loc, args...)
    qᵀ =    top_domain_flux(i, j, kᵀ, grid, bcs.top,    loc, args...)

    return div(i, j, k, grid, loc, qᵂ, qᴱ, qˢ, qᴺ, qᴮ, qᵀ)
end

#####
##### Divergence of the fluxes through the boundary faces of a cell, on the domain sides and on the immersed boundary
#####

@inline boundary_∂ⱼ_τ₁ⱼ(i, j, k, grid, u, u_bcs, closure, K, clock, fields, buoyancy) =
    immersed_flux_divergence(i, j, k, grid, u_bcs.immersed, (f, c, c), u, closure, K, nothing, clock, fields) +
      domain_flux_divergence(i, j, k, grid, u_bcs, (f, c, c), u, closure, K, nothing, clock, fields, buoyancy)

@inline boundary_∂ⱼ_τ₂ⱼ(i, j, k, grid, v, v_bcs, closure, K, clock, fields, buoyancy) =
    immersed_flux_divergence(i, j, k, grid, v_bcs.immersed, (c, f, c), v, closure, K, nothing, clock, fields) +
      domain_flux_divergence(i, j, k, grid, v_bcs, (c, f, c), v, closure, K, nothing, clock, fields, buoyancy)

@inline boundary_∂ⱼ_τ₃ⱼ(i, j, k, grid, w, w_bcs, closure, K, clock, fields, buoyancy) =
    immersed_flux_divergence(i, j, k, grid, w_bcs.immersed, (c, c, f), w, closure, K, nothing, clock, fields) +
      domain_flux_divergence(i, j, k, grid, w_bcs, (c, c, f), w, closure, K, nothing, clock, fields, buoyancy)

@inline boundary_∇_dot_qᶜ(i, j, k, grid, C, c_bcs, closure, K, id, clock, fields, buoyancy) =
    immersed_flux_divergence(i, j, k, grid, c_bcs.immersed, (c, c, c), C, closure, K, id, clock, fields) +
      domain_flux_divergence(i, j, k, grid, c_bcs, (c, c, c), C, closure, K, id, clock, fields, buoyancy)
