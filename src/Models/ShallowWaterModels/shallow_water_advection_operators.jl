using Oceananigans.Advection: 
    advective_momentum_flux_Uu,
    advective_momentum_flux_Uv,
    advective_momentum_flux_Vu,
    advective_momentum_flux_Vv

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ

#####
##### Momentum flux operators
#####

@inline momentum_flux_huu(i, j, k, grid, advection, solution) =
    @inbounds advective_momentum_flux_Uu(i, j, k, grid, advection, solution.uh, solution.uh) / solution.h[i, j, k]

@inline momentum_flux_hvu(i, j, k, grid, advection, solution) =
    @inbounds advective_momentum_flux_Vu(i, j, k, grid, advection, solution.vh, solution.uh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline momentum_flux_huv(i, j, k, grid, advection, solution) =
    @inbounds advective_momentum_flux_Uv(i, j, k, grid, advection, solution.uh, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline momentum_flux_hvv(i, j, k, grid, advection, solution) =
    @inbounds advective_momentum_flux_Vv(i, j, k, grid, advection, solution.vh, solution.vh) / solution.h[i, j, k]

#####
##### Momentum flux divergence operators
#####

@inline div_hUu(i, j, k, grid, advection, solution) =
    1 / Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᶜᶜ(i, j, k, grid, momentum_flux_huu, advection, solution) +
                               δyᶠᶜᶜ(i, j, k, grid, momentum_flux_hvu, advection, solution))

@inline div_hUv(i, j, k, grid, advection, solution) =
    1 / Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᶠᶜ(i, j, k, grid, momentum_flux_huv, advection, solution) +
                               δyᶜᶠᶜ(i, j, k, grid, momentum_flux_hvv, advection, solution))

# Support for no advection
@inline div_hUu(i, j, k, grid, ::Nothing, solution) = zero(grid)
@inline div_hUv(i, j, k, grid, ::Nothing, solution) = zero(grid)

#####
##### Mass transport divergence operator
#####

"""
    div_Uh(i, j, k, grid, solution)

Calculates the divergence of the mass flux into a cell.
"""
@inline function div_Uh(i, j, k, grid, solution)
    1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, solution.uh) + 
                             δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, solution.vh))
end

#####
##### Tracer advection operator
#####

@inline transport_tracer_flux_x(i, j, k, grid, advection, uh, h, c) =
    @inbounds advective_tracer_flux_x(i, j, k, grid, advection, uh, c) / h[i, j, k]

@inline transport_tracer_flux_y(i, j, k, grid, advection, vh, h, c) =
    @inbounds advective_tracer_flux_y(i, j, k, grid, advection, vh, c) / h[i, j, k]

"""
    div_Uc(i, j, k, grid, advection, U, c)

Calculates the divergence of the flux of a tracer quantity c being advected by
a velocity field U = (u, v), ∇·(Uc).
"""
@inline function div_Uc(i, j, k, grid, advection, solution, c)
    1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᶜ(i, j, k, grid, transport_tracer_flux_x, advection, solution.uh, solution.h, c) +        
                             δyᶜᶜᶜ(i, j, k, grid, transport_tracer_flux_y, advection, solution.vh, solution.h, c))
end

# Support for no advection
@inline div_Uc(i, j, k, grid, ::Nothing, solution, c) = zero(grid)

@inline u(i, j, k, grid, solution) = @inbounds solution.uh[i, j, k] / solution.h[i, j, k]
@inline v(i, j, k, grid, solution) = @inbounds solution.vh[i, j, k] / solution.h[i, j, k]

"""
    c_div_U(i, j, k, grid, advection, U)

Calculates the product of the tracer concentration c with 
the horizontal divergence of the velocity field U = (u, v), c ∇·(U),
"""
@inline function c_div_U(i, j, k, grid, solution, c)
    @inbounds c[i, j, k] * 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᶜ(i, j, k, grid, u, solution) + δyᶜᶜᶜ(i, j, k, grid, v, solution))
end

# Support for no advection
@inline c_div_Uc(i, j, k, grid, ::Nothing, solution, c) = zero(grid)
