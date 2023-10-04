using Oceananigans.Advection: 
    _advective_momentum_flux_Uu,
    _advective_momentum_flux_Uv,
    _advective_momentum_flux_Vu,
    _advective_momentum_flux_Vv,
    _advective_tracer_flux_x, 
    _advective_tracer_flux_y,
    horizontal_advection_U,
    horizontal_advection_V,
    bernoulli_head_U,
    bernoulli_head_V

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ


#####
##### Momentum flux operators
#####

@inline momentum_flux_huu(i, j, k, grid, advection, solution) =
    @inbounds _advective_momentum_flux_Uu(i, j, k, grid, advection, solution[1], solution[1]) / solution.h[i, j, k]

@inline momentum_flux_hvu(i, j, k, grid, advection, solution) =
    @inbounds _advective_momentum_flux_Vu(i, j, k, grid, advection, solution[2], solution[1]) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline momentum_flux_huv(i, j, k, grid, advection, solution) =
    @inbounds _advective_momentum_flux_Uv(i, j, k, grid, advection, solution[1], solution[2]) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline momentum_flux_hvv(i, j, k, grid, advection, solution) =
    @inbounds _advective_momentum_flux_Vv(i, j, k, grid, advection, solution[2], solution[2]) / solution.h[i, j, k]

#####
##### Momentum flux divergence operators
#####

@inline div_mom_u(i, j, k, grid, advection, solution, formulation) =
    1 / Azᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_huu, advection, solution) +
                                δyᵃᶜᵃ(i, j, k, grid, momentum_flux_hvu, advection, solution))

@inline div_mom_v(i, j, k, grid, advection, solution, formulation) =
    1 / Azᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_huv, advection, solution) +
                                δyᵃᶠᵃ(i, j, k, grid, momentum_flux_hvv, advection, solution))

@inline div_mom_u(i, j, k, grid, advection, solution, ::VectorInvariantFormulation) = (
    + horizontal_advection_U(i, j, k, grid, advection, solution[1], solution[2])  # Vertical relative vorticity term
    + bernoulli_head_U(i, j, k, grid, advection, solution[1], solution[2]))     # Bernoulli head term
    
@inline div_mom_v(i, j, k, grid, advection, solution, ::VectorInvariantFormulation) = (
    + horizontal_advection_V(i, j, k, grid, advection, solution[1], solution[2])  # Vertical relative vorticity term
    + bernoulli_head_V(i, j, k, grid, advection, solution[1], solution[2]))     # Bernoulli head term

# Support for no advection
@inline div_mom_u(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution, formulation) where FT = zero(FT)
@inline div_mom_v(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution, formulation) where FT = zero(FT)
@inline div_mom_u(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution, ::VectorInvariantFormulation) where FT = zero(FT)
@inline div_mom_v(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution, ::VectorInvariantFormulation) where FT = zero(FT)

#####
##### Mass transport divergence operator
#####

"""
    div_Uh(i, j, k, grid, advection, solution, formulation)

Calculate the divergence of the mass flux into a cell,

```
1/Az * [δxᶜᵃᵃ(Δy * uh) + δyᵃᶜᵃ(Δx * vh)]
```

which ends up at the location `ccc`.
"""
@inline function div_Uh(i, j, k, grid, advection, solution, formulation)
    return 1/Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, solution[1]) + 
                                     δyᵃᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, solution[2]))
end

@inline div_Uh(i, j, k, grid, advection, solution, formulation::VectorInvariantFormulation) =
        div_Uc(i, j, k, grid, advection, solution, solution.h, formulation)

#####
##### Tracer advection operator
#####

@inline transport_tracer_flux_x(i, j, k, grid, advection, uh, h, c) =
    @inbounds _advective_tracer_flux_x(i, j, k, grid, advection, uh, c) / ℑxᶠᵃᵃ(i, j, k, grid, h)

@inline transport_tracer_flux_y(i, j, k, grid, advection, vh, h, c) =
    @inbounds _advective_tracer_flux_y(i, j, k, grid, advection, vh, c) / ℑyᵃᶠᵃ(i, j, k, grid, h)

"""
    div_Uc(i, j, k, grid, advection, solution, c, formulation)

Calculate the divergence of the flux of a tracer quantity ``c`` being advected by
a velocity field ``𝐔 = (u, v)``, ``𝛁·(𝐔c)``,

```
1/Az * [δxᶜᵃᵃ(Δy * uh * ℑxᶠᵃᵃ(c) / h) + δyᵃᶜᵃ(Δx * vh * ℑyᵃᶠᵃ(c) / h)]
```

which ends up at the location `ccc`.
"""

@inline function div_Uc(i, j, k, grid, advection, solution, c, formulation)
    return 1/Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, transport_tracer_flux_x, advection, solution[1], solution.h, c) +        
                                     δyᵃᶜᵃ(i, j, k, grid, transport_tracer_flux_y, advection, solution[2], solution.h, c))
end

@inline function div_Uc(i, j, k, grid, advection, solution, c, ::VectorInvariantFormulation)
    return 1/Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, solution[1], c) +
                                     δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, solution[2], c)) 
end

# Support for no advection
@inline div_Uc(i, j, k, grid::AbstractGrid, ::Nothing, solution, c, formulation) = zero(grid)
@inline div_Uh(i, j, k, grid::AbstractGrid, ::Nothing, solution, formulation)    = zero(grid)

# Disambiguation
@inline div_Uc(i, j, k, grid::AbstractGrid, ::Nothing, solution, c, ::VectorInvariantFormulation) = zero(grid)
@inline div_Uh(i, j, k, grid::AbstractGrid, ::Nothing, solution, ::VectorInvariantFormulation)    = zero(grid)

@inline u(i, j, k, grid, solution) = @inbounds solution.uh[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, solution.h)
@inline v(i, j, k, grid, solution) = @inbounds solution.vh[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, solution.h)

"""
    c_div_U(i, j, k, grid, solution, c, formulation)

Calculate the product of the tracer concentration ``c`` with 
the horizontal divergence of the velocity field ``𝐔 = (u, v)``, ``c ∇·𝐔``,

```
c * 1/Az * [δxᶜᵃᵃ(Δy * uh / h) + δyᵃᶜᵃ(Δx * vh / h)]
```

which ends up at the location `ccc`.
"""
@inline c_div_U(i, j, k, grid, solution, c, formulation) = 
    @inbounds c[i, j, k] * 1/Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u, solution) + δyᵃᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v, solution))

@inline c_div_U(i, j, k, grid, solution, c, ::VectorInvariantFormulation) = 
    @inbounds c[i, j, k] * 1/Azᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, solution[1]) + δyᵃᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, solution[2]))

# Support for no advection
@inline c_div_Uc(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution, c, formulation) where FT = zero(FT)
