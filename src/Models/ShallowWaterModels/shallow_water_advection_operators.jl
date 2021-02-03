using Oceananigans.Grids: AbstractGrid

import Oceananigans.Advection: div_Uc

#####
##### Momentum flux operators
#####

@inline momentum_flux_huu(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_uu(i, j, k, grid, advection, solution.uh, solution.uh) / solution.h[i, j, k]

@inline momentum_flux_huv(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_uv(i, j, k, grid, advection, solution.uh, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline momentum_flux_hvu(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_vu(i, j, k, grid, advection, solution.uh, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline momentum_flux_hvv(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_vv(i, j, k, grid, advection, solution.vh, solution.vh) / solution.h[i, j, k]

#####
##### Momentum flux divergence operators
#####

@inline div_hUu(i, j, k, grid, advection, solution) =
    1 / Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_huu, advection, solution) +
                               δyᵃᶜᵃ(i, j, k, grid, momentum_flux_huv, advection, solution))

@inline div_hUv(i, j, k, grid, advection, solution) =
    1 / Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_hvu, advection, solution) +
                               δyᵃᶠᵃ(i, j, k, grid, momentum_flux_hvv, advection, solution))

# Support for no advection
@inline div_hUu(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution) where FT = zero(FT)
@inline div_hUv(i, j, k, grid::AbstractGrid{FT}, ::Nothing, solution) where FT = zero(FT)

#####
##### Primitive advection
#####

U_dot_grad_u(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, u) where FT = zero(FT)
U_dot_grad_v(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, u) where FT = zero(FT)

"""
    U_dot_grad_u(i, j, k, grid, advection, U::PrimitiveSolutionLinearizedHeightFields, u)

Returns...
"""
@inline U_dot_grad_u(i, j, k, grid, advection, u, v) = @inbounds (               u[i, j, k] * ℑxᶠᵃᵃ(i, j, k, grid, ∂xᶜᵃᵃ, u)
                                                                  + ℑyᵃᶜᵃ(i, j, k, grid, v) * ℑyᵃᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, u) )

"""
    U_dot_grad_v(i, j, k, grid, advection, U::PrimitiveSolutionLinearizedHeightFields, v)

"""
@inline U_dot_grad_v(i, j, k, grid, advection, u, v) = @inbounds (ℑyᵃᶠᵃ(i, j, k, grid, u) * ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᵃᵃ, v)
                                                                  +            v[i, j, k] * ℑyᵃᶠᵃ(i, j, k, grid, ∂yᵃᶜᵃ, v))

"""
    div_Uc(i, j, k, grid, advection, U::PrimitiveSolutionLinearizedHeightFields, v)

Calculate the advection of shallow water momentum in the y-direction using the conservative form, ∇·(Uv)

```math
    1/Aᵘ * [δxᶠᵃᵃ(ℑxᶜᵃᵃ(Ax * u) * ℑxᶜᵃᵃ(u)) + δy_fca(ℑxᶠᵃᵃ(Ay * v) * ℑyᵃᶠᵃ(u))]
```

which will end up at the location `fcc`.
"""
@inline function div_Uc(i, j, k, grid, advection, U::PrimitiveSolutionLinearizedHeightFields, c)
    return 1/Azᵃᵃᵃ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, advective_tracer_flux_x, advection, U.u, c) +
                                     δyᵃᶜᵃ(i, j, k, grid, advective_tracer_flux_y, advection, U.v, c))
end



