using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Fields: ZeroField

const ZeroU = NamedTuple{(:u, :v, :w), Tuple{ZeroField, ZeroField, ZeroField}}

@inline div_Uc(i, j, k, grid, advection, ::ZeroU, c) = zero(eltype(grid))
@inline div_Uc(i, j, k, grid, advection, U, ::ZeroField) = zero(eltype(grid))

#####
##### Tracer advection operator
#####

"""
    div_uc(i, j, k, grid, advection, U, c)

Calculates the divergence of the flux of a tracer quantity c being advected by
a velocity field U = (u, v, w), ∇·(Uc),

    1/V * [δxᶜᵃᵃ(Ax * u * ℑxᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * v * ℑyᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * w * ℑzᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function div_Uc(i, j, k, grid, advection, U, c)
    1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, advective_tracer_flux_x, advection, U.u, c) +
                             δyᵃᶜᵃ(i, j, k, grid, advective_tracer_flux_y, advection, U.v, c) +
                             δzᵃᵃᶜ(i, j, k, grid, advective_tracer_flux_z, advection, U.w, c))
end

@inline div_Uc(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, c) where FT = zero(FT)
