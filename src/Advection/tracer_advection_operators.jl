using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Fields: ZeroField

const ZeroU = NamedTuple{(:u, :v, :w), Tuple{ZeroField, ZeroField, ZeroField}}

@inline div_Uc(i, j, k, grid, advection, ::ZeroU, c) = zero(eltype(grid))
@inline div_Uc(i, j, k, grid, advection, U, ::ZeroField) = zero(eltype(grid))

@inline div_Uc(i, j, k, grid, ::Nothing, U, c) = zero(eltype(grid))
@inline div_Uc(i, j, k, grid, ::Nothing, ::ZeroU, c) = zero(eltype(grid))
@inline div_Uc(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(eltype(grid))

#####
##### Tracer advection operator
#####

"""
    div_uc(i, j, k, grid, advection, U, c)

Calculates the divergence of the flux of a tracer quantity ``c`` being advected by
a velocity field, ``𝛁⋅(𝐯 c)``.
"""
@inline function div_Uc(i, j, k, grid, advection, U, c)
    1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᶜ(i, j, k, grid, advective_tracer_flux_x, advection, U.u, c) +
                             δyᶜᶜᶜ(i, j, k, grid, advective_tracer_flux_y, advection, U.v, c) +
                             δzᶜᶜᶜ(i, j, k, grid, advective_tracer_flux_z, advection, U.w, c))
end

