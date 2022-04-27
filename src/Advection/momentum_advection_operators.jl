using Oceananigans.Fields: ZeroField

#####
##### Momentum advection operators
#####

const ZeroU = NamedTuple{(:u, :v, :w), Tuple{ZeroField, ZeroField, ZeroField}}

# Compiler hints
@inline div_𝐯u(i, j, k, grid, advection, ::ZeroU, u) = zero(grid)
@inline div_𝐯v(i, j, k, grid, advection, ::ZeroU, v) = zero(grid)
@inline div_𝐯w(i, j, k, grid, advection, ::ZeroU, w) = zero(grid)

@inline div_𝐯u(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)
@inline div_𝐯v(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)
@inline div_𝐯w(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)

@inline div_𝐯u(i, j, k, grid, ::Nothing, U, u) = zero(grid)
@inline div_𝐯v(i, j, k, grid, ::Nothing, U, v) = zero(grid)
@inline div_𝐯w(i, j, k, grid, ::Nothing, U, w) = zero(grid)

@inline div_𝐯u(i, j, k, grid, ::Nothing, ::ZeroU, u) = zero(grid)
@inline div_𝐯v(i, j, k, grid, ::Nothing, ::ZeroU, v) = zero(grid)
@inline div_𝐯w(i, j, k, grid, ::Nothing, ::ZeroU, w) = zero(grid)

@inline div_𝐯u(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)
@inline div_𝐯v(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)
@inline div_𝐯w(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)

"""
    div_𝐯u(i, j, k, grid, advection, U, u)

Calculate the advection of momentum in the ``x``-direction using the flux form, ``𝛁⋅(𝐯 u)``.
"""
@inline function div_𝐯u(i, j, k, grid, advection, U, u)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᶜᶜ(i, j, k, grid, advective_momentum_flux_Uu, advection, U[1], u) +
                                    δyᶠᶜᶜ(i, j, k, grid, advective_momentum_flux_Vu, advection, U[2], u) +
                                    δzᶠᶜᶜ(i, j, k, grid, advective_momentum_flux_Wu, advection, U[3], u))
end

"""
    div_𝐯v(i, j, k, grid, advection, U, v)

Calculate the advection of momentum in the ``y``-direction using the flux form, ``𝛁⋅(𝐯 v)``.
"""
@inline function div_𝐯v(i, j, k, grid, advection, U, v)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᶠᶜ(i, j, k, grid, advective_momentum_flux_Uv, advection, U[1], v) +
                                    δyᶜᶠᶜ(i, j, k, grid, advective_momentum_flux_Vv, advection, U[2], v)    +
                                    δzᶜᶠᶜ(i, j, k, grid, advective_momentum_flux_Wv, advection, U[3], v))
end

"""
    div_𝐯w(i, j, k, grid, advection, U, w)

Calculate the advection of momentum in the ``z``-direction using the flux form, ``𝛁⋅(𝐯 w)``.
"""
@inline function div_𝐯w(i, j, k, grid, advection, U, w)
    return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᶜᶠ(i, j, k, grid, advective_momentum_flux_Uw, advection, U[1], w) +
                                    δyᶜᶜᶠ(i, j, k, grid, advective_momentum_flux_Vw, advection, U[2], w) +
                                    δzᶜᶜᶠ(i, j, k, grid, advective_momentum_flux_Ww, advection, U[3], w))
end
