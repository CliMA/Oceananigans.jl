using Oceananigans.Fields: ZeroField
using Oceananigans.Operators: δzᵃᵃᶠ

#####
##### Momentum advection operators
#####

# Alternate names for advective fluxes
@inline _advective_momentum_flux_Uu(i, j, k, grid, scheme, U, u) = advective_momentum_flux_Uu(i, j, k, grid, scheme, time_discretization(scheme), U, u)
@inline _advective_momentum_flux_Vu(i, j, k, grid, scheme, V, u) = advective_momentum_flux_Vu(i, j, k, grid, scheme, time_discretization(scheme), V, u)
@inline _advective_momentum_flux_Wu(i, j, k, grid, scheme, W, u) = advective_momentum_flux_Wu(i, j, k, grid, scheme, time_discretization(scheme), W, u)

@inline _advective_momentum_flux_Uv(i, j, k, grid, scheme, U, v) = advective_momentum_flux_Uv(i, j, k, grid, scheme, time_discretization(scheme), U, v)
@inline _advective_momentum_flux_Vv(i, j, k, grid, scheme, V, v) = advective_momentum_flux_Vv(i, j, k, grid, scheme, time_discretization(scheme), V, v)
@inline _advective_momentum_flux_Wv(i, j, k, grid, scheme, W, v) = advective_momentum_flux_Wv(i, j, k, grid, scheme, time_discretization(scheme), W, v)

@inline _advective_momentum_flux_Uw(i, j, k, grid, scheme, U, w) = advective_momentum_flux_Uw(i, j, k, grid, scheme, time_discretization(scheme), U, w)
@inline _advective_momentum_flux_Vw(i, j, k, grid, scheme, V, w) = advective_momentum_flux_Vw(i, j, k, grid, scheme, time_discretization(scheme), V, w)
@inline _advective_momentum_flux_Ww(i, j, k, grid, scheme, W, w) = advective_momentum_flux_Ww(i, j, k, grid, scheme, time_discretization(scheme), W, w)

const ZeroU = NamedTuple{(:u, :v, :w), Tuple{ZeroField, ZeroField, ZeroField}}

# Compiler hints
@inline div_𝐯u(i, j, k, grid, advection, ::ZeroU, u) = zero(grid)
@inline div_𝐯v(i, j, k, grid, advection, ::ZeroU, v) = zero(grid)
@inline div_𝐯w(i, j, k, grid, advection, ::ZeroU, w) = zero(grid)

@inline div_𝐯u(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)
@inline div_𝐯v(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)
@inline div_𝐯w(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)

@inline div_𝐯u(i, j, k, grid, advection, ::ZeroU, ::ZeroField) = zero(grid)
@inline div_𝐯v(i, j, k, grid, advection, ::ZeroU, ::ZeroField) = zero(grid)
@inline div_𝐯w(i, j, k, grid, advection, ::ZeroU, ::ZeroField) = zero(grid)

"""
    div_𝐯u(i, j, k, grid, advection, U, u)

Calculate the advection of momentum in the ``x``-direction using the conservative form, ``𝛁⋅(𝐯 u)``,

```
1/Vᵘ * [δxᶠᵃᵃ(ℑxᶜᵃᵃ(Ax * u) * ℑxᶜᵃᵃ(u)) + δy_fca(ℑxᶠᵃᵃ(Ay * v) * ℑyᵃᶠᵃ(u)) + δz_fac(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(u))]
```

which ends up at the location `fcc`.
"""
@inline function div_𝐯u(i, j, k, grid, advection, U, u)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uu, advection, U[1], u) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vu, advection, U[2], u) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, advection, U[3], u))
end

"""
    div_𝐯v(i, j, k, grid, advection, U, v)

Calculate the advection of momentum in the ``y``-direction using the conservative form, ``𝛁⋅(𝐯 v)``,

```
1/Vʸ * [δx_cfa(ℑyᵃᶠᵃ(Ax * u) * ℑxᶠᵃᵃ(v)) + δyᵃᶠᵃ(ℑyᵃᶜᵃ(Ay * v) * ℑyᵃᶜᵃ(v)) + δz_afc(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(w))]
```

which ends up at the location `cfc`.
"""
@inline function div_𝐯v(i, j, k, grid, advection, U, v)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uv, advection, U[1], v) +
                                    δyᵃᶠᵃ(i, j, k, grid, _advective_momentum_flux_Vv, advection, U[2], v)    +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, advection, U[3], v))
end

"""
    div_𝐯w(i, j, k, grid, advection, U, w)

Calculate the advection of momentum in the ``z``-direction using the conservative form, ``𝛁⋅(𝐯 w)``,

```
1/Vʷ * [δx_caf(ℑzᵃᵃᶠ(Ax * u) * ℑxᶠᵃᵃ(w)) + δy_acf(ℑzᵃᵃᶠ(Ay * v) * ℑyᵃᶠᵃ(w)) + δzᵃᵃᶠ(ℑzᵃᵃᶜ(Az * w) * ℑzᵃᵃᶜ(w))]
```
which ends up at the location `ccf`.
"""
@inline function div_𝐯w(i, j, k, grid, advection, U, w)
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uw, advection, U[1], w) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vw, advection, U[2], w) +
                                    δzᵃᵃᶠ(i, j, k, grid, _advective_momentum_flux_Ww, advection, U[3], w))
end

#####
##### Fallback advection fluxes!
#####

# Fallback for `nothing` advection
@inline _advective_momentum_flux_Uu(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Uv(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Uw(i, j, k, grid, ::Nothing, args...) = zero(grid)

@inline _advective_momentum_flux_Vu(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Vv(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Vw(i, j, k, grid, ::Nothing, args...) = zero(grid)

@inline _advective_momentum_flux_Wu(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Wv(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_momentum_flux_Ww(i, j, k, grid, ::Nothing, args...) = zero(grid)
