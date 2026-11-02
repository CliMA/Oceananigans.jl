using Oceananigans.Fields: ZeroField, OneField
using Oceananigans.Operators: δzᵃᵃᶠ

#####
##### Momentum advection operators
#####

# Alternate names for advective fluxes
@inline _advective_momentum_flux_Uu(i, j, k, grid, scheme, U, u) = advective_momentum_flux_Uu(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), U, u)
@inline _advective_momentum_flux_Vu(i, j, k, grid, scheme, V, u) = advective_momentum_flux_Vu(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), V, u)
@inline _advective_momentum_flux_Wu(i, j, k, grid, scheme, W, u) = advective_momentum_flux_Wu(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), W, u)

@inline _advective_momentum_flux_Uv(i, j, k, grid, scheme, U, v) = advective_momentum_flux_Uv(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), U, v)
@inline _advective_momentum_flux_Vv(i, j, k, grid, scheme, V, v) = advective_momentum_flux_Vv(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), V, v)
@inline _advective_momentum_flux_Wv(i, j, k, grid, scheme, W, v) = advective_momentum_flux_Wv(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), W, v)

@inline _advective_momentum_flux_Uw(i, j, k, grid, scheme, U, w) = advective_momentum_flux_Uw(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), U, w)
@inline _advective_momentum_flux_Vw(i, j, k, grid, scheme, V, w) = advective_momentum_flux_Vw(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), V, w)
@inline _advective_momentum_flux_Ww(i, j, k, grid, scheme, W, w) = advective_momentum_flux_Ww(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), W, w)

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

# Equals the flux of `u - u₀`, since reconstructions are shift-equivariant
@inline anomaly_flux(i, j, k, grid, flux, advection, U, u, u₀) =
    flux(i, j, k, grid, advection, U, u) - u₀ * flux(i, j, k, grid, advection, U, OneField())

"""
$(TYPEDSIGNATURES)

Calculate the advection of momentum in the ``x``-direction as the flux divergence of the anomaly from the local value `u₀ = u[i, j, k]`, ``𝛁⋅(𝐯 (u - u₀))``,

```
1/Vᵘ * [δxᶠᵃᵃ(ℑxᶜᵃᵃ(Ax * u) * ℑxᶜᵃᵃ(u - u₀)) + δy_fca(ℑxᶠᵃᵃ(Ay * v) * ℑyᵃᶠᵃ(u - u₀)) + δz_fac(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(u - u₀))]
```

which ends up at the location `fcc`.
"""
@inline function div_𝐯u(i, j, k, grid, advection, U, u)
    u₀ = @inbounds u[i, j, k]
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, anomaly_flux, _advective_momentum_flux_Uu, advection, U[1], u, u₀) +
                                    δyᵃᶜᵃ(i, j, k, grid, anomaly_flux, _advective_momentum_flux_Vu, advection, U[2], u, u₀) +
                                    δzᵃᵃᶜ(i, j, k, grid, anomaly_flux, _advective_momentum_flux_Wu, advection, U[3], u, u₀))
end

"""
$(TYPEDSIGNATURES)

Calculate the advection of momentum in the ``y``-direction as the flux divergence of the anomaly from the local value `v₀ = v[i, j, k]`, ``𝛁⋅(𝐯 (v - v₀))``,

```
1/Vʸ * [δx_cfa(ℑyᵃᶠᵃ(Ax * u) * ℑxᶠᵃᵃ(v - v₀)) + δyᵃᶠᵃ(ℑyᵃᶜᵃ(Ay * v) * ℑyᵃᶜᵃ(v - v₀)) + δz_afc(ℑyᵃᶠᵃ(Az * w) * ℑzᵃᵃᶠ(v - v₀))]
```

which ends up at the location `cfc`.
"""
@inline function div_𝐯v(i, j, k, grid, advection, U, v)
    v₀ = @inbounds v[i, j, k]
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, anomaly_flux, _advective_momentum_flux_Uv, advection, U[1], v, v₀) +
                                    δyᵃᶠᵃ(i, j, k, grid, anomaly_flux, _advective_momentum_flux_Vv, advection, U[2], v, v₀)    +
                                    δzᵃᵃᶜ(i, j, k, grid, anomaly_flux, _advective_momentum_flux_Wv, advection, U[3], v, v₀))
end

"""
$(TYPEDSIGNATURES)

Calculate the advection of momentum in the ``z``-direction as the flux divergence of the anomaly from the local value `w₀ = w[i, j, k]`, ``𝛁⋅(𝐯 (w - w₀))``,

```
1/Vʷ * [δx_caf(ℑzᵃᵃᶠ(Ax * u) * ℑxᶠᵃᵃ(w - w₀)) + δy_acf(ℑzᵃᵃᶠ(Ay * v) * ℑyᵃᶠᵃ(w - w₀)) + δzᵃᵃᶠ(ℑzᵃᵃᶜ(Az * w) * ℑzᵃᵃᶜ(w - w₀))]
```
which ends up at the location `ccf`.
"""
@inline function div_𝐯w(i, j, k, grid, advection, U, w)
    w₀ = @inbounds w[i, j, k]
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, anomaly_flux, _advective_momentum_flux_Uw, advection, U[1], w, w₀) +
                                    δyᵃᶜᵃ(i, j, k, grid, anomaly_flux, _advective_momentum_flux_Vw, advection, U[2], w, w₀) +
                                    δzᵃᵃᶠ(i, j, k, grid, anomaly_flux, _advective_momentum_flux_Ww, advection, U[3], w, w₀))
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
