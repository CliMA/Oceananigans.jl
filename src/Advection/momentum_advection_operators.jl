using Oceananigans.Fields: ZeroField

#####
##### Momentum advection operators
#####

# Alternate names for advective fluxes
@inline _advective_momentum_flux_Uu(args...) = advective_momentum_flux_Uu(args...)
@inline _advective_momentum_flux_Vu(args...) = advective_momentum_flux_Vu(args...)
@inline _advective_momentum_flux_Wu(args...) = advective_momentum_flux_Wu(args...)

@inline _advective_momentum_flux_Uv(args...) = advective_momentum_flux_Uv(args...)
@inline _advective_momentum_flux_Vv(args...) = advective_momentum_flux_Vv(args...)
@inline _advective_momentum_flux_Wv(args...) = advective_momentum_flux_Wv(args...)

@inline _advective_momentum_flux_Uw(args...) = advective_momentum_flux_Uw(args...)
@inline _advective_momentum_flux_Vw(args...) = advective_momentum_flux_Vw(args...)
@inline _advective_momentum_flux_Ww(args...) = advective_momentum_flux_Ww(args...)

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

Calculate the advection of momentum in the ``x``-direction using the conservative form, ``𝛁⋅(𝐯 u)``,

```
1/Vᵘ * [δxᶠᵃᵃ(ℑxᶜᵃᵃ(Ax * u) * ℑxᶜᵃᵃ(u)) + δy_fca(ℑxᶠᵃᵃ(Ay * v) * ℑyᵃᶠᵃ(u)) + δz_fac(ℑxᶠᵃᵃ(Az * w) * ℑzᵃᵃᶠ(u))]
```

which ends up at the location `fcc`.
"""
@inline function div_𝐯u(i, j, k, grid, advection, U, u)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uu, advection, U[1], u) +
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
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uv, advection, U[1], v) +
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
    return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uw, advection, U[1], w) +
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

# Fallback for `nothing` advection and `ZeroField` tracers and velocities
@inline _advective_momentum_flux_Uu(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Uv(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Uw(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)

@inline _advective_momentum_flux_Vu(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Vv(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Vw(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)

@inline _advective_momentum_flux_Wu(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Wv(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Ww(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)

@inline _advective_momentum_flux_Uu(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Uv(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Uw(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Uu(i, j, k, grid, ::Nothing, ::ZeroField, u) = zero(grid)
@inline _advective_momentum_flux_Uv(i, j, k, grid, ::Nothing, ::ZeroField, v) = zero(grid)
@inline _advective_momentum_flux_Uw(i, j, k, grid, ::Nothing, ::ZeroField, w) = zero(grid)

@inline _advective_momentum_flux_Vu(i, j, k, grid, ::Nothing, V, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Vv(i, j, k, grid, ::Nothing, V, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Vw(i, j, k, grid, ::Nothing, V, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Vu(i, j, k, grid, ::Nothing, ::ZeroField, u) = zero(grid)
@inline _advective_momentum_flux_Vv(i, j, k, grid, ::Nothing, ::ZeroField, v) = zero(grid)
@inline _advective_momentum_flux_Vw(i, j, k, grid, ::Nothing, ::ZeroField, w) = zero(grid)

@inline _advective_momentum_flux_Wu(i, j, k, grid, ::Nothing, W, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Wv(i, j, k, grid, ::Nothing, W, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Ww(i, j, k, grid, ::Nothing, W, ::ZeroField) = zero(grid)
@inline _advective_momentum_flux_Wu(i, j, k, grid, ::Nothing, ::ZeroField, u) = zero(grid)
@inline _advective_momentum_flux_Wv(i, j, k, grid, ::Nothing, ::ZeroField, v) = zero(grid)
@inline _advective_momentum_flux_Ww(i, j, k, grid, ::Nothing, ::ZeroField, w) = zero(grid)

for scheme in (:UpwindBiased, :Centered, :WENO, :FluxFormAdvection)
    @eval begin
        # Fallback for `ZeroField` tracers and velocities
        @inline _advective_momentum_flux_Uu(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Uv(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Uw(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)

        @inline _advective_momentum_flux_Vu(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Vv(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Vw(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)

        @inline _advective_momentum_flux_Wu(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Wv(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Ww(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)

        # Fallback for `ZeroField` tracers
        @inline _advective_momentum_flux_Uu(i, j, k, grid, ::$Scheme, U, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Uv(i, j, k, grid, ::$Scheme, U, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Uw(i, j, k, grid, ::$Scheme, U, ::ZeroField) = zero(grid)

        @inline _advective_momentum_flux_Vu(i, j, k, grid, ::$Scheme, V, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Vv(i, j, k, grid, ::$Scheme, V, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Vw(i, j, k, grid, ::$Scheme, V, ::ZeroField) = zero(grid)

        @inline _advective_momentum_flux_Wu(i, j, k, grid, ::$Scheme, W, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Wv(i, j, k, grid, ::$Scheme, W, ::ZeroField) = zero(grid)
        @inline _advective_momentum_flux_Ww(i, j, k, grid, ::$Scheme, W, ::ZeroField) = zero(grid)

        # Fallback for `ZeroField` velocities
        @inline _advective_momentum_flux_Uu(i, j, k, grid, ::$Scheme, ::ZeroField, u) = zero(grid)
        @inline _advective_momentum_flux_Uv(i, j, k, grid, ::$Scheme, ::ZeroField, v) = zero(grid)
        @inline _advective_momentum_flux_Uw(i, j, k, grid, ::$Scheme, ::ZeroField, w) = zero(grid)

        @inline _advective_momentum_flux_Vu(i, j, k, grid, ::$Scheme, ::ZeroField, u) = zero(grid)
        @inline _advective_momentum_flux_Vv(i, j, k, grid, ::$Scheme, ::ZeroField, v) = zero(grid)
        @inline _advective_momentum_flux_Vw(i, j, k, grid, ::$Scheme, ::ZeroField, w) = zero(grid)

        @inline _advective_momentum_flux_Wu(i, j, k, grid, ::$Scheme, ::ZeroField, u) = zero(grid)
        @inline _advective_momentum_flux_Wv(i, j, k, grid, ::$Scheme, ::ZeroField, v) = zero(grid)
        @inline _advective_momentum_flux_Ww(i, j, k, grid, ::$Scheme, ::ZeroField, w) = zero(grid)
    end
end