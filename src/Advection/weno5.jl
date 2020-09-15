#####
##### Weighted Essentially Non-Oscillatory (WENO) scheme of order 5
#####

struct WENO5 <: AbstractAdvectionScheme end

#####
##### ENO interpolants of size 3
#####

@inline px₀(i, j, k, grid, f) = @inbounds  11/6 * f[i,   j, k] - 7/6 * f[i+1, j, k] + 1/3 * f[i+2, j, k]
@inline px₁(i, j, k, grid, f) = @inbounds   1/3 * f[i-1, j, k] + 5/6 * f[i,   j, k] - 1/6 * f[i+1, j, k]
@inline px₂(i, j, k, grid, f) = @inbounds - 1/6 * f[i-2, j, k] + 5/6 * f[i-1, j, k] + 1/3 * f[i,   j, k]

@inline py₀(i, j, k, grid, f) = @inbounds  11/6 * f[i, j,   k] - 7/6 * f[i, j+1, k] + 1/3 * f[i, j+2, k]
@inline py₁(i, j, k, grid, f) = @inbounds   1/3 * f[i, j-1, k] + 5/6 * f[i, j,   k] - 1/6 * f[i, j+1, k]
@inline py₂(i, j, k, grid, f) = @inbounds - 1/6 * f[i, j-2, k] + 5/6 * f[i, j-1, k] + 1/3 * f[i, j,   k]

@inline pz₀(i, j, k, grid, f) = @inbounds  11/6 * f[i, j,   k] - 7/6 * f[i, j, k+1] + 1/3 * f[i, j, k+2]
@inline pz₁(i, j, k, grid, f) = @inbounds   1/3 * f[i, j, k-1] + 5/6 * f[i, j,   k] - 1/6 * f[i, j, k+1]
@inline pz₂(i, j, k, grid, f) = @inbounds - 1/6 * f[i, j, k-2] + 5/6 * f[i, j, k-1] + 1/3 * f[i, j,   k]

#####
##### Jiang & Shu (1996) WENO smoothness indicators
#####

@inline βx₀(i, j, k, grid, f) = @inbounds 13/12 * (f[i,   j, k] - 2f[i+1, j, k] + f[i+2, j, k])^2 + 1/4 * (3f[i,   j, k] - 4f[i+1, j, k] +  f[i+2, j, k])^2
@inline βx₁(i, j, k, grid, f) = @inbounds 13/12 * (f[i-1, j, k] - 2f[i,   j, k] + f[i+1, j, k])^2 + 1/4 * ( f[i-1, j, k]                 -  f[i+1, j, k])^2
@inline βx₂(i, j, k, grid, f) = @inbounds 13/12 * (f[i-2, j, k] - 2f[i-1, j, k] + f[i,   j, k])^2 + 1/4 * ( f[i-2, j, k] - 4f[i-1, j, k] + 3f[i,   j, k])^2

@inline βy₀(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j,   k] - 2f[i, j+1, k] + f[i, j+2, k])^2 + 1/4 * (3f[i, j,   k] - 4f[i, j+1, k] +  f[i, j+2, k])^2
@inline βy₁(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j-1, k] - 2f[i, j,   k] + f[i, j+1, k])^2 + 1/4 * ( f[i, j-1, k]                 -  f[i, j+1, k])^2
@inline βy₂(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j-2, k] - 2f[i, j-1, k] + f[i, j,   k])^2 + 1/4 * ( f[i, j-2, k] - 4f[i, j-1, k] + 3f[i, j,   k])^2

@inline βz₀(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j,   k] - 2f[i, j, k+1] + f[i, j, k+2])^2 + 1/4 * (3f[i, j,   k] - 4f[i, j, k+1] +  f[i, j, k+2])^2
@inline βz₁(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j, k-1] - 2f[i, j,   k] + f[i, j, k+1])^2 + 1/4 * ( f[i, j, k-1]                 -  f[i, j, k+1])^2
@inline βz₂(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j, k-2] - 2f[i, j, k-1] + f[i, j,   k])^2 + 1/4 * ( f[i, j, k-2] - 4f[i, j, k-1] + 3f[i, j,   k])^2

#####
##### WENO-5 optimal weights
#####

const C3₀ = 3/10
const C3₁ = 3/5
const C3₂ = 1/10

#####
##### WENO-5 raw weights
#####

const ε = 1e-6
const ƞ = 2  # WENO exponent

@inline αx₀(i, j, k, grid, f) = C3₀ / (βx₀(i, j, k, grid, f) + ε)^ƞ
@inline αx₁(i, j, k, grid, f) = C3₁ / (βx₁(i, j, k, grid, f) + ε)^ƞ
@inline αx₂(i, j, k, grid, f) = C3₂ / (βx₂(i, j, k, grid, f) + ε)^ƞ

@inline αy₀(i, j, k, grid, f) = C3₀ / (βy₀(i, j, k, grid, f) + ε)^ƞ
@inline αy₁(i, j, k, grid, f) = C3₁ / (βy₁(i, j, k, grid, f) + ε)^ƞ
@inline αy₂(i, j, k, grid, f) = C3₂ / (βy₂(i, j, k, grid, f) + ε)^ƞ

@inline αz₀(i, j, k, grid, f) = C3₀ / (βz₀(i, j, k, grid, f) + ε)^ƞ
@inline αz₁(i, j, k, grid, f) = C3₁ / (βz₁(i, j, k, grid, f) + ε)^ƞ
@inline αz₂(i, j, k, grid, f) = C3₂ / (βz₂(i, j, k, grid, f) + ε)^ƞ

#####
##### WENO-5 normalized weights
#####

@inline function weno5_weights_x(i, j, k, grid, f)
    α₀ = αx₀(i, j, k, grid, f)
    α₁ = αx₁(i, j, k, grid, f)
    α₂ = αx₂(i, j, k, grid, f)
    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    return w₀, w₁, w₂
end

@inline function weno5_weights_y(i, j, k, grid, f)
    α₀ = αy₀(i, j, k, grid, f)
    α₁ = αy₁(i, j, k, grid, f)
    α₂ = αy₂(i, j, k, grid, f)
    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    return w₀, w₁, w₂
end

@inline function weno5_weights_z(i, j, k, grid, f)
    α₀ = αz₀(i, j, k, grid, f)
    α₁ = αz₁(i, j, k, grid, f)
    α₂ = αz₂(i, j, k, grid, f)
    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    return w₀, w₁, w₂
end

#####
##### WENO-5 flux reconstruction
#####

@inline function weno5_flux_x(i, j, k, grid, f)
    w₀, w₁, w₂ = weno5_weights_x(i, j, k, grid, f)
    return w₀ * px₀(i, j, k, grid, f) + w₁ * px₁(i, j, k, grid, f) + w₂ * px₂(i, j, k, grid, f)
end

@inline function weno5_flux_y(i, j, k, grid, f)
    w₀, w₁, w₂ = weno5_weights_y(i, j, k, grid, f)
    return w₀ * py₀(i, j, k, grid, f) + w₁ * py₁(i, j, k, grid, f) + w₂ * py₂(i, j, k, grid, f)
end

@inline function weno5_flux_z(i, j, k, grid, f)
    w₀, w₁, w₂ = weno5_weights_z(i, j, k, grid, f)
    return w₀ * pz₀(i, j, k, grid, f) + w₁ * pz₁(i, j, k, grid, f) + w₂ * pz₂(i, j, k, grid, f)
end

#####
##### Momentum advection fluxes
#####

@inline momentum_flux_uu(i, j, k, grid, ::WENO5, u)    = ℑxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno5_flux_x(i, j, k, grid, u)
@inline momentum_flux_uv(i, j, k, grid, ::WENO5, u, v) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno5_flux_y(i, j, k, grid, u)
@inline momentum_flux_uw(i, j, k, grid, ::WENO5, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno5_flux_z(i, j, k, grid, u)

@inline momentum_flux_vu(i, j, k, grid, ::WENO5, u, v) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno5_flux_x(i, j, k, grid, v)
@inline momentum_flux_vv(i, j, k, grid, ::WENO5, v)    = ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno5_flux_y(i, j, k, grid, v)
@inline momentum_flux_vw(i, j, k, grid, ::WENO5, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno5_flux_z(i, j, k, grid, v)

@inline momentum_flux_wu(i, j, k, grid, ::WENO5, u, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno5_flux_x(i, j, k, grid, w)
@inline momentum_flux_wv(i, j, k, grid, ::WENO5, v, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno5_flux_y(i, j, k, grid, w)
@inline momentum_flux_ww(i, j, k, grid, ::WENO5, w)    = ℑzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno5_flux_z(i, j, k, grid, w)

#####
##### Advective tracer fluxes
#####

@inline advective_tracer_flux_x(i, j, k, grid, ::WENO5, u, c) = Ax_ψᵃᵃᶠ(i, j, k, grid, u) * weno5_flux_x(i, j, k, grid, c)
@inline advective_tracer_flux_y(i, j, k, grid, ::WENO5, v, c) = Ay_ψᵃᵃᶠ(i, j, k, grid, v) * weno5_flux_y(i, j, k, grid, c)
@inline advective_tracer_flux_z(i, j, k, grid, ::WENO5, w, c) = Az_ψᵃᵃᵃ(i, j, k, grid, w) * weno5_flux_z(i, j, k, grid, c)
