#####
##### Weighted Essentially Non-Oscillatory (WENO) scheme of order 5
#####

struct WENO5 <: AbstractAdvectionScheme end

#####
##### ENO interpolants
#####

const C = eno_coefficients_matrix(Float64, 3)

@inline px(i, j, k, r, ϕ) = @inbounds sum(C[ℓ+1, r+1] * ϕ[i-r-1+ℓ, j, k] for ℓ in 0:2)
@inline py(i, j, k, r, ϕ) = @inbounds sum(C[ℓ+1, r+1] * ϕ[i, j-r-1+ℓ, k] for ℓ in 0:2)
@inline pz(i, j, k, r, ϕ) = @inbounds sum(C[ℓ+1, r+1] * ϕ[i, j, k-r-1+ℓ] for ℓ in 0:2)

#####
##### Jiang & Shu (1996) WENO smoothness indicators
#####

const B = β_coefficients(Float64, 3)

@inline βx(i, j, k, r, ϕ) = @inbounds sum(B[m+1, n+1, r+1] * ϕ[i-r-1+m, j, k] * ϕ[i-r-1+n, j, k] for m in 0:2 for n in 0:m)
@inline βy(i, j, k, r, ϕ) = @inbounds sum(B[m+1, n+1, r+1] * ϕ[i, j-r-1+m, k] * ϕ[i, j-r-1+n, k] for m in 0:2 for n in 0:m)
@inline βz(i, j, k, r, ϕ) = @inbounds sum(B[m+1, n+1, r+1] * ϕ[i, j, k-r-1+m] * ϕ[i, j, k-r-1+n] for m in 0:2 for n in 0:m)

#####
##### WENO optimal weights
#####

const γ = optimal_weno_weights(Float64, 3)

const C3₀ = 3/10
const C3₁ = 3/5
const C3₂ = 1/10

#####
##### WENO-5 raw weights
#####

const ε = 1e-6
const ƞ = 2  # WENO exponent

@inline αx₀(i, j, k, f) = C3₀ / (βx(i, j, k, 0, f) + ε)^ƞ
@inline αx₁(i, j, k, f) = C3₁ / (βx(i, j, k, 1, f) + ε)^ƞ
@inline αx₂(i, j, k, f) = C3₂ / (βx(i, j, k, 2, f) + ε)^ƞ

@inline αy₀(i, j, k, f) = C3₀ / (βy(i, j, k, 0, f) + ε)^ƞ
@inline αy₁(i, j, k, f) = C3₁ / (βy(i, j, k, 1, f) + ε)^ƞ
@inline αy₂(i, j, k, f) = C3₂ / (βy(i, j, k, 2, f) + ε)^ƞ

@inline αz₀(i, j, k, f) = C3₀ / (βz(i, j, k, 0, f) + ε)^ƞ
@inline αz₁(i, j, k, f) = C3₁ / (βz(i, j, k, 1, f) + ε)^ƞ
@inline αz₂(i, j, k, f) = C3₂ / (βz(i, j, k, 2, f) + ε)^ƞ

#####
##### WENO-5 normalized weights
#####

@inline function weno5_weights_x(i, j, k, f)
    α₀ = αx₀(i, j, k, f)
    α₁ = αx₁(i, j, k, f)
    α₂ = αx₂(i, j, k, f)
    
    Σα = α₀ + α₁ + α₂ 
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    
    return w₀, w₁, w₂
end

@inline function weno5_weights_y(i, j, k, f)
    α₀ = αy₀(i, j, k, f)
    α₁ = αy₁(i, j, k, f)
    α₂ = αy₂(i, j, k, f)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    
    return w₀, w₁, w₂
end

@inline function weno5_weights_z(i, j, k, f)
    α₀ = αz₀(i, j, k, f)
    α₁ = αz₁(i, j, k, f)
    α₂ = αz₂(i, j, k, f)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    
    return w₀, w₁, w₂
end

#####
##### WENO-5 flux reconstruction
#####

@inline function weno5_flux_x(i, j, k, f)
    w₀, w₁, w₂ = weno5_weights_x(i, j, k, f)
    return w₀ * px(i, j, k, 0, f) + w₁ * px(i, j, k, 1, f) + w₂ * px(i, j, k, 2, f)
end

@inline function weno5_flux_y(i, j, k, f)
    w₀, w₁, w₂ = weno5_weights_y(i, j, k, f)
    return w₀ * py(i, j, k, 0, f) + w₁ * py(i, j, k, 1, f) + w₂ * py(i, j, k, 2, f)
end

@inline function weno5_flux_z(i, j, k, f)
    w₀, w₁, w₂ = weno5_weights_z(i, j, k, f)
    return w₀ * pz(i, j, k, 0, f) + w₁ * pz(i, j, k, 1, f) + w₂ * pz(i, j, k, 2, f)
end

#####
##### Momentum advection fluxes
#####

@inline momentum_flux_uu(i, j, k, grid, ::WENO5, u)    = ℑxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno5_flux_x(i, j, k, u)
@inline momentum_flux_uv(i, j, k, grid, ::WENO5, u, v) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno5_flux_y(i, j, k, u)
@inline momentum_flux_uw(i, j, k, grid, ::WENO5, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno5_flux_z(i, j, k, u)

@inline momentum_flux_vu(i, j, k, grid, ::WENO5, u, v) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno5_flux_x(i, j, k, v)
@inline momentum_flux_vv(i, j, k, grid, ::WENO5, v)    = ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno5_flux_y(i, j, k, v)
@inline momentum_flux_vw(i, j, k, grid, ::WENO5, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno5_flux_z(i, j, k, v)

@inline momentum_flux_wu(i, j, k, grid, ::WENO5, u, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno5_flux_x(i, j, k, w)
@inline momentum_flux_wv(i, j, k, grid, ::WENO5, v, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno5_flux_y(i, j, k, w)
@inline momentum_flux_ww(i, j, k, grid, ::WENO5, w)    = ℑzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno5_flux_z(i, j, k, w)

#####
##### Advective tracer fluxes
#####

@inline advective_tracer_flux_x(i, j, k, grid, ::WENO5, u, c) = Ax_ψᵃᵃᶠ(i, j, k, grid, u) * weno5_flux_x(i, j, k, c)
@inline advective_tracer_flux_y(i, j, k, grid, ::WENO5, v, c) = Ay_ψᵃᵃᶠ(i, j, k, grid, v) * weno5_flux_y(i, j, k, c)
@inline advective_tracer_flux_z(i, j, k, grid, ::WENO5, w, c) = Az_ψᵃᵃᵃ(i, j, k, grid, w) * weno5_flux_z(i, j, k, c)

#####
##### Need to advect momentum like tracers
#####

@inline div_ũu(i, j, k, grid, advection::WENO5, U) = div_uc(i, j, k, grid, advection, U, U.u)
@inline div_ũv(i, j, k, grid, advection::WENO5, U) = div_uc(i, j, k, grid, advection, U, U.v)
@inline div_ũw(i, j, k, grid, advection::WENO5, U) = div_uc(i, j, k, grid, advection, U, U.w)
