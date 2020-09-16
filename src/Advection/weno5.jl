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
##### WENO raw weights
#####
const γ = optimal_weno_weights(Float64, 3)

const ε = 1e-6  # To avoid dividing by zero
const η = 2     # WENO exponent

@inline αx(i, j, k, r, ϕ) = γ[r+1] / (βx(i, j, k, r, ϕ) + ε)^η
@inline αy(i, j, k, r, ϕ) = γ[r+1] / (βy(i, j, k, r, ϕ) + ε)^η
@inline αz(i, j, k, r, ϕ) = γ[r+1] / (βz(i, j, k, r, ϕ) + ε)^η

#####
##### WENO normalized weights
#####

@inline function weno_weights_x(i, j, k, ϕ)
    α = SVector{3}(αx(i, j, k, r, ϕ) for r in 0:2)
    Σα = sum(α)
    return α ./ Σα
end

@inline function weno_weights_y(i, j, k, ϕ)
    α = SVector{3}(αy(i, j, k, r, ϕ) for r in 0:2)
    Σα = sum(α)
    return α ./ Σα
end

@inline function weno_weights_z(i, j, k, ϕ)
    α = SVector{3}(αz(i, j, k, r, ϕ) for r in 0:2)
    Σα = sum(α)
    return α ./ Σα
end

#####
##### WENO flux reconstruction
#####

@inline function weno_flux_x(i, j, k, ϕ)
    w = weno_weights_x(i, j, k, ϕ)
    return sum(w[r+1] * px(i, j, k, r, ϕ) for r in 0:2)
end

@inline function weno_flux_y(i, j, k, ϕ)
    w = weno_weights_y(i, j, k, ϕ)
    return sum(w[r+1] * py(i, j, k, r, ϕ) for r in 0:2)
end

@inline function weno_flux_z(i, j, k, ϕ)
    w = weno_weights_z(i, j, k, ϕ)
    return sum(w[r+1] * pz(i, j, k, r, ϕ) for r in 0:2)
end

#####
##### Momentum advection fluxes
#####

@inline momentum_flux_uu(i, j, k, grid, ::WENO5, u)    = ℑxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno_flux_x(i, j, k, u)
@inline momentum_flux_uv(i, j, k, grid, ::WENO5, u, v) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno_flux_y(i, j, k, u)
@inline momentum_flux_uw(i, j, k, grid, ::WENO5, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno_flux_z(i, j, k, u)

@inline momentum_flux_vu(i, j, k, grid, ::WENO5, u, v) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno_flux_x(i, j, k, v)
@inline momentum_flux_vv(i, j, k, grid, ::WENO5, v)    = ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno_flux_y(i, j, k, v)
@inline momentum_flux_vw(i, j, k, grid, ::WENO5, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno_flux_z(i, j, k, v)

@inline momentum_flux_wu(i, j, k, grid, ::WENO5, u, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno_flux_x(i, j, k, w)
@inline momentum_flux_wv(i, j, k, grid, ::WENO5, v, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno_flux_y(i, j, k, w)
@inline momentum_flux_ww(i, j, k, grid, ::WENO5, w)    = ℑzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno_flux_z(i, j, k, w)

#####
##### Advective tracer fluxes
#####

@inline advective_tracer_flux_x(i, j, k, grid, ::WENO5, u, c) = Ax_ψᵃᵃᶠ(i, j, k, grid, u) * weno_flux_x(i, j, k, c)
@inline advective_tracer_flux_y(i, j, k, grid, ::WENO5, v, c) = Ay_ψᵃᵃᶠ(i, j, k, grid, v) * weno_flux_y(i, j, k, c)
@inline advective_tracer_flux_z(i, j, k, grid, ::WENO5, w, c) = Az_ψᵃᵃᵃ(i, j, k, grid, w) * weno_flux_z(i, j, k, c)

#####
##### Need to advect momentum like tracers
#####

@inline div_ũu(i, j, k, grid, advection::WENO5, U) = div_uc(i, j, k, grid, advection, U, U.u)
@inline div_ũv(i, j, k, grid, advection::WENO5, U) = div_uc(i, j, k, grid, advection, U, U.v)
@inline div_ũw(i, j, k, grid, advection::WENO5, U) = div_uc(i, j, k, grid, advection, U, U.w)
