#####
##### Implementation of WENO-5 following §5.7 of Durran 2ed, Numerical Methods for Fluid Dynamics.
#####

# WENO-5 interpolation functions (or stencils)

@inline px₀(i, j, k, grid, f) = @inbounds  1/3 * f[i-1, j, k] + 5/6 * f[i,   j, k] -  1/6 * f[i+1, j, k]
@inline px₁(i, j, k, grid, f) = @inbounds -1/6 * f[i-2, j, k] + 5/6 * f[i-1, j, k] +  1/3 * f[i,   j, k]
@inline px₂(i, j, k, grid, f) = @inbounds  1/3 * f[i-3, j, k] - 7/6 * f[i-2, j, k] + 11/6 * f[i-1, j, k]

@inline py₀(i, j, k, grid, f) = @inbounds  1/3 * f[i, j-1, k] + 5/6 * f[i, j,   k] -  1/6 * f[i, j+1, k]
@inline py₁(i, j, k, grid, f) = @inbounds -1/6 * f[i, j-2, k] + 5/6 * f[i, j-1, k] +  1/3 * f[i, j  , k]
@inline py₂(i, j, k, grid, f) = @inbounds  1/3 * f[i, j-3, k] - 7/6 * f[i, j-2, k] + 11/6 * f[i, j-1, k]

@inline pz₀(i, j, k, grid, f) = @inbounds  1/3 * f[i, j, k-1] + 5/6 * f[i, j,   k] -  1/6 * f[i, j, k+1]
@inline pz₁(i, j, k, grid, f) = @inbounds -1/6 * f[i, j, k-2] + 5/6 * f[i, j, k-1] +  1/3 * f[i, j  , k]
@inline pz₂(i, j, k, grid, f) = @inbounds  1/3 * f[i, j, k-3] - 7/6 * f[i, j, k-2] + 11/6 * f[i, j, k-1]

# WENO-5 weight calculation

@inline βx₀(i, j, k, grid, f) = @inbounds 13/12 * (f[i-1, j, k] - 2f[i,   j, k] + f[i+1, j, k])^2 + 1/4 * (3f[i-1, j, k] - 4f[i,   j, k] +  f[i+1, j, k])^2
@inline βx₁(i, j, k, grid, f) = @inbounds 13/12 * (f[i-2, j, k] - 2f[i-1, j, k] + f[i,   j, k])^2 + 1/4 * ( f[i-2, j, k] -  f[i,   j, k])^2
@inline βx₂(i, j, k, grid, f) = @inbounds 13/12 * (f[i-3, j, k] - 2f[i-2, j, k] + f[i-1, j, k])^2 + 1/4 * ( f[i-3, j, k] - 4f[i-2, j, k] + 3f[i-1, j, k])^2

@inline βy₀(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j-1, k] - 2f[i,   j, k] + f[i, j+1, k])^2 + 1/4 * (3f[i, j-1, k] - 4f[i,   j, k] +  f[i, j+1, k])^2
@inline βy₁(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j-2, k] - 2f[i, j-1, k] + f[i,   j, k])^2 + 1/4 * ( f[i, j-2, k] -  f[i,   j, k])^2
@inline βy₂(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j-3, k] - 2f[i, j-2, k] + f[i, j-1, k])^2 + 1/4 * ( f[i, j-3, k] - 4f[i, j-2, k] + 3f[i, j-1, k])^2

@inline βz₀(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j, k-1] - 2f[i,   j, k] + f[i, j, k+1])^2 + 1/4 * (3f[i, j, k-1] - 4f[i,   j, k] +  f[i, j, k+1])^2
@inline βz₁(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j, k-2] - 2f[i, j, k-1] + f[i, j,   k])^2 + 1/4 * ( f[i, j, k-2] -  f[i,   j, k])^2
@inline βz₂(i, j, k, grid, f) = @inbounds 13/12 * (f[i, j, k-3] - 2f[i, j, k-2] + f[i, j, k-1])^2 + 1/4 * ( f[i, j, k-3] - 4f[i, j, k-2] + 3f[i, j, k-1])^2

# WENO-5 (stencil size 3) optimal weights

const C3₀ = 3/10
const C3₁ = 3/5
const C3₂ = 1/10

# WENO-5 raw weights

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

# WENO-5 normalized weights

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

# WENO-5 flux reconstruction

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

struct FirstOrderUpwind <: AbstractAdvectionScheme end
@inline ∂x_advective_flux(i, Δx, u, ϕ, ::FirstOrderUpwind) =
    max(u[i], 0) * (ϕ[i] - ϕ[i-1])/Δx + min(u[i], 0) * (ϕ[i+1] - ϕ[i])/Δx

struct SecondOrderCentered <: AbstractAdvectionScheme end
@inline advective_flux(i, u, ϕ, ::SecondOrderCentered) = u[i] * (ϕ[i-1] + ϕ[i]) / 2
@inline ∂x_advective_flux(i, Δx, u, ϕ, scheme) =
    (advective_flux(i+1, u, ϕ, scheme) - advective_flux(i, u, ϕ, scheme)) / Δx

@inline ∂x_advective_flux(i, Δx, u, ϕ, ::WENO5) = u[i] * (weno5_flux(i+1, ϕ) - weno5_flux(i, ϕ)) / Δx
