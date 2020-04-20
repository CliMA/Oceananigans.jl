"""
Implementation of WENO-5 following §5.7 of Durran 2ed, Numerical Methods for Fluid Dynamics.
"""

# WENO-5 interpolation functions (or stencils)

@inline px₀(i, j, f) =  1/3 * f[i-1, j] + 5/6 * f[i, j]   -  1/6 * f[i+1, j]
@inline px₁(i, j, f) = -1/6 * f[i-2, j] + 5/6 * f[i-1, j] +  1/3 * f[i, j]
@inline px₂(i, j, f) =  1/3 * f[i-3, j] - 7/6 * f[i-2, j] + 11/6 * f[i-1, j]

@inline py₀(i, j, f) =  1/3 * f[i, j-1] + 5/6 * f[i, j]   -  1/6 * f[i, j+1]
@inline py₁(i, j, f) = -1/6 * f[i, j-2] + 5/6 * f[i, j-1] +  1/3 * f[i, j]
@inline py₂(i, j, f) =  1/3 * f[i, j-3] - 7/6 * f[i, j-2] + 11/6 * f[i, j-1]

# WENO-5 weight calculation
@inline βx₀(i, j, f) = 13/12 * (f[i-1, j] - 2f[i, j]   + f[i+1, j])^2 + 1/4 * (3f[i-1, j] - 4f[i, j] + f[i+1, j])^2
@inline βx₁(i, j, f) = 13/12 * (f[i-2, j] - 2f[i-1, j] +   f[i, j])^2 + 1/4 * (f[i-2, j]  - f[i, j])^2
@inline βx₂(i, j, f) = 13/12 * (f[i-3, j] - 2f[i-2, j] + f[i-1, j])^2 + 1/4 * (f[i-3, j]  - 4f[i-2, j] + 3f[i-1, j])^2

@inline βy₀(i, j, f) = 13/12 * (f[i, j-1] - 2f[i, j]   + f[i, j+1])^2 + 1/4 * (3f[i, j-1] - 4f[i, j] + f[i, j+1])^2
@inline βy₁(i, j, f) = 13/12 * (f[i, j-2] - 2f[i, j-1] +   f[i, j])^2 + 1/4 * (f[i, j-2]  - f[i, j])^2
@inline βy₂(i, j, f) = 13/12 * (f[i, j-3] - 2f[i, j-2] + f[i, j-1])^2 + 1/4 * (f[i, j-3]  - 4f[i, j-2] + 3f[i, j-1])^2

# WENO-5 (stencil size 3) optimal weights
const C3₀ = 3/10
const C3₁ = 3/5
const C3₂ = 1/10

# WENO-5 raw weights
const ε = 1e-6
const ƞ = 2  # WENO exponent

@inline αx₀(i, j, f) = C3₀ / (βx₀(i, j, f) + ε)^ƞ
@inline αx₁(i, j, f) = C3₁ / (βx₁(i, j, f) + ε)^ƞ
@inline αx₂(i, j, f) = C3₂ / (βx₂(i, j, f) + ε)^ƞ

@inline αy₀(i, j, f) = C3₀ / (βy₀(i, j, f) + ε)^ƞ
@inline αy₁(i, j, f) = C3₁ / (βy₁(i, j, f) + ε)^ƞ
@inline αy₂(i, j, f) = C3₂ / (βy₂(i, j, f) + ε)^ƞ

# WENO-5 normalized weights
function weno5_weights_x(i, j, f)
    a₀ = αx₀(i, j, f)
    a₁ = αx₁(i, j, f)
    a₂ = αx₂(i, j, f)
    Σa = a₀ + a₁ + a₂
    w₀ = a₀ / Σa
    w₁ = a₁ / Σa
    w₂ = a₂ / Σa
    return w₀, w₁, w₂
end

function weno5_weights_y(i, j, f)
    a₀ = αy₀(i, j, f)
    a₁ = αy₁(i, j, f)
    a₂ = αy₂(i, j, f)
    Σa = a₀ + a₁ + a₂
    w₀ = a₀ / Σa
    w₁ = a₁ / Σa
    w₂ = a₂ / Σa
    return w₀, w₁, w₂
end

# WENO-5 flux reconstruction
function weno5_flux_x(i, j, f)
    w₀, w₁, w₂ = weno5_weights_x(i, j, f)
    return w₀ * px₀(i, j, f) + w₁ * px₁(i, j, f) + w₂ * px₂(i, j, f)
end

function weno5_flux_y(i, j, f)
    w₀, w₁, w₂ = weno5_weights_y(i, j, f)
    return w₀ * py₀(i, j, f) + w₁ * py₁(i, j, f) + w₂ * py₂(i, j, f)
end
