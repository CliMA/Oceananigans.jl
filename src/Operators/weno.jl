"""
Implementation of WENO-5 following §5.7 of Durran 2ed, Numerical Methods for Fluid Dynamics.
"""

# WENO-5 interpolation functions (or stencils)

@inline p₀(i, f) =  1/3 * f[i-1] + 5/6 * f[i]   -  1/6 * f[i+1]
@inline p₁(i, f) = -1/6 * f[i-2] + 5/6 * f[i-1] +  1/3 * f[i]
@inline p₂(i, f) =  1/3 * f[i-3] - 7/6 * f[i-2] + 11/6 * f[i-1]

# WENO-5 weight calculation
@inline β₀(i, f) = 13/12 * (f[i-1] - 2f[i]   + f[i+1])^2 + 1/4 * (3f[i-1] - 4f[i] + f[i+1])^2
@inline β₁(i, f) = 13/12 * (f[i-2] - 2f[i-1] +   f[i])^2 + 1/4 * (f[i-2]  - f[i])^2
@inline β₂(i, f) = 13/12 * (f[i-3] - 2f[i-2] + f[i-1])^2 + 1/4 * (f[i-3]  - 4f[i-2] + 3f[i-1])^2

# WENO-5 (stencil size 3) optimal weights
const C3₀ = 3/10
const C3₁ = 3/5
const C3₂ = 1/10

# WENO-5 raw weights
const ϵ = 1e-6
const ƞ = 2  # WENO exponent

@inline α₀(i, f) = C3₀ / (β₀(i, f) + ϵ)^ƞ
@inline α₁(i, f) = C3₁ / (β₁(i, f) + ϵ)^ƞ
@inline α₂(i, f) = C3₂ / (β₂(i, f) + ϵ)^ƞ

# WENO-5 normalized weights
function weno5_weights(i, f)
    a₀ = α₀(i, f)
    a₁ = α₁(i, f)
    a₂ = α₂(i, f)
    Σa = a₀ + a₁ + a₂
    w₀ = a₀ / Σa
    w₁ = a₁ / Σa
    w₂ = a₂ / Σa
    return w₀, w₁, w₂
end

# WENO-5 flux reconstruction
function weno5_flux(i, f)
    w₀, w₁, w₂ = weno5_weights(i, f)
    return w₀ * p₀(i, f) + w₁ * p₁(i, f) + w₂ * p₂(i, f)
end

####
#### Testing with linear advection equation
####

using Printf, PyPlot

const n = 128
const L = 1
const c = 1

const Δx = L / n
const Δt = 0.1Δx
const M = 5

ϕ₀ = vcat(zeros(16), ones(32), zeros(n-32-16))
ϕ = copy(ϕ₀)
f = copy(ϕ₀)
F = similar(f)

for m in 1:M
    f .= ϕ
    for i in 3:n-3
        F[i] = c * weno5_flux(i, f)
    end
    for i in 3:n-3
        ϕ[i] = ϕ[i] - (Δt/Δx) * (F[i+1] - F[i])
        # ϕ[i] = ϕ[i] - (Δt/Δx) * (c/2) * ((ϕ[i+1] + ϕ[i]) - (ϕ[i] - ϕ[i-1]))
    end
end

x = range(0, 1; length=n)

plot(x, ϕ₀, label="t = 0")
plot(x, ϕ,  label=@sprintf("t = %.4f", M*Δt))
title("WENO-5 advection of a square after $M time steps")
xlabel("x")
ylabel("ϕ")
legend()
