using JULES.Operators
using JULES.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶠ

####
#### Convinient aliases
####

const IG = IdealGas
const MPT = ModifiedPotentialTemperature

####
#### Pressure gradient ∇p terms for modified potential temperature Θᵐ = ρθᵐ
####

# Exner function
@inline Π(i, j, k, grid, gas::IG, Θᵐ) = @inbounds (gas.Rᵈ * Θᵐ[i, j, k] / gas.p₀)^(gas.Rᵈ/gas.cᵥ)

@inline ∂p∂x(i, j, k, grid, tvar::MPT, gas::IG, ρ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, C.Θᵐ) * ∂xᶠᵃᵃ(i, j, k, grid, C.Θᵐ)
@inline ∂p∂y(i, j, k, grid, tvar::MPT, gas::IG, ρ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, C.Θᵐ) * ∂yᵃᶠᵃ(i, j, k, grid, C.Θᵐ)
@inline ∂p∂z(i, j, k, grid, tvar::MPT, gas::IG, ρ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, C.Θᵐ) * ∂zᵃᵃᶠ(i, j, k, grid, C.Θᵐ)

####
#### Pressure gradient ∇p terms for entropy S = ρs
####

@inline function p(i, j, k, grid, tvar::Entropy, gas::IG, density, tracers)
    @inbounds s = tracers.S[i,j,k]/density[i,j,k]
    @inbounds ρ = density[i,j,k]
    p₀ = gas.p₀
    s₀ = gas.s₀
    cₚ = gas.cₚ
    cᵥ = gas.cᵥ
    ρ₀ = gas.ρ₀
    return p₀ * exp((s - s₀ + cₚ*log(ρ/ρ₀))/cᵥ)
end

@inline ∂p∂x(i, j, k, grid, tvar::Entropy, gas::IG, ρ, C) = ∂xᶠᵃᵃ(i, j, k, grid, p, tvar, gas, ρ, C)
@inline ∂p∂y(i, j, k, grid, tvar::Entropy, gas::IG, ρ, C) = ∂yᵃᶠᵃ(i, j, k, grid, p, tvar, gas, ρ, C)
@inline ∂p∂z(i, j, k, grid, tvar::Entropy, gas::IG, ρ, C) = ∂zᵃᵃᶠ(i, j, k, grid, p, tvar, gas, ρ, C)
