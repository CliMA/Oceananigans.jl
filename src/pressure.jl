using JULES.Operators
using JULES.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶠ

####
#### Convinient aliases
####

const IG = IdealGas
const MPT = ModifiedPotentialTemperature

####
#### Pressure gradient ∇p terms for regular temperature ρT
####

@inline ∂p∂x(i, j, k, grid, pt::Temperature, gas::IG, ρ, C) = gas.Rᵈ * ∂xᶠᵃᵃ(i, j, k, grid, C.T)
@inline ∂p∂y(i, j, k, grid, pt::Temperature, gas::IG, ρ, C) = gas.Rᵈ * ∂yᵃᶠᵃ(i, j, k, grid, C.T)
@inline ∂p∂z(i, j, k, grid, pt::Temperature, gas::IG, ρ, C) = gas.Rᵈ * ∂zᵃᵃᶠ(i, j, k, grid, C.T)

####
#### Pressure gradient ∇p terms for modified potential temperature Θᵐ = ρθᵐ
####

# Exner function
@inline Π(i, j, k, grid, gas::IG, Θᵐ) = @inbounds (gas.Rᵈ * Θᵐ[i, j, k] / gas.p₀)^(gas.Rᵈ/gas.cᵥ)

@inline ∂p∂x(i, j, k, grid, pt::MPT, gas::IG, ρ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, C.Θᵐ) * ∂xᶠᵃᵃ(i, j, k, grid, C.Θᵐ)
@inline ∂p∂y(i, j, k, grid, pt::MPT, gas::IG, ρ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, C.Θᵐ) * ∂yᵃᶠᵃ(i, j, k, grid, C.Θᵐ)
@inline ∂p∂z(i, j, k, grid, pt::MPT, gas::IG, ρ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, C.Θᵐ) * ∂zᵃᵃᶠ(i, j, k, grid, C.Θᵐ)

####
#### Pressure gradient ∇p terms for entropy S = ρs
####

@inline p(i, j, k, grid, pt::Entropy, gas::IG, ρ, C) = @inbounds gas.p₀ * exp( (C.S[i, j, k] - gas.s₀ + gas.cₚ * log(ρ[i, j, k] / gas.ρ₀) ) / gas.cᵥ)

@inline ∂p∂x(i, j, k, grid, pt::Entropy, gas::IG, ρ, C) = ∂xᶠᵃᵃ(i, j, k, grid, p, pt, gas, ρ, C)
@inline ∂p∂y(i, j, k, grid, pt::Entropy, gas::IG, ρ, C) = ∂yᵃᶠᵃ(i, j, k, grid, p, pt, gas, ρ, C)
@inline ∂p∂z(i, j, k, grid, pt::Entropy, gas::IG, ρ, C) = ∂zᵃᵃᶠ(i, j, k, grid, p, pt, gas, ρ, C)

