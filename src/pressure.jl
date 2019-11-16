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

@inline ∂p∂x(i, j, k, grid, pt::Temperature, gas::IG, pₛ, C) = gas.Rᵈ * ∂xᶠᵃᵃ(i, j, k, grid, C.T)
@inline ∂p∂y(i, j, k, grid, pt::Temperature, gas::IG, pₛ, C) = gas.Rᵈ * ∂yᵃᶠᵃ(i, j, k, grid, C.T)
@inline ∂p∂z(i, j, k, grid, pt::Temperature, gas::IG, pₛ, C) = gas.Rᵈ * ∂zᵃᵃᶠ(i, j, k, grid, C.T)

####
#### Pressure gradient ∇p terms for modified potential temperature Θᵐ = ρθᵐ
####

@inline Π(i, j, k, grid, gas::IG, pₛ, Θᵐ) = @inbounds (gas.Rᵈ * Θᵐ[i, j, k] / pₛ)^(gas.Rᵈ/gas.cᵥ)

@inline ∂p∂x(i, j, k, grid, pt::MPT, gas::IG, pₛ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, pₛ, C.Θᵐ) * ∂xᶠᵃᵃ(i, j, k, grid, C.Θᵐ)
@inline ∂p∂y(i, j, k, grid, pt::MPT, gas::IG, pₛ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, pₛ, C.Θᵐ) * ∂yᵃᶠᵃ(i, j, k, grid, C.Θᵐ)
@inline ∂p∂z(i, j, k, grid, pt::MPT, gas::IG, pₛ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, pₛ, C.Θᵐ) * ∂zᵃᵃᶠ(i, j, k, grid, C.Θᵐ)

