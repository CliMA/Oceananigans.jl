using JULES.Operators
using JULES.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶠ

####
#### Convinient aliases
####

const IG = IdealGas
const MPT = ModifiedPotentialTemperature

####
#### Exner function
####

@inline Π(i, j, k, grid, pt::MPT, gas::IG, p₀, Θᵐ) = @inbounds (gas.Rᵈ * Θᵐ[i, j, k] / p₀)^(gas.Rᵈ/gas.cᵥ)

####
#### Pressure gradient terms
####

@inline ∂p∂x(i, j, k, grid, pt::MPT, gas::IG, p₀, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, pt, gas, p₀, C.Θᵐ) * ∂xᶠᵃᵃ(i, j, k, grid, C.Θᵐ)
@inline ∂p∂y(i, j, k, grid, pt::MPT, gas::IG, p₀, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, pt, gas, p₀, C.Θᵐ) * ∂yᵃᶠᵃ(i, j, k, grid, C.Θᵐ)
@inline ∂p∂z(i, j, k, grid, pt::MPT, gas::IG, p₀, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, pt, gas, p₀, C.Θᵐ) * ∂zᵃᵃᶠ(i, j, k, grid, C.Θᵐ)

@inline ∂p′∂x(i, j, k, grid, pt, gas, pₛ, C, base_state) = ∂p∂x(i, j, k, grid, pt, gas, pₛ, C) - ∂xᶠᵃᵃ(i, j, k, grid, base_pressure, base_state)
@inline ∂p′∂y(i, j, k, grid, pt, gas, pₛ, C, base_state) = ∂p∂y(i, j, k, grid, pt, gas, pₛ, C) - ∂yᵃᶠᵃ(i, j, k, grid, base_pressure, base_state)
@inline ∂p′∂z(i, j, k, grid, pt, gas, pₛ, C, base_state) = ∂p∂z(i, j, k, grid, pt, gas, pₛ, C) - ∂zᵃᵃᶠ(i, j, k, grid, base_pressure, base_state)
