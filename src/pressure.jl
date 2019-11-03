####
#### Convinient aliases
####

const IG = IdealGas
const MPT = ModifiedPotentialTemperature

####
#### Exner function
####

@inline Π(i, j, k, grid, pt::MPT, gas::IG, p₀, Θᵐ) = @inbounds (gas.Rᵈ * Θᵐ[i, j, k] / p₀)^(Rᵈ/cᵥ)
####
#### Pressure gradient terms
####

@inline ∂x_pressure(i, j, k, grid, pt::MPT, gas::IG, p₀, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, pt, gas, C.Θᵐ) * ∂xᶠᵃᵃ(i, j, k, grid, C.Θᵐ)
@inline ∂y_pressure(i, j, k, grid, pt::MPT, gas::IG, p₀, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, pt, gas, C.Θᵐ) * ∂yᵃᶠᵃ(i, j, k, grid, C.Θᵐ)
@inline ∂z_pressure(i, j, k, grid, pt::MPT, gas::IG, p₀, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, pt, gas, C.Θᵐ) * ∂xᵃᵃᶠ(i, j, k, grid, C.Θᵐ)

