using JULES.Operators
using JULES.Operators: ∂xᶠᵃᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶠ

####
#### Convinient aliases
####

# const IG = IdealGas
# const MPT = ModifiedPotentialTemperature

####
#### Pressure gradient ∇p terms for modified potential temperature Θᵐ = ρθᵐ
#### Currentl not supported
####

# Exner function
# @inline Π(i, j, k, grid, gas::IG, Θᵐ) = @inbounds (gas.Rᵈ * Θᵐ[i, j, k] / gas.p₀)^(gas.Rᵈ/gas.cᵥ)

# @inline ∂p∂x(i, j, k, grid, tvar::MPT, gas::IG, ρ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, C.Θᵐ) * ∂xᶠᵃᵃ(i, j, k, grid, C.Θᵐ)
# @inline ∂p∂y(i, j, k, grid, tvar::MPT, gas::IG, ρ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, C.Θᵐ) * ∂yᵃᶠᵃ(i, j, k, grid, C.Θᵐ)
# @inline ∂p∂z(i, j, k, grid, tvar::MPT, gas::IG, ρ, C) = gas.γ * gas.Rᵈ * Π(i, j, k, grid, gas, C.Θᵐ) * ∂zᵃᵃᶠ(i, j, k, grid, C.Θᵐ)

####
#### Pressure gradient ∇p terms for entropy S = ρs
####

@inline ∂p∂x(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers) = ∂xᶠᵃᵃ(i, j, k, grid, diagnose_p, tvar, gravity, momenta, total_density, densities, tracers)
@inline ∂p∂y(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers) = ∂yᵃᶠᵃ(i, j, k, grid, diagnose_p, tvar, gravity, momenta, total_density, densities, tracers)
@inline ∂p∂z(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers) = ∂zᵃᵃᶠ(i, j, k, grid, diagnose_p, tvar, gravity, momenta, total_density, densities, tracers)
