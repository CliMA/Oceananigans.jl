using JULES.Operators

using Oceananigans: AbstractBuoyancy, AbstractEquationOfState

####
#### Some constants for dry and moist gases on Earth
####

# From:
# https://en.wikipedia.org/wiki/Gas_constant
# https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere
# https://en.wikipedia.org/wiki/Heat_capacity_ratio#Real-gas_relations

const gas_constant = 8.31446261815324  # Universal gas constant [J/mol/K]

const molar_mass_dry_air     = 28.97e-3  # Molar mass of dry air [kg/mol]
const molar_mass_water_vapor = 18.015e-3 # Molar mass of water vapor [kg/mol]

const κᵈ = 2//7  # Poisson constant (?) for a diatomic ideal gas
const γᵈ = 7//5  # Heat capacity ratio for a diatomic ideal gas

const Rᵈ_air = gas_constant / molar_mass_dry_air      # Specific gas constant for dry air [J/kg/K]
const Rᵛ_air = gas_constant / molar_mass_water_vapor  # Specific gas constant for water vapor [J/kg/K]

const cₚ_dry = Rᵈ_air / κᵈ      # Isobaric specific heat for dry air [J/kg/K]
const cᵥ_dry = cₚ_dry - Rᵈ_air  # Isochoric specific heat for dry air [J/kg/K]

####
#### Dry ideal gas
####

struct IdealGas{FT} <: AbstractEquationOfState
    Rᵈ :: FT
    Rᵛ :: FT
    cₚ :: FT
    cᵥ :: FT
    κ  :: FT
    γ  :: FT
end

IdealGas(FT=Float64; Rᵈ=Rᵈ_air, Rᵛ=Rᵛ_air, cₚ=cₚ_dry, cᵥ=cᵥ_dry, κ=κᵈ, γ=γᵈ) = IdealGas{FT}(Rᵈ, Rᵛ, cₚ, cᵥ, κ, γ)

####
#### Buoyancy term
####

@inline buoyancy_perturbation(i, j, k, grid, grav, ρᵈ, C) = grav * ρᵐ(i, j, k, grid, ρᵈ, C)

