using Oceananigans: AbstractBuoyancy, AbstractEquationOfState

####
#### Some constants for dry and moist gases on Earth
####

# From:
# https://en.wikipedia.org/wiki/Gas_constant
# https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere
# https://en.wikipedia.org/wiki/Heat_capacity_ratio#Real-gas_relations
const gas_constant = 8.31446261815324  # Universal gas constant [J/mol/K]
const molar_mass_dry_air = 28.97e-3    # Molar mass of dry air [kg/mol]
const molar_mass_moist_air = 18.015e-3 # Molar mass of moist air [kg/mol]
const κ_d = 2//7                       # Adiabatic exponent for dry air
const γ_d = 7//5                       # Heat capacity ratio for a diatomic ideal gas

const R_d_air = gas_constant / molar_mass_dry_air    # Specific gas constant for dry air [J/kg/K]
const R_v_air = gas_constant / molar_mass_moist_air  # Specific gas constant for moist air [J/kg/K]
const c_p_d = R_d / κ_d                              # Isobaric specific heat for dry air [J/kg/K]
const c_v_d = c_p_d - R_d                            # Isochoric specific heat for dry air [J/kg/K]

####
#### Dry ideal gas
####

struct DryIdealGas{FT} <: AbstractEquationOfState
    R   :: FT
    c_p :: FT
    c_v :: FT
    κ   :: FT
    γ   :: FT
end

DryIdealGas(FT; R=R_d_air, c_p=c_p_d, c_v=c_v_d, κ=κ_d, γ=γ_d) = DryIdealGas{FT}(R, c_p, c_v, κ, γ)

