using JULES.Operators

using Oceananigans.Buoyancy: AbstractEquationOfState

####
#### Universal gas constant and default reference states
####

const atm = 101325.0         # 1 atmosphere in Pa
const R⁰ = 8.31446261815324  # Universal gas constant [J/mol/K]
const T₀ = 273.16            # Reference temperature [K]
const p₀ = 1atm              # Reference pressure [Pa]
const s₀ = 0.0               # Reference entropy [J/kg/K]

####
#### Molar masses and degrees of freedom for common gases
####
const M_N₂ = 28e-3           # Molar mass of diatomic nitrogen (kg/mol)
const dof_N₂ = 5.0           # DOF of diatomic nitrogren (nondim)
const M_O₂ = 32e-3           # Molar mass of diatomic nitrogen (kg/mol)
const dof_O₂ = 5.0           # DOF of diatomic nitrogren (nondim)

####
#### Composition information for common atmospheres
####
const η_N₂_earth = 0.7809    # Molar mixing ratio of N₂ on earth (nondim)
const η_O₂_earth = 0.2095    # Molar mixing ratio of O₂ on earth (nondim)

####
#### Non-condensible ideal gas
####

struct IdealGas{FT} <: AbstractEquationOfState
     R :: FT
    cₚ :: FT
    cᵥ :: FT
    T₀ :: FT
    p₀ :: FT
    ρ₀ :: FT
    s₀ :: FT
end

function EarthN₂O₂(FT = Float64; T₀ = T₀, p₀ = p₀, s₀ = s₀)
    # Calculate abundance-weighted molar masses and DOFs
    M = (η_N₂_earth * M_N₂ + η_O₂_earth * M_O₂)/(η_N₂_earth + η_O₂_earth)
    dof = (η_N₂_earth * dof_N₂ + η_O₂_earth * dof_O₂)/(η_N₂_earth + η_O₂_earth)
    return IdealGas(FT, M, dof; T₀ = T₀, p₀ = p₀, s₀ = s₀)
end

function IdealGas(FT, M, dof; T₀ = T₀, p₀ = p₀, s₀ = s₀)
    R = R⁰/M
    cᵥ = R*dof/2
    cₚ = cᵥ + R
    ρ₀ = p₀/(R*T₀)
    return IdealGas{FT}(R, cₚ, cᵥ, T₀, p₀, ρ₀, s₀)
end

####
#### Thermodynamic variables
####

abstract type AbstractThermodynamicVariable end

# struct ModifiedPotentialTemperature <: AbstractThermodynamicVariable end
struct Entropy <: AbstractThermodynamicVariable end

###
### Thermodynamic state diagnostics
###

@inline function diagnose_s(gas::IdealGas, ρ, T)
    return gas.s₀ + gas.cᵥ*log(T/gas.T₀) - gas.R*log(ρ/gas.ρ₀)
end

@inline function diagnose_T(i, j, k, grid, tvar::Entropy, densities, tracers)
    @inbounds begin
        numerator = tracers.ρs.data[i,j,k]
        denominator = 0.0
        for ind_gas in 1:length(densities)
            ρ = tracers[ind_gas + 1].data[i,j,k]
            gas = densities[ind_gas]
            cᵥ = gas.cᵥ
            R = gas.R
            ρ₀ = gas.ρ₀
            T₀ = gas.T₀
            s₀ = gas.s₀
            numerator += ρ*R*log(ρ/ρ₀) - ρ*s₀
            denominator += ρ*cᵥ
        end
        return T₀*exp(numerator/denominator)
    end
end

@inline function diagnose_p(i, j, k, grid, tvar::Entropy, densities, tracers)
    @inbounds begin
        T = diagnose_T(i, j, k, grid, tvar, densities, tracers)
        p = 0.0
        for ind_gas in 1:length(densities)
            R = densities[ind_gas].R
            ρ = tracers[ind_gas + 1].data[i, j, k]
            p += ρ*R*T
        end
        return p
    end
end

@inline function diagnose_ρ(i, j, k, grid, densities, tracers)
    @inbounds begin
        ρ = 0.0
        for ind_gas in 1:length(densities)
            ρ += tracers[ind_gas + 1].data[i,j,k]
        end
        return ρ
    end
end
