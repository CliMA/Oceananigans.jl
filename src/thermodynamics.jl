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
const u₀ = 0.0               # Reference internal energy [J/kg]

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
    u₀ :: FT
end

function EarthN₂O₂(FT = Float64; T₀ = T₀, p₀ = p₀, s₀ = s₀, u₀ = u₀)
    # Calculate abundance-weighted molar masses and DOFs
    M = (η_N₂_earth * M_N₂ + η_O₂_earth * M_O₂)/(η_N₂_earth + η_O₂_earth)
    dof = (η_N₂_earth * dof_N₂ + η_O₂_earth * dof_O₂)/(η_N₂_earth + η_O₂_earth)
    return IdealGas(FT, M, dof; T₀ = T₀, p₀ = p₀, s₀ = s₀, u₀ = u₀)
end

function IdealGas(FT, M, dof; T₀ = T₀, p₀ = p₀, s₀ = s₀, u₀ = u₀)
    R = R⁰/M
    cᵥ = R*dof/2
    cₚ = cᵥ + R
    ρ₀ = p₀/(R*T₀)
    return IdealGas{FT}(R, cₚ, cᵥ, T₀, p₀, ρ₀, s₀, u₀)
end

####
#### Thermodynamic variables
####

abstract type AbstractThermodynamicVariable end

# struct ModifiedPotentialTemperature <: AbstractThermodynamicVariable end
struct Entropy <: AbstractThermodynamicVariable end
struct Energy  <: AbstractThermodynamicVariable end

###
### Thermodynamic state diagnostics
###
@inline function diagnose_ρs(i, j, k, grid, tvar, tracer_index, gravity, momenta, total_density, densities, tracers)
    @inbounds begin
        T = diagnose_T(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers)
        ρ = tracers[tracer_index].data[i, j, k]
        gas = densities[tracer_index - 1]
        return (ρ > 0.0 ? ρ*(gas.s₀ + gas.cᵥ*log(T/gas.T₀) - gas.R*log(ρ/gas.ρ₀)) : 0.0)
    end
end

@inline function diagnose_T(i, j, k, grid, tvar::Entropy, gravity, momenta, total_density, densities, tracers)
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
            numerator += (ρ > 0 ? (ρ*R*log(ρ/ρ₀) - ρ*s₀) : 0.0)
            denominator += ρ*cᵥ
        end
        return T₀*exp(numerator/denominator)
    end
end

@inline function diagnose_T(i, j, k, grid, tvar::Energy, gravity, momenta, total_density, densities, tracers)
    @inbounds begin
        numerator = tracers.ρe.data[i,j,k]
        denominator = 0.0
        K = kinetic_energy(i, j, k, grid, momenta, total_density)
        Φ = gravity * grid.zC[k]
        for ind_gas in 1:length(densities)
            ρ = tracers[ind_gas + 1].data[i,j,k]
            gas = densities[ind_gas]
            numerator += -ρ*(gas.u₀ + Φ + K - gas.cᵥ*gas.T₀)
            denominator += ρ*gas.cᵥ
        end
        return numerator/denominator
    end
end

@inline function diagnose_p(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers)
    @inbounds begin
        T = diagnose_T(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers)
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

@inline function diagnose_p_over_ρ(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers)
    @inbounds begin
        p = diagnose_p(i, j, k, grid, tvar, gravity, momenta, total_density, densities, tracers)
        ρ = total_density[i, j, k]
        return p/ρ
    end
end
