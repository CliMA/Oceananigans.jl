"""
    LinearEquationOfState

Linear equation of state for seawater. Constants taken from Table 1.2 (page 33)
and functional form taken from Eq. (1.57) of Vallis, "Atmospheric and Oceanic
Fluid Dynamics: Fundamentals and Large-Scale Circulation" (2ed, 2017).
"""
Base.@kwdef struct LinearEquationOfState{T<:AbstractFloat} <: AbstractEquationOfState
    ρ₀ :: T = 1027.0  # Reference density [kg/m³]
    T₀ :: T = 9.85    # Reference temperature [°C]
    S₀ :: T = 35.0    # Reference salinity [g/kg]
    βT :: T = 1.67e-4 # First thermal expansion coefficient [1/K]
    βS :: T = 0.78e-3 # Haline contraction coefficient [1/ppt]
end

@inline δρ(eos::LinearEquationOfState, T, S, i, j, k) =
    @inbounds -eos.ρ₀ * (eos.βT * (T[i, j, k] - eos.T₀)
                       - eos.βS * (S[i, j, k] - eos.S₀))

@inline buoyancy_perturbation(i, j, k, grid, eos::LinearEquationOfState, grav, T, S) =
    @inbounds eos.βT * (T[i, j, k] - eos.T₀) - eos.βS * (S[i, j, k] - eos.S₀)

NoEquationOfState() = LinearEquationOfState(βT=0.0, βS=0.0)
