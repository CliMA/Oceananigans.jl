"""
    LinearEquationOfState{T<:AbstractFloat} <: AbstractEquationOfState

    LinearEquationOfState{T}(; ρ₀, T₀, S₀, βT, βS)

An approximation to the equation of state for seawater using linear functions of
temperature and salinity expanded about a reference state

    ``ρ = ρ(T, S) = ρ₀ + β_T (T - T₀) - β_S (S - S₀)``

where ``ρ₀`` is the reference density [kg/m³], ``T₀`` is the reference temperature [°C],
``S₀`` is the reference salinity [g/kg], ``β_T`` is the thermal expansion coefficient
[1/K], and ``β_S`` is the haline contraction coefficient [1/ppt].

Constants are stored as floating point numbers of type `T`. Default values are taken from
Table 1.2 (page 33) of Vallis, "Atmospheric and Oceanic Fluid Dynamics: Fundamentals and
Large-Scale Circulation" (2ed, 2017).
"""
Base.@kwdef struct LinearEquationOfState{T<:AbstractFloat} <: AbstractEquationOfState
    ρ₀ :: T = 1027.0  # Reference density [kg/m³]
    T₀ :: T = 9.85    # Reference temperature [°C]
    S₀ :: T = 35.0    # Reference salinity [g/kg]
    βT :: T = 1.67e-4 # Thermal expansion coefficient [1/K]
    βS :: T = 0.78e-3 # Haline contraction coefficient [1/ppt]
end

@inline δρ(eos::LinearEquationOfState, T, S, i, j, k) =
    @inbounds -eos.ρ₀ * (eos.βT * (T[i, j, k] - eos.T₀)
                       - eos.βS * (S[i, j, k] - eos.S₀))

@inline buoyancy_perturbation(i, j, k, grid, eos::LinearEquationOfState, grav, T, S) =
    @inbounds eos.βT * (T[i, j, k] - eos.T₀) - eos.βS * (S[i, j, k] - eos.S₀)

"""
    NoEquationOfState()

An equation of state that induces no density perturbations and thus no buoyant forces.
Useful for when you do not want any active tracers such as temperature to affect the flow
or when you want to treat temperature or salinity as passive tracers.
"""
NoEquationOfState() = LinearEquationOfState(βT=0.0, βS=0.0)
