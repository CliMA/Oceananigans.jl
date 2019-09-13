"""
    LinearEquationOfState(; kwargs...)

Linear equation of state for seawater. Constants taken from Table 1.2 (page 33)
and functional form taken from Eq. (1.57) of Vallis, "Atmospheric and Oceanic
Fluid Dynamics: Fundamentals and Large-Scale Circulation" (2ed, 2017).
"""
struct LinearEquationOfState{T<:AbstractFloat} <: EquationOfState
    ρ₀::T  # Reference density [kg/m³]
    βT::T  # First thermal expansion coefficient [1/K]
    βS::T  # Haline contraction coefficient [1/ppt]
    βp::T  # Compressibility coefficient [ms²/kg]
    T₀::T  # Reference temperature [°C]
    S₀::T  # Reference salinity [g/kg]
    p₀::T  # Reference pressure [Pa].
    cᵥ::T  # Isobaric mass heat capacity [J / kg·K].
    αᵥ::T  # Volumetric coefficient of thermal expansion for water [K⁻¹].
end

function LinearEquationOfState(T=Float64;
    ρ₀ = 1.027e3,
    βT = 1.67e-4,
    βS = 0.78e-3,
    βp = 4.39e-10,
    T₀ = 9.85,
    S₀ = 35,
    p₀ = 1e5,
    cᵥ = 4181.3,
    αᵥ = 2.07e-4)

    LinearEquationOfState{T}(ρ₀, βT, βS, βp, T₀, S₀, p₀, cᵥ, αᵥ)
end

@inline δρ(eos::LinearEquationOfState, T, S, i, j, k) =
    @inbounds -eos.ρ₀ * (eos.βT * (T[i, j, k] - eos.T₀)
                       - eos.βS * (S[i, j, k] - eos.S₀))

@inline buoyancy_perturbation(i, j, k, grid, eos::LinearEquationOfState, grav, T, S) =
    @inbounds eos.βT * (T[i, j, k] - eos.T₀) - eos.βS * (S[i, j, k] - eos.S₀)

NoEquationOfState() = LinearEquationOfState(βT=0, βS=0)
