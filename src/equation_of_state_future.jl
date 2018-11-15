#=
Linear equation of state for seawater. Constants taken from Table 1.2 (page 33)
and functional form taken from Eq. (1.57) of Vallis, "Atmospheric and Oceanic
Fluid Dynamics: Fundamentals and Large-Scale Circulation" (2ed, 2017). Note
that a linear equation of state is not accurate enough for serious quantitative
oceanography as the expansion and contraction β coefficients vary with
temperature, pressure, and salinity.
=#

struct LinearEquationOfState <: EquationOfStateParameters
    ρ₀::Float64  # Reference density [kg/m³]
    βᵀ::Float64  # First thermal expansion coefficient [1/K]
    βˢ::Float64  # Haline contraction coefficient [1/ppt]
    βᵖ::Float64  # Compressibility coefficient [ms²/kg]
    T₀::Float64  # Reference temperature [K]
    S₀::Float64  # Reference salinity [g/kg]
    p₀::Float64  # Reference pressure [Pa].
end

const cᵥ = 4181.3  # Isobaric mass heat capacity [J / kg·K].

const ρ₀ = 1.027e3
const βᵀ = 1.67e-4
const βˢ = 0.78e-3
const βᵖ = 4.39e-10
const T₀ = 283
const S₀ = 35
const p₀ = 1e5
const αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].

ρ(T, S, p) = ρ₀ * (1 - βᵀ*(T-T₀))
