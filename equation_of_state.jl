#=
Linear equation of state for seawater. Constants taken from Table 1.2 (page 33)
and functional form taken from Eq. (1.57) of Vallis, "Atmospheric and Oceanic
Fluid Dynamics: Fundamentals and Large-Scale Circulation" (2ed, 2017). Note
that a linear equation of state is not accurate enough for serious quantitative
oceanography as the expansion and contraction β coefficients vary with
temperature, pressure, and salinity.
=#

ρ₀ = 1.027e3  # Reference density [kg/m³]
βᵀ = 1.67e-4  # First thermal expansion coefficient [1/K]
βˢ = 0.78e-3  # Haline contraction coefficient [1/ppt]
βᵖ = 4.39e-10 # Compressibility coefficient [ms²/kg]
T₀ = 283      # Reference temperature [K]
S₀ = 35       # Reference salinity [g/kg]
p₀ = 1e5      # Reference pressure [Pa]. Not from Table 1.2 but text itself.
αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].

function ρ(T, S, p)
  return ρ₀ * (1 - βᵀ*(T-T₀) + βˢ*(S-S₀) + βᵖ*(p-p₀))
end
