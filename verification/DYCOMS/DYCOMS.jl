"""
This verification experiment uses large eddy simulation (LES) to represent the turbulent structure of
stratocumulus-topped boundary layer and compares with data from the first research flight (RF01) of the
second Dynamics and Chemistry of Marine Stratocumulus (DYCOMS-II) field study (Stevens et al, 2005).

Stevens et al. (2005): "Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine
    Stratocumulus", Monthly Weather Review 133(6), pp. 1443–62. DOI: https://doi.org/10.1175/MWR2930.1
"""

#####
##### Physical constants
#####

const hPa = 100
const  km = 1000

const pₛ = 1017.8hPa  # Surface pressure [Pa]
const cₚ = 1.015e3    # Isobaric specific heat capacity of dry air [J/kg/K]
const Rᵈ = 287.0      # Gas constant for dry air [J/kg/K]
const Lᵥ = 2.47e6     # Latent heat of vaporization for water [J/kg]

#####
##### Mean state
#####

const zᵢ = 840  # Cloud top or inversion height [m]

# Quasi-two-layer structure in liquid water potential temperature θₗ [K]
function θₗ(x, y, z)
    z <= zᵢ && return 289.0
    z > zᵢ  && return 297.5 + (z - zᵢ)^(1//3)
end

# Quasi-two-layer structure in total-water specific humidity qₜ [g/kg]
function qₜ(x, y, z)
    z <= zᵢ && return 9.0
    z > zᵢ  && return 1.5
end

const Uᵍ  = 7.0      # Zonal geostrophic wind [m/s]
const Vᵍ  = 5.5      # Meridional geostrophic wind [m/s]
const D   = 3.75e-6  # Divergence of the large-scale winds [s⁻¹]
const SST = 292.5    # Sea surface temperature [K]
const Tₛ  = SST-2.1  # Surface air temperature [K]
const Qₛ  = 15       # Surface sensible heat flux [W/m²]
const Qₗ  = 115      # Surface latent heat flux [W/m²]
const ρ₀  = 1.22     # Surface air density [kg/m³]
const ρᵢ  = 1.13     # Air density just below cloud top [kg/m³]

# Bulk aerodynamic drag coefficient
Cᴰ = Cᴴ = Cᴷ = 0.0011

#####
##### Radiative forcing
#####

const ∞ = Inf

const F₀ = 70 # [W/m²]
const F₁ = 22 # [W/m²]
const κ  = 85 # [m²/kg]
const αᶻ = 1  # [m^(-4/3)]

Q(a, b) = κ * ∫(a, b, ρ*rₗ) * dz

F_rad(x, y, z, t) = (  F₀ * exp(-Q(z, ∞))  # Cloud-top cooling
                     + F₁ * exp(-Q(0, z))  # Cloud-base warming
                     + ρᵢ*cₚ*D*αᶻ * ((z-zᵢ)^(4//3) / 4 + zᵢ*(z-zᵢ)^(1//3))) # Cooling in the free troposphere just above cloud top.

#####
##### Model setup
#####

Nx = Ny = 96
Δx = Δy = 35
Lx = Nx*Δx
Ly = Ny*Δy

Δz = 20  # Should probably make it closer to 5 m for a real test.
Lz = 1.5km
Nz = Int(Lz/Δz)

end_time = 4hour


