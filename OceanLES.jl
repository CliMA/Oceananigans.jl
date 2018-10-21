# Importing functions from packages as needed.
using Statistics: mean

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
βᴾ = 4.39e-10 # Compressibility coefficient [ms²/kg]
T₀ = 283      # Reference temperature [K]
S₀ = 35       # Reference salinity [g/kg]
p₀ = 1e5      # Reference pressure [Pa]. Not from Table 1.2 but convention.
αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].

function ρ(T, S, p)
  return ρ₀ * (1 - βᵀ*(T-T₀) + βˢ*(S-S₀) + βᵖ*(p-p₀))
end

# ### Physical constants.
Ω = 7.2921150e-5  # Rotation rate of the Earth [rad/s].
f = 1e-4  # Nominal value for the Coriolis frequency [rad/s].
g = 9.80665  # Standard acceleration due to gravity [m/s²].

# ### Defining model parameters.
NumType = Float64  # Number data type.

Nˣ, Nʸ, Nᶻ = 100, 100, 50  # Number of grid points in (x,y,z).
Lˣ, Lʸ, Lᶻ = 2000, 2000, 1000  # Domain size [m].

Δx, Δy, Δz = Lˣ/Nˣ, Lʸ/Nʸ, Lᶻ/Nᶻ  # Grid spacing [m].
Aˣ, Aʸ, Aᶻ = Δy*Δz, Δx*Δz, Δx*Δy  # Cell face areas [m²].
V = Δx*Δy*Δz  # Volume of a cell [m³].

Nᵗ = 10  # Number of time steps to run for.
Δt = 20  # Time step [s].

# Defining awesome looking operators.
function δˣ(f::Array{NumType, 3})
  return f - cat(f[2:end,:,:], f[1:1,:,:]; dims=1)
end

function δʸ(f::Array{NumType, 3})
  return f - cat(f[:,2:end,:], f[:,1:1,:]; dims=2)
end

function δᶻ(f::Array{NumType, 3})
  return f - cat(f[:,:,2:end], f[:,:,1:1]; dims=3)
end

function ∇o(f::Array{NumType, 3})
end

function ∇x(f::Array{NumType, 3})
end

@info begin
  string("Ocean LES model key parameters:\n",
         "NumType: $NumType\n",
         "(Nˣ, Nʸ, Nᶻ) = ($Nˣ, $Nʸ, $Nᶻ) [m]\n",
         "(Lˣ, Lʸ, Lᶻ) = ($Lˣ, $Lʸ, $Lᶻ) [m]\n",
         "(Δx, Δy, Δz) = ($Δx, $Δy, $Δz) [m]\n",
         "(Aˣ, Aʸ, Aᶻ) = ($Aˣ, $Aʸ, $Aᶻ) [m²]\n",
         "V = $V [m³]\n",
         "Nᵗ = $Nᵗ [s]\n",
         "Δt = $Δt [s]\n")
end

# ### Initializing prognostic and diagnostic variable fields.
uⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in x-direction [m/s].
vⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in y-direction [m/s].
wⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in z-direction [m/s].
θⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Potential temperature [K].
Sⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Salinity [g/kg].
pⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Pressure [Pa].
ρⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Density [kg/m³].

# ### Parameters for generating initial surface heat flux.
Rᶜ = 600  # Radius of cooling disk [m].
Tˢ = 20  # Surface temperature [°C].
Q₀ = 800  # Cooling disk flux? [W?].
Q₁ = 10  # Noise added to cooling? [W?].
Nˢ = 10 * (f*Rᶜ/Lᶻ)  # Stratification or Brunt–Väisälä frequency [s⁻¹].
Tᶻ = Nˢ^2 / (g*αᵥ)  # Vertical temperature gradient [K/m].

# Coordinates used to generate surface heat flux.
x₀ = (1:Nˣ)*Δx
y₀ = (1:Nʸ)*Δy
z₀ = -Δz/2:-Δz:-Lᶻ

# Center horizontal coordinates so that (x,y) = (0,0) corresponds to the center
# of the domain (and the cooling disk).
x₀ = x₀ .- mean(x₀)
y₀ = y₀ .- mean(y₀)

# Calculate vertical temperature profile.
T_ref = Tˢ .+ Tᶻ .* (z₀ .- mean(Tᶻ*z₀))

# Generate surface heat flux field.
Q = Q₀ .+ Q₁ * (0.5 .+ rand(Nˣ, Nʸ))

# Set surface heat flux to zero outside of cooling disk of radius Rᶜ.
x₀ = repeat(transpose(x₀), Nˣ, 1)
y₀ = repeat(y₀, 1, Nʸ)
r₀ = x₀.*x₀ + y₀.*y₀
Q[findall(r₀ .> Rᶜ^2)] .= 0

Fᵀ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Fᵀ[:, :, 1] = Q

# Impose initial conditions.
uⁿ .= 0; vⁿ .= 0; wⁿ .= 0;

θⁿ = repeat(reshape(T_ref, 1, 1, 50), Nˣ, Nʸ, 1)

pHY = [-ρ₀*g*h for h in z₀]
pⁿ = repeat(reshape(pHY, 1, 1, 50), Nˣ, Nʸ, 1)

ρⁿ = ρ(Tⁿ, Sⁿ, pⁿ)

Gᵘⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵘⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gʷⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵀⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gˢⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

Gᵘⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵘⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gʷⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵀⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gˢⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

for n in 1:Nᵗ
  Gᵘ = (3/2 + χ)*Gᵘⁿ - (1/2 + χ)*Gᵘⁿ⁻¹
  Gᵛ = (3/2 + χ)*Gᵛⁿ - (1/2 + χ)*Gᵛⁿ⁻¹
  Gʷ = (3/2 + χ)*Gʷⁿ - (1/2 + χ)*Gʷⁿ⁻¹

  u = u + (Gᵘ - δˣ(p)) / Δt
  v = v + (Gᵛ - δʸ(p)) / Δt
  w = w + (Gʷ - δᶻ(pⁿʰ)) / Δt
  w = -∫(∇ʰ(u,v,w))
  S = S + Gˢ/Δt
  T = T + Gᵀ/Δt
end
