# Importing functions from packages as needed.
using Statistics: mean

# Equation of state for seawater.
function ρ(T, S, p)
end

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

# ### Defining physical constants.
Ω = 7.2921150e-5  # Rotation rate of the Earth [rad/s].
f = 1e-4  # Nominal value for the Coriolis frequency [rad/s].
g = 9.80665  # Standard acceleration due to gravity [m/s²].
αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].
ρ₀ = 1025  # Reference density for seawater [kg/m³].

# ### Defining model parameters.
NumType = Float64  # Number data type.

Nˣ, Nʸ, Nᶻ = 100, 100, 50  # Number of grid points in (x,y,z).
Lˣ, Lʸ, Lᶻ = 2000, 2000, 1000  # Domain size [m].

Δx, Δy, Δz = Lˣ/Nˣ, Lʸ/Nʸ, Lᶻ/Nᶻ  # Grid spacing [m].
Aˣ, Aʸ, Aᶻ = Δy*Δz, Δx*Δz, Δx*Δy  # Cell face areas [m²].
V = Δx*Δy*Δz  # Volume of a cell [m³].

Nᵗ = 10  # Number of time steps to run for.
Δt = 20  # Time step [s].

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
u = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in x-direction [m/s].
v = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in y-direction [m/s].
w = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in z-direction [m/s].
θ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Potential temperature [K].
S = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Salinity [g/kg].
p = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Pressure [Pa].
ρ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Density [kg/m³].

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

# Impose initial conditions.
u .= 0; v .= 0; w .= 0;

θ = repeat(reshape(T_ref, 1, 1, 50), Nˣ, Nʸ, 1)

pHY = [-ρ₀*g*h for h in z₀]
p = repeat(reshape(pHY, 1, 1, 50), Nˣ, Nʸ, 1)

# for n in 1:Nᵗ
#   Gᵘ = (3/2 + χ)*Gᵘⁿ - (1/2 + χ)*Gᵘⁿ⁻¹
#   Gᵛ = (3/2 + χ)*Gᵛⁿ - (1/2 + χ)*Gᵛⁿ⁻¹
#   Gʷ = (3/2 + χ)*Gʷⁿ - (1/2 + χ)*Gʷⁿ⁻¹
#
#   u = u + (Gᵘ - δˣ(p)) / Δt
#   v = v + (Gᵛ - δʸ(p)) / Δt
#   w = w + (Gʷ - δᶻ(pⁿʰ)) / Δt
#   w = -∫(∇ʰ(u,v,w))
#   S = S + Gˢ/Δt
#   T = T + Gᵀ/Δt
# end
