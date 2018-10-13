using Statistics: mean

# ### Defining physical constants.
Ω = 7.2921150e-5  # Rotation rate of the Earth [rad/s].
f = 1e-4  # Nominal value for the Coriolis frequency [rad/s].
g = 9.80665  # Standard acceleration due to gravity [m/s²].
αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].

# ### Defining model parameters.
NumType = Float64  # Number data type.

Nˣ, Nʸ, Nᶻ = 100, 100, 50  # Number of grid points in (x,y,z).
Lˣ, Lʸ, Lᶻ = 2000, 2000, 1000  # Domain size [m].
Δx, Δy, Δz = Lˣ/Nˣ, Lʸ/Nʸ, Lᶻ/Nᶻ  # Grid spacing [m].

Nᵗ = 10  # Number of time steps to run for.
Δt = 20  # Time step [s].


@info "Ocean LES model starting up with parameters" Nˣ Nʸ Nᶻ NType

# ### Initializing prognostic and diagnostic variable fields.
u = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in x-direction [m/s].
v = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in y-direction [m/s].
w = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in z-direction [m/s].
θ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Potential temperature [K].
S = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Salinity [g/kg].
ρ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Density [kg/m³].

# ### Parameters for generating initial conditions.
Rᶜ = 600  # Radius of cooling disk [m].
Tₛ = 20  # Surface temperature [°C].
Q₀ = 800  # Cooling disk flux? [W?].
Q₁ = 10  # Noise added to cooling? [W?].
Nₛ = 0 * (f*Rᶜ/Lᶻ)  # Stratification or Brunt–Väisälä frequency [s⁻¹].
Tᶻ = Nₛ^2 / (g*αᵥ)  # Vertical temperature gradient [K/m].

# Coordinates used to generate initial conditions.
x₀ = (1:Nˣ)*Δx
y₀ = (1:Nʸ)*Δy
z₀ = -Δz/2:-Δz:-Lᶻ

# Center horizontal coordinates so that (x,y) = (0,0) corresponds to the center
# of the domain (and the cooling disk).
x₀ = x₀ .- mean(x₀)
y₀ = y₀ .- mean(y₀)

# Calculate vertical temperature profile.
T_ref = Tₛ .+ Tᶻ .* (z₀ .- mean(Tᶻ*z₀))

# Generate surface heat flux field.
Q = Q₀ .+ Q₁ * (0.5 .+ rand(Nˣ, Nʸ))
Qᶜ = zeros(NumType, Nˣ, Nʸ)

# Set surface heat flux to zero outside of cooling disk of radius Rᶜ.
x₀ = repeat(transpose(x₀), Nˣ, 1)
y₀ = repeat(y₀, 1, Nʸ)
r₀ = x₀.*x₀ + y₀.*y₀
Q[findall(r₀ .> Rᶜ^2)] .= 0

function δˣ(f::Array{NumType, 3})
  return f - cat(f[2:end,:,:], f[1:1,:,:]; dims=1)
end

function δʸ(f::Array{NumType, 3})
  return f - cat(f[:,2:end,:], f[:,1:1,:]; dims=2)
end

function δᶻ(f::Array{NumType, 3})
  return f - cat(f[:,:,2:end], f[:,:,1:1]; dims=3)
end
