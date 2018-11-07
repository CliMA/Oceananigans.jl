using Statistics: mean

# Initializing prognostic and diagnostic variable fields.
uⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in x-direction [m/s].
vⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in y-direction [m/s].
wⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in z-direction [m/s].
Tⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Potential temperature [K].
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

# Calculate vertical temperature profile and convert to Kelvin.
T_ref = 273.15 .+ Tˢ .+ Tᶻ .* (z₀ .- mean(Tᶻ*z₀))

# Generate surface heat flux field.
Q = Q₀ .+ Q₁ * (0.5 .+ rand(Nˣ, Nʸ))

# Set surface heat flux to zero outside of cooling disk of radius Rᶜ.
x₀ = repeat(transpose(x₀), Nˣ, 1)
y₀ = repeat(y₀, 1, Nʸ)
r₀ = x₀.*x₀ + y₀.*y₀
Q[findall(r₀ .> Rᶜ^2)] .= 0

# Convert surface heat flux into 3D forcing term for use when calculating
# source terms at each time step.
Fᵀ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Fᵀ[:, :, 1] = Q

# Zero momentum and salinity forcing term.
Fᵘ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Fᵛ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Fʷ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Fˢ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

# Impose initial conditions.
uⁿ .= 0; vⁿ .= 0; wⁿ .= 0;

# Tⁿ = repeat(reshape(T_ref, 1, 1, 50), Nˣ, Nʸ, 1)

pHY_profile = [-ρ₀*g*h for h in z₀]
pʰʸ = repeat(reshape(pHY_profile, 1, 1, 50), Nˣ, Nʸ, 1)
pⁿ = copy(pʰʸ)

ρⁿ .= ρ.(Tⁿ, Sⁿ, pⁿ)
