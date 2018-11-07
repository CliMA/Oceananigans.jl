using 
  Seapickle

using Statistics: mean

include("../src/constants.jl")

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

@info begin
  string("Ocean LES model parameters:\n",
         "NumType: $NumType\n",
         "(Nˣ, Nʸ, Nᶻ) = ($Nˣ, $Nʸ, $Nᶻ) [m]\n",
         "(Lˣ, Lʸ, Lᶻ) = ($Lˣ, $Lʸ, $Lᶻ) [m]\n",
         "(Δx, Δy, Δz) = ($Δx, $Δy, $Δz) [m]\n",
         "(Aˣ, Aʸ, Aᶻ) = ($Aˣ, $Aʸ, $Aᶻ) [m²]\n",
         "V = $V [m³]\n",
         "Nᵗ = $Nᵗ [s]\n",
         "Δt = $Δt [s]\n")
end

# Initialize arrays used to store source terms at current and previous
# timesteps, and other variables.
Gᵘⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵛⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gʷⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵀⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gˢⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

Gᵘⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵛⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gʷⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵀⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gˢⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

Gᵘⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵛⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gʷⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵀⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gˢⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

pⁿʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
δρ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

for n in 1:Nᵗ
  global pⁿʰ, δρ
  global uⁿ, vⁿ, wⁿ, Tⁿ, Sⁿ, pⁿ, ρⁿ
  global Gᵘⁿ, Gᵛⁿ, Gʷⁿ, Gᵀⁿ, Gˢⁿ
  global Gᵘⁿ⁻¹, Gᵛⁿ⁻¹, Gʷⁿ⁻¹, Gᵀⁿ⁻¹, Gˢⁿ⁻¹
  global Gᵘⁿ⁺ʰ, Gᵛⁿ⁺ʰ, Gʷⁿ⁺ʰ, Gᵀⁿ⁺ʰ, Gˢⁿ⁺ʰ

  # Calculate new density and density deviation.
  δρ .= ρ.(Tⁿ, Sⁿ, pⁿ) .- ρⁿ
  ρⁿ = ρⁿ + δρ

  # Store source terms from previous iteration.
  Gᵘⁿ⁻¹ = Gᵘⁿ; Gᵛⁿ⁻¹ = Gᵛⁿ; Gʷⁿ⁻¹ = Gʷⁿ; Gᵀⁿ⁻¹ = Gᵀⁿ; Gˢⁿ⁻¹ = Gˢⁿ;

  # Calculate source terms for the current time step.
  Gˢⁿ = -div_flux(uⁿ, vⁿ, wⁿ, Sⁿ) + Fˢ
  Gᵀⁿ = -div_flux(uⁿ, vⁿ, wⁿ, Tⁿ) + Fᵀ

  Gᵘⁿ = -u_dot_u(uⁿ, vⁿ, wⁿ) + f.*vⁿ + Fᵘ
  Gᵛⁿ = -u_dot_v(uⁿ, vⁿ, wⁿ) - f.*uⁿ + Fᵛ
  Gʷⁿ = -u_dot_w(uⁿ, vⁿ, wⁿ) - g.* (δρ ./ ρ₀) + Fʷ

  # Calculate midpoint source terms using the Adams-Bashforth (AB2) method.
  @. begin
    Gᵘⁿ⁺ʰ = (3/2 + χ)*Gᵘⁿ - (1/2 + χ)*Gᵘⁿ⁻¹
    Gᵛⁿ⁺ʰ = (3/2 + χ)*Gᵛⁿ - (1/2 + χ)*Gᵛⁿ⁻¹
    Gʷⁿ⁺ʰ = (3/2 + χ)*Gʷⁿ - (1/2 + χ)*Gʷⁿ⁻¹
    Gᵀⁿ⁺ʰ = (3/2 + χ)*Gᵀⁿ - (1/2 + χ)*Gᵀⁿ⁻¹
    Gˢⁿ⁺ʰ = (3/2 + χ)*Gˢⁿ - (1/2 + χ)*Gˢⁿ⁻¹
  end

  pⁿ = solve_for_pressure(Gᵘⁿ⁺ʰ, Gᵛⁿ⁺ʰ, Gʷⁿ⁺ʰ)

  # Calculate non-hydrostatic component of pressure (as a residual from the
  # total pressure) for vertical velocity time-stepping.
  pⁿʰ = pⁿ .- pʰʸ

  uⁿ = uⁿ .+ (Gᵘⁿ⁺ʰ .- (Aˣ/V).*δˣ(pⁿ)) ./ Δt
  vⁿ = vⁿ .+ (Gᵛⁿ⁺ʰ .- (Aʸ/V).*δʸ(pⁿ)) ./ Δt
  wⁿ = wⁿ .+ (Gʷⁿ⁺ʰ .- (Aᶻ/V).*δᶻ(pⁿʰ)) ./ Δt
  Sⁿ = Sⁿ .+ Gˢⁿ⁺ʰ./Δt
  Tⁿ = Tⁿ .+ Gᵀⁿ⁺ʰ./Δt
end

# TODO: We can use Makie to create 3D OpenGL visualizations to debug the runs.


