# Importing packages and functions as needed.
import FFTW
using Statistics: mean

# ### Physical constants.
Ω = 7.2921150e-5  # Rotation rate of the Earth [rad/s].
f = 1e-4  # Nominal value for the Coriolis frequency [rad/s].
g = 9.80665  # Standard acceleration due to gravity [m/s²].

# ### Numerical method parameters.
χ = 0.1  # Adams-Bashforth (AB2) parameter.

# ### Defining model parameters.
NumType = Float64  # Number data type.

Nˣ, Nʸ, Nᶻ = 100, 100, 50  # Number of grid points in (x,y,z).
Lˣ, Lʸ, Lᶻ = 2000, 2000, 1000  # Domain size [m].

Δx, Δy, Δz = Lˣ/Nˣ, Lʸ/Nʸ, Lᶻ/Nᶻ  # Grid spacing [m].
Aˣ, Aʸ, Aᶻ = Δy*Δz, Δx*Δz, Δx*Δy  # Cell face areas [m²].
V = Δx*Δy*Δz  # Volume of a cell [m³].

Nᵗ = 10  # Number of time steps to run for.
Δt = 20  # Time step [s].

# Functions to calculate the x, y, and z-derivatives on an Arakawa C-grid at
# every grid point:
#     δˣ(f) = (f)ᴱ - (f)ᵂ,   δʸ(f) = (f)ᴺ - (f)ˢ,   δᶻ(f) = (f)ᵀ - (f)ᴮ
# where the E, W, N, and S superscripts indicate that the value of f is
# evaluated on the eastern, western, northern, and southern walls of the cell,
# respectively. Similarly, the T and B superscripts indicate the top and bottom
# walls of the cell.
function δˣ(f::Array{NumType, 3}) = (f - cat(f[2:end,:,:], f[1:1,:,:]; dims=1)) / Δx
function δʸ(f::Array{NumType, 3}) = (f - cat(f[:,2:end,:], f[:,1:1,:]; dims=2)) / Δy
function δᶻ(f::Array{NumType, 3}) = (f - cat(f[:,:,2:end], f[:,:,1:1]; dims=3)) / Δz

# Functions to calculate the value of a quantity on a face as the average of
# the quantity in the two cells to which the face is common:
#     ̅qˣ = (qᴱ + qᵂ) / 2,   ̅qʸ = (qᴺ + qˢ) / 2,   ̅qᶻ = (qᵀ + qᴮ) / 2
# where the superscripts are as defined for the derivative operators.
function avgˣ(f::Array{NumType, 3}) = (f + cat(f[:,:,2:end], f[:,:,1:1]; dims=1)) / 2
function avgʸ(f::Array{NumType, 3}) = (f + cat(f[:,2:end,:], f[:,1:1,:]; dims=2)) / 2
function avgᶻ(f::Array{NumType, 3}) = (f + cat(f[:,:,2:end], f[:,:,1:1]; dims=3)) / 2

# Calculate the divergence of a flux of Q with velocity field V = (u,v,w):
# ∇ ⋅ (VQ).
function div_flux(u::Array{NumType, 3}, v::Array{NumType, 3},
  w::Array{NumType, 3}, Q::Array{NumType, 3})
  Vᵘ = V
  div_flux_x = δˣ(Aˣ .* u .* avgˣ(Q))
  div_flux_y = δʸ(Aʸ .* v .* avgʸ(Q))
  div_flux_z = δᶻ(Aᶻ .* w .* avgᶻ(Q))
  return (1/Vᵘ) .* (div_flux_x .+ div_flux_y .+ div_flux_z)
end

# Calculate the nonlinear advection (inertiaL acceleration or convective
# acceleration in other fields) terms ∇ ⋅ (Vu), ∇ ⋅ (Vv), and ∇ ⋅ (Vw) where
# V = (u,v,w). Each component gets its own function for now until we can figure
# out how to combine them all into one function.
function u_dot_u(u::Array{NumType, 3}, v::Array{NumType, 3},
  w::Array{NumType, 3})
  Vᵘ = V
  advection_x = δˣ(avgˣ(Aˣ.*u) .* avgˣ(u))
  advection_y = δʸ(avgˣ(Aʸ.*v) .* avgʸ(u))
  advection_z = δᶻ(avgˣ(Aᶻ.*w) .* avgᶻ(u))
  return (1/Vᵘ) .* (advection_x + advection_y + advection_z)
end

function u_dot_v(u::Array{NumType, 3}, v::Array{NumType, 3},
  w::Array{NumType, 3})
  Vᵘ = V
  advection_x = δˣ(avgʸ(Aˣ.*u) .* avgˣ(v))
  advection_y = δʸ(avgʸ(Aʸ.*v) .* avgʸ(v))
  advection_z = δᶻ(avgʸ(Aᶻ.*w) .* avgᶻ(v))
  return (1/Vᵘ) .* (advection_x + advection_y + advection_z)
end

function u_dot_w(u::Array{NumType, 3}, v::Array{NumType, 3},
  w::Array{NumType, 3})
  Vᵘ = V
  advection_x = δˣ(avgᶻ(Aˣ.*u) .* avgˣ(w))
  advection_y = δʸ(avgᶻ(Aʸ.*v) .* avgʸ(w))
  advection_z = δᶻ(avgᶻ(Aᶻ.*w) .* avgᶻ(w))
  return (1/Vᵘ) .* (advection_x + advection_y + advection_z)
end

# Precalculate wavenumbers and prefactor required to convert Fourier
# coefficients during the spectral pressure solve.
kˣ² = (2 / Δx^2) .* (cos.(π*[1:Nˣ;] / Nˣ) .- 1)
kʸ² = (2 / Δy^2) .* (cos.(π*[1:Nʸ;] / Nʸ) .- 1)
kᶻ² = (2 / Δz^2) .* (cos.(π*[1:Nᶻ;] / Nᶻ) .- 1)

kˣ² = repeat(reshape(kˣ², Nˣ, 1, 1), 1, Nʸ, Nᶻ)
kʸ² = repeat(reshape(kʸ², 1, Nʸ, 1), Nˣ, 1, Nᶻ)
kᶻ² = repeat(reshape(kᶻ², 1, 1, Nᶻ), Nˣ, Nʸ, 1)

prefactor = 1 ./ (kˣ² .+ kʸ² .+ kᶻ²)
prefactor[1,1,1] = 0  # Solvability condition: DC component is zero.

# Solve an elliptic Poisson equation ∇²ϕ = ℱ for the pressure field ϕ where
# ℱ = ∇ ⋅ G and G = (Gᵘ,Gᵛ,Gʷ) using the Fourier-spectral method.
function solve_for_pressure(Gᵘ, Gᵛ, Gʷ)
  RHS = δˣ(Gᵘ) + δʸ(Gᵛ) + δᶻ(Gʷ)
  RHS_hat = FFTW.fft(RHS)
  φ_hat = prefactor .* RHS_hat
  φ = FFTW.ifft(φ_hat)
  return real.(φ)
end

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

# ### Initializing prognostic and diagnostic variable fields.
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

Tⁿ = repeat(reshape(T_ref, 1, 1, 50), Nˣ, Nʸ, 1)

pHY_profile = [-ρ₀*g*h for h in z₀]
pʰʸ = repeat(reshape(pHY_profile, 1, 1, 50), Nˣ, Nʸ, 1)
pⁿ = copy(pʰʸ)

ρⁿ .= ρ.(Tⁿ, Sⁿ, pⁿ)

# Initialize arrays used to store source terms at current and previous
# timesteps, and other variables.
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

Gᵘⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵘⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gʷⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵀⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gˢⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

pⁿʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

for n in 1:Nᵗ
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
  Gᵘⁿ⁺ʰ = (3/2 + χ)*Gᵘⁿ - (1/2 + χ)*Gᵘⁿ⁻¹
  Gᵛⁿ⁺ʰ = (3/2 + χ)*Gᵛⁿ - (1/2 + χ)*Gᵛⁿ⁻¹
  Gʷⁿ⁺ʰ = (3/2 + χ)*Gʷⁿ - (1/2 + χ)*Gʷⁿ⁻¹
  Gᵀⁿ⁺ʰ = (3/2 + χ)*Gᵀⁿ - (1/2 + χ)*Gᵀⁿ⁻¹
  Gˢⁿ⁺ʰ = (3/2 + χ)*Gˢⁿ - (1/2 + χ)*Gˢⁿ⁻¹

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
