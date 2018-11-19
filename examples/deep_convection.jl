using Statistics: mean, std
using Printf

using Interact
# import Makie
using PyPlot
PyPlot.pygui(true)
# using GR

include("../src/operators.jl")
include("../src/equation_of_state.jl")
include("../src/pressure_solve.jl")

# ### Physical constants.
const Ω = 7.2921150e-5  # Rotation rate of the Earth [rad/s].
const f = 1e-4  # Nominal value for the Coriolis frequency [rad/s].
const g = 9.80665  # Standard acceleration due to gravity [m/s²].
const cᵥ = 4181.3  # Isobaric mass heat capacity [J / kg·K].
# const ρ₀ = 1.027e3  # Reference density [kg/m³]

# ### Numerical method parameters.
χ = 0.1  # Adams-Bashforth (AB2) parameter.

# ### Defining model parameters.
const NumType = Float64  # Number data type.

Nˣ, Nʸ, Nᶻ = 100, 100, 50  # Number of grid points in (x,y,z).
Lˣ, Lʸ, Lᶻ = 2000, 2000, 1000  # Domain size [m].

Δx, Δy, Δz = Lˣ/Nˣ, Lʸ/Nʸ, Lᶻ/Nᶻ  # Grid spacing [m].
Aˣ, Aʸ, Aᶻ = Δy*Δz, Δx*Δz, Δx*Δy  # Cell face areas [m²].
V = Δx*Δy*Δz  # Volume of a cell [m³].
M = ρ₀*V  # Mass of water in a cell [kg].

Nᵗ = 10  # Number of time steps to run for.
Δt = 20  # Time step [s].

# List and array of grid coordinates at the centers of the cells.
xC = Δx/2:Δx:Lˣ
yC = Δy/2:Δy:Lʸ
zC = -Δz/2:-Δz:-Lᶻ

xCA = repeat(reshape(xC, Nˣ, 1, 1), 1, Nʸ, Nᶻ)
yCA = repeat(reshape(yC, 1, Nʸ, 1), Nˣ, 1, Nᶻ)
zCA = repeat(reshape(zC, 1, 1, Nᶻ), Nˣ, Nʸ, 1)

# List and array of grid coordinates at the faces of the cells. Note that there
# are Nˣ+1 faces in the ̂x-dimension, Nʸ+1 in the ̂y, and Nᶻ+1 in the ̂z.
xF = 0:Δx:Lˣ
yF = 0:Δy:Lʸ
zF = 0:-Δz:-Lᶻ

xFA = repeat(reshape(xF, Nˣ+1, 1, 1), 1, Nʸ+1, Nᶻ+1)
yFA = repeat(reshape(yF, 1, Nʸ+1, 1), Nˣ+1, 1, Nᶻ+1)
zFA = repeat(reshape(zF, 1, 1, Nᶻ+1), Nˣ+1, Nʸ+1, 1)

# Initializing prognostic and diagnostic variable fields.
uⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in x-direction [m/s].
vⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in y-direction [m/s].
wⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in z-direction [m/s].
Tⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Potential temperature [K].
Sⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Salinity [g/kg].
pʰʸ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ) # Hydrostatic pressure [Pa].
pⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Pressure [Pa].
ρⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Density [kg/m³].

# ### Parameters for generating initial surface heat flux.
Rᶜ = 600  # Radius of cooling disk [m].
Tˢ = 20  # Surface temperature [°C].
Q₀ = -800  # Cooling disk heat flux [W/m²].
Q₁ = 10  # Noise added to cooling disk heat flux [W/m²].
Nˢ = 0 * (f*Rᶜ/Lᶻ)  # Stratification or Brunt–Väisälä frequency [s⁻¹].

const αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].
Tᶻ = Nˢ^2 / (g*αᵥ)  # Vertical temperature gradient [K/m].

# Center horizontal coordinates so that (x,y) = (0,0) corresponds to the center
# of the domain (and the cooling disk).
x₀ = xCA[:, :, 1] .- mean(xCA[:, :, 1])
y₀ = yCA[:, :, 1] .- mean(yCA[:, :, 1])

# Calculate vertical temperature profile and convert to Kelvin.
T_ref = 273.15 .+ Tˢ .+ Tᶻ .* (zC .- mean(Tᶻ*zC))

# Set surface heat flux to zero outside of cooling disk of radius Rᶜ.
# x₀ = xCA[:, :, 1]
# y₀ = yCA[:, :, 1]
r₀² = x₀.*x₀ + y₀.*y₀

# Generate surface heat flux field.
# Cooling disk of radius Rᶜ. Disabling for now as I think the sharp (∞) slope
# at the edge of the disk is causing huge fluxes and we have no flux limiter
# yet.
Q = Q₀ .+ Q₁ * (0.5 .+ rand(Nˣ, Nʸ))
Q[findall(r₀² .> Rᶜ^2)] .= 0

# Gaussian cooling disk with similar radius but it much smoother and should work
# without flux limiters.
# Add a little bit of noise but only in the center then impose a Gaussian
# heat flux profile.
# Q = Q₁ * (0.5 .+ rand(Nˣ, Nʸ))
# Q[findall(r₀² .> Rᶜ^2)] .= 0
# @. Q = Q + Q₀ * exp(-r₀² / (0.75*Rᶜ^2))

# Convert surface heat flux into 3D forcing term for use when calculating
# source terms at each time step.
Fᵀ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

# Converting surface heat flux [W/m²] into a temperature tendency forcing [K/s].
@. Fᵀ[:, :, 1] = (Q / cᵥ) * (Aᶻ / M)

# Zero momentum and salinity forcing term.
Fᵘ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Fᵛ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Fʷ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Fˢ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

# Impose initial conditions.
uⁿ .= 0; vⁿ .= 0; wⁿ .= 0; Sⁿ .= 35;

# Tⁿ = repeat(reshape(T_ref, 1, 1, 50), Nˣ, Nʸ, 1)
Tⁿ .= 283

pHY_profile = [-ρ₀*g*h for h in zC]
pʰʸ = repeat(reshape(pHY_profile, 1, 1, 50), Nˣ, Nʸ, 1)
pⁿ = copy(pʰʸ)  # Initial pressure is just the hydrostatic pressure.

ρⁿ .= ρ.(Tⁿ, Sⁿ, pⁿ)

@info begin
  string("Ocean LES model parameters:\n",
         "NumType: $NumType\n",
         "(Nˣ, Nʸ, Nᶻ) = ($Nˣ, $Nʸ, $Nᶻ) [#]\n",
         "(Lˣ, Lʸ, Lᶻ) = ($Lˣ, $Lʸ, $Lᶻ) [m]\n",
         "(Δx, Δy, Δz) = ($Δx, $Δy, $Δz) [m]\n",
         "(Aˣ, Aʸ, Aᶻ) = ($Aˣ, $Aʸ, $Aᶻ) [m²]\n",
         "V = $V [m³]\n",
         "M = $M [kg]\n",
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

pʰʸ′ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
pⁿʰ⁺ˢ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
g′ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
δρ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

Ru = Array{NumType, 4}(undef, 10, Nˣ, Nʸ, Nᶻ)
Rw = Array{NumType, 4}(undef, 10, Nˣ, Nʸ, Nᶻ)
RT = Array{NumType, 4}(undef, 10, Nˣ, Nʸ, Nᶻ)
RpHY′ = Array{NumType, 4}(undef, 10, Nˣ, Nʸ, Nᶻ)
RpNHS = Array{NumType, 4}(undef, 10, Nˣ, Nʸ, Nᶻ)
RRHS = Array{NumType, 4}(undef, 10, Nˣ, Nʸ, Nᶻ)
RRHS_rec = Array{NumType, 4}(undef, 10, Nˣ, Nʸ, Nᶻ)

@info string(@sprintf("T⁰[50, 50, 1] = %.6g K\n", Tⁿ[50, 50, 1]))

function time_stepping(uⁿ, vⁿ, wⁿ, Tⁿ, Sⁿ, pⁿ, pʰʸ, pʰʸ′, pⁿʰ⁺ˢ, g′, ρⁿ, δρ, Gᵘⁿ, Gᵛⁿ, Gʷⁿ, Gᵀⁿ, Gˢⁿ, Gᵘⁿ⁻¹, Gᵛⁿ⁻¹, Gʷⁿ⁻¹, Gᵀⁿ⁻¹, Gˢⁿ⁻¹, Gᵘⁿ⁺ʰ, Gᵛⁿ⁺ʰ, Gʷⁿ⁺ʰ, Gᵀⁿ⁺ʰ, Gˢⁿ⁺ʰ)
  for n in 1:10

    # Calculate new density and density deviation.
    @. δρ = ρ(Tⁿ, Sⁿ, pⁿ) - ρ₀
    @. ρⁿ = ρ₀ + δρ

    # Calculate the hydrostatic pressure.
    # @. pʰʸ = -ρ₀ * g * zCA

    # Calculate the hydrostatic pressure anomaly pʰʸ′ by calculating the
    # effective weight of the water column above at every grid point, i.e. using
    # the reduced gravity g′ = g·δρ/ρ₀. Remember we are assuming the Boussinesq
    # approximation holds.
    # @. g′ = g * δρ / ρ₀
    # g′ᶻ = avgᶻ(g′)
    δρ̅ᶻ = avgᶻ(δρ)

    for j in 1:Nʸ, i in 1:Nˣ
      pʰʸ′[i, j, 1] = - δρ[i, j, 1] * g * Δz / 2
    end

    for k in 2:Nᶻ, j in 1:Nʸ, i in 1:Nˣ
      pʰʸ′[i, j, k] = pʰʸ′[i, j, k-1] - (δρ̅ᶻ[i, j, k] * g * Δz)
    end

    # Store source terms from previous iteration.
    Gᵘⁿ⁻¹ = Gᵘⁿ; Gᵛⁿ⁻¹ = Gᵛⁿ; Gʷⁿ⁻¹ = Gʷⁿ; Gᵀⁿ⁻¹ = Gᵀⁿ; Gˢⁿ⁻¹ = Gˢⁿ;

    # Calculate source terms for the current time step.
    # Gˢⁿ = -div_flux(uⁿ, vⁿ, wⁿ, Sⁿ) + laplacian_diffusion_zone(Sⁿ) + Fˢ
    # Gᵀⁿ = -div_flux(uⁿ, vⁿ, wⁿ, Tⁿ) + laplacian_diffusion_zone(Tⁿ) + Fᵀ
    Gˢⁿ = laplacian_diffusion_zone(Sⁿ) + Fˢ
    Gᵀⁿ = laplacian_diffusion_zone(Tⁿ) + Fᵀ

    GTn_div_flux = -div_flux(uⁿ, vⁿ, wⁿ, Tⁿ)
    GTn_lap_diff = laplacian_diffusion_zone(Tⁿ)
    @info begin
      string("Temperature source term:\n",
            @sprintf("div_flux:  mean=%.6g, absmean=%.6g, std=%.6g\n", mean(GTn_div_flux), mean(abs.(GTn_div_flux)), std(GTn_div_flux)),
            @sprintf("lap_diff:  mean=%.6g, absmean=%.6g, std=%.6g\n", mean(GTn_lap_diff), mean(abs.(GTn_lap_diff)), std(GTn_lap_diff)),
            @sprintf("Fᵀ[:,:,1]: mean=%.6g, absmean=%.6g, std=%.6g\n", mean(Fᵀ[:, :, 1]), mean(abs.(Fᵀ[:, :, 1])), std(Fᵀ[:, :, 1]))
            )
    end

    # Gᵘⁿ = -u_dot_u(uⁿ, vⁿ, wⁿ) .+ f.*vⁿ .- (Aˣ/V) .* (1/ρ₀).*δˣ(pʰʸ′) .+ laplacian_diffusion_face_h(uⁿ) .+ Fᵘ
    # Gᵛⁿ = -u_dot_v(uⁿ, vⁿ, wⁿ) .- f.*uⁿ .- (Aʸ/V) .* (1/ρ₀).*δʸ(pʰʸ′) .+ laplacian_diffusion_face_h(vⁿ) .+ Fᵛ
    Gᵘⁿ = f.*vⁿ .-  (1/ρ₀) .* (δˣ(pʰʸ′) ./ Δx) .+ laplacian_diffusion_face_h(uⁿ) .+ Fᵘ
    Gᵛⁿ = .- f.*uⁿ .- (1/ρ₀) .* (δʸ(pʰʸ′) ./ Δy) .+ laplacian_diffusion_face_h(vⁿ) .+ Fᵛ

    # Note that I call Gʷⁿ is actually Ĝ_w from Eq. (43b) of Marshall
    # et al. (1997) so it includes the reduced gravity buoyancy term.
    # Gʷⁿ = -u_dot_w(uⁿ, vⁿ, wⁿ) .- (1/ρ₀).*δᶻ(pʰʸ′) .+ laplacian_diffusion_face(wⁿ) .+ Fʷ
    # Gʷⁿ = -u_dot_w(uⁿ, vⁿ, wⁿ) .+ laplacian_diffusion_face_v(wⁿ) .+ Fʷ
    Gʷⁿ = laplacian_diffusion_face_v(wⁿ) .+ Fʷ

    Gwn_u_dot_w = u_dot_w(uⁿ, vⁿ, wⁿ)
    Gwn_lap_diff = laplacian_diffusion_face_v(wⁿ)
    Gwn_Fw = Fʷ
    @info begin
      string("Vertical velocity source term:\n",
            @sprintf("Gwn_u_dot_w: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(Gwn_u_dot_w), maximum(Gwn_u_dot_w), mean(Gwn_u_dot_w), mean(abs.(Gwn_u_dot_w)), std(Gwn_u_dot_w)),
            @sprintf("Gwn_u_dot_w: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(Gwn_lap_diff), maximum(Gwn_lap_diff), mean(Gwn_lap_diff), mean(abs.(Gwn_lap_diff)), std(Gwn_lap_diff)),
            @sprintf("Gwn_u_dot_w: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(Gwn_Fw), maximum(Gwn_Fw), mean(Gwn_Fw), mean(abs.(Gwn_Fw)), std(Gwn_Fw))
            )
    end

    # Calculate midpoint source terms using the Adams-Bashforth (AB2) method.
    # TODO: χ = -1/2 for time step #1. Might matter in other cases.
    @. begin
      Gᵘⁿ⁺ʰ = (3/2 + χ)*Gᵘⁿ - (1/2 + χ)*Gᵘⁿ⁻¹
      Gᵛⁿ⁺ʰ = (3/2 + χ)*Gᵛⁿ - (1/2 + χ)*Gᵛⁿ⁻¹
      Gʷⁿ⁺ʰ = (3/2 + χ)*Gʷⁿ - (1/2 + χ)*Gʷⁿ⁻¹
      Gᵀⁿ⁺ʰ = (3/2 + χ)*Gᵀⁿ - (1/2 + χ)*Gᵀⁿ⁻¹
      Gˢⁿ⁺ʰ = (3/2 + χ)*Gˢⁿ - (1/2 + χ)*Gˢⁿ⁻¹
    end

    # Calculate non-hydrostatic + surface component of pressure. As we have
    # built in the hydrostatic pressure into the Gᵘ source terms, what we get
    # back is the nonhydrostatic
    pⁿʰ⁺ˢ = solve_for_pressure(Gᵘⁿ⁺ʰ, Gᵛⁿ⁺ʰ, Gʷⁿ⁺ʰ)

    RHS = div(Gᵘⁿ⁺ʰ, Gᵛⁿ⁺ʰ, Gʷⁿ⁺ʰ)
    RHS_rec = laplacian(pⁿʰ⁺ˢ) ./ (Δx)^2
    error = RHS_rec .- RHS
    @info begin
      string("Fourier-spectral solver diagnostics:\n",
            @sprintf("RHS:     min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(RHS), maximum(RHS), mean(RHS), mean(abs.(RHS)), std(RHS)),
            @sprintf("RHS_rec: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(RHS_rec), maximum(RHS_rec), mean(RHS_rec), mean(abs.(RHS_rec)), std(RHS_rec)),
            @sprintf("error:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(error), maximum(error), mean(error), mean(abs.(error)), std(error))
            )
    end

    # Calculate the full pressure field.
    # @. pⁿ = p₀ + pʰʸ + pʰʸ′ + pⁿʰ⁺ˢ
    @. pⁿ = pʰʸ′ + pⁿʰ⁺ˢ

    uⁿ = uⁿ .+ (Gᵘⁿ⁺ʰ .- (1/ρ₀) .* (δˣ(pⁿʰ⁺ˢ) ./ Δx)) .* Δt
    vⁿ = vⁿ .+ (Gᵛⁿ⁺ʰ .- (1/ρ₀) .* (δʸ(pⁿʰ⁺ˢ) ./ Δy)) .* Δt
    wⁿ = wⁿ .+ (Gʷⁿ⁺ʰ .- (1/ρ₀) .* (δᶻ(pⁿʰ⁺ˢ) ./ Δz)) .* Δt
    # uⁿ = uⁿ .+ (Gᵘⁿ⁺ʰ .* Δt)
    # vⁿ = vⁿ .+ (Gᵛⁿ⁺ʰ .* Δt)
    # wⁿ = wⁿ .+ (Gʷⁿ⁺ʰ .* Δt)

    @. Sⁿ = Sⁿ + (Gˢⁿ⁺ʰ * Δt)
    @. Tⁿ = Tⁿ + (Gᵀⁿ⁺ʰ * Δt)

    @info begin
      string("Imposing w=0 at top and bottom boundary:\n",
             @sprintf("Before w[:, :, 1]:  min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
              minimum(wⁿ[:, :, 1]), maximum(wⁿ[:, :, 1]), mean(wⁿ[:, :, 1]), mean(abs.(wⁿ[:, :, 1])), std(wⁿ[:, :, 1])),
             @sprintf("Before w[:, :, 50]: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
              minimum(wⁿ[:, :, 50]), maximum(wⁿ[:, :, 50]), mean(wⁿ[:, :,50]), mean(abs.(wⁿ[:, :, 50])), std(wⁿ[:, :, 50])))
    end

    @. wⁿ[:, :, 1]  = 0
    @. wⁿ[:, :, Nᶻ] = 0

    @info begin
      string("Time: $(n*Δt)\n",
             @sprintf("Tⁿ[50, 50, 1] = %.6g K\n", Tⁿ[50, 50, 1]),
             @sprintf("Tⁿ[50, 50, 2] = %.6g K\n", Tⁿ[50, 50, 2]),
             @sprintf("ΔT[50, 50, 1] = %.6g K\n", Tⁿ[50, 50, 1] - 283),
             @sprintf("pʰʸ[1, 1, 1]  = %.6g kPa\n", pʰʸ[1, 1, 1] / 1000),
             @sprintf("pʰʸ[1, 1, 50] = %.6g kPa\n", pʰʸ[1, 1, 50] / 1000),
             @sprintf("uⁿ:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(uⁿ), maximum(uⁿ), mean(uⁿ), mean(abs.(uⁿ)), std(uⁿ)),
             @sprintf("vⁿ:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(vⁿ), maximum(vⁿ), mean(vⁿ), mean(abs.(vⁿ)), std(vⁿ)),
             @sprintf("wⁿ:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(wⁿ), maximum(wⁿ), mean(wⁿ), mean(abs.(wⁿ)), std(wⁿ)),
             @sprintf("Tⁿ:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(Tⁿ), maximum(Tⁿ), mean(Tⁿ), mean(abs.(Tⁿ)), std(Tⁿ)),
             @sprintf("Sⁿ:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(Sⁿ), maximum(Sⁿ), mean(Sⁿ), mean(abs.(Sⁿ)), std(Sⁿ)),
             @sprintf("pʰʸ:  min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(pʰʸ), maximum(pʰʸ), mean(pʰʸ), mean(abs.(pʰʸ)), std(pʰʸ)),
             @sprintf("pʰʸ′: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(pʰʸ′), maximum(pʰʸ′), mean(pʰʸ′), mean(abs.(pʰʸ′)), std(pʰʸ′)),
             @sprintf("pⁿʰ⁺ˢ:min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(pⁿʰ⁺ˢ), maximum(pⁿʰ⁺ˢ), mean(pⁿʰ⁺ˢ), mean(abs.(pⁿʰ⁺ˢ)), std(pⁿʰ⁺ˢ)),
             @sprintf("pⁿ:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(pⁿ), maximum(pⁿ), mean(pⁿ), mean(abs.(pⁿ)), std(pⁿ)),
             # @sprintf("g′:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(g′), maximum(g′), mean(g′), mean(abs.(g′)), std(g′)),
             @sprintf("ρⁿ:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(ρⁿ), maximum(ρⁿ), mean(ρⁿ), mean(abs.(ρⁿ)), std(ρⁿ)),
             @sprintf("δρ:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(δρ), maximum(δρ), mean(δρ), mean(abs.(δρ)), std(δρ)),
             @sprintf("Gᵘⁿ⁺ʰ: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(Gᵘⁿ⁺ʰ), maximum(Gᵘⁿ⁺ʰ), mean(Gᵘⁿ⁺ʰ), mean(abs.(Gᵘⁿ⁺ʰ)), std(Gᵘⁿ⁺ʰ)),
             @sprintf("Gᵛⁿ⁺ʰ: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(Gᵛⁿ⁺ʰ), maximum(Gᵛⁿ⁺ʰ), mean(Gᵛⁿ⁺ʰ), mean(abs.(Gᵛⁿ⁺ʰ)), std(Gᵛⁿ⁺ʰ)),
             @sprintf("Gʷⁿ⁺ʰ: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(Gʷⁿ⁺ʰ), maximum(Gʷⁿ⁺ʰ), mean(Gʷⁿ⁺ʰ), mean(abs.(Gʷⁿ⁺ʰ)), std(Gʷⁿ⁺ʰ)),
             @sprintf("Gᵀⁿ⁺ʰ: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(Gᵀⁿ⁺ʰ), maximum(Gᵀⁿ⁺ʰ), mean(Gᵀⁿ⁺ʰ), mean(abs.(Gᵀⁿ⁺ʰ)), std(Gᵀⁿ⁺ʰ)),
             @sprintf("Gˢⁿ⁺ʰ: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(Gˢⁿ⁺ʰ), maximum(Gˢⁿ⁺ʰ), mean(Gˢⁿ⁺ʰ), mean(abs.(Gˢⁿ⁺ʰ)), std(Gˢⁿ⁺ʰ))
            )
    end  # @info

    Ru[n, :, :, :] = copy(uⁿ)
    Rw[n, :, :, :] = copy(wⁿ)
    RT[n, :, :, :] = copy(Tⁿ)
    RpHY′[n, :, :, :] = copy(pʰʸ′)
    RpNHS[n, :, :, :] = copy(pⁿʰ⁺ˢ)
    RRHS[n, :, :, :] = copy(RHS)
    RRHS_rec[n, :, :, :] = copy(RHS_rec)

  end  # time stepping for loop
end  # time_stepping function

time_stepping(uⁿ, vⁿ, wⁿ, Tⁿ, Sⁿ, pⁿ, pʰʸ, pʰʸ′, pⁿʰ⁺ˢ, g′, ρⁿ, δρ, Gᵘⁿ, Gᵛⁿ, Gʷⁿ, Gᵀⁿ, Gˢⁿ, Gᵘⁿ⁻¹, Gᵛⁿ⁻¹, Gʷⁿ⁻¹, Gᵀⁿ⁻¹, Gˢⁿ⁻¹, Gᵘⁿ⁺ʰ, Gᵛⁿ⁺ʰ, Gʷⁿ⁺ʰ, Gᵀⁿ⁺ʰ, Gˢⁿ⁺ʰ)

# f = PyPlot.figure()
# @manipulate for n in 1:5; withfig(f) do
#     PyPlot.pcolormesh(xC, yC, pⁿʰ[:, :, n], cmap="seismic")
#   end
# end
