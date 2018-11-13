using Statistics: mean, std
using Printf

# import Makie

using PyPlot
PyPlot.pygui(true)

# using Seapickle
# include("../src/Seapickle.jl")

include("../src/operators.jl")
include("../src/equation_of_state.jl")
include("../src/pressure_solve.jl")

# ### Physical constants.
Ω = 7.2921150e-5  # Rotation rate of the Earth [rad/s].
f = 1e-4  # Nominal value for the Coriolis frequency [rad/s].
g = 9.80665  # Standard acceleration due to gravity [m/s²].

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
Q₀ = 800  # Cooling disk heat flux [W/m²].
Q₁ = 10  # Noise added to cooling disk heat flux [W/m²].
Nˢ = 0 * (f*Rᶜ/Lᶻ)  # Stratification or Brunt–Väisälä frequency [s⁻¹].

const αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].
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

# Set surface heat flux to zero outside of cooling disk of radius Rᶜ.
x₀ = repeat(transpose(x₀), Nˣ, 1)
y₀ = repeat(y₀, 1, Nʸ)
r₀² = x₀.*x₀ + y₀.*y₀

# Generate surface heat flux field.
# Q = Q₀ .+ Q₁ * (0.5 .+ rand(Nˣ, Nʸ))

# Cooling disk of radius Rᶜ. Disabling for now as I think the sharp (∞) slope
# at the edge of the disk is causing huge fluxes and we have no flux limiter
# yet.
# Q[findall(r₀² .> Rᶜ^2)] .= 0

# Gaussian cooling disk with similar radius but it much smoother and should work
# without flux limiters.
# Add a little bit of noise but only in the center then impose a Gaussian
# heat flux profile.
Q = Q₁ * (0.5 .+ rand(Nˣ, Nʸ))
Q[findall(r₀² .> Rᶜ^2)] .= 0
@. Q = Q + Q₀ * exp(-r₀^2 / (0.75*Rᶜ^2))

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

Tⁿ = repeat(reshape(T_ref, 1, 1, 50), Nˣ, Nʸ, 1)
const ρ₀ = 1.027e3  # Reference density [kg/m³]
pHY_profile = [-ρ₀*g*h for h in z₀]
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
pⁿʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
g′ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
δρ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

@info string(@sprintf("T⁰[50, 50, 1] = %.4g K\n", Tⁿ[50, 50, 1]))

function time_stepping(uⁿ, vⁿ, wⁿ, Tⁿ, Sⁿ, pⁿ, pʰʸ, pʰʸ′, pⁿʰ, g′, ρⁿ, δρ, Gᵘⁿ, Gᵛⁿ, Gʷⁿ, Gᵀⁿ, Gˢⁿ, Gᵘⁿ⁻¹, Gᵛⁿ⁻¹, Gʷⁿ⁻¹, Gᵀⁿ⁻¹, Gˢⁿ⁻¹, Gᵘⁿ⁺ʰ, Gᵛⁿ⁺ʰ, Gʷⁿ⁺ʰ, Gᵀⁿ⁺ʰ, Gˢⁿ⁺ʰ)
  for n in 1:10

    # Calculate new density and density deviation.
    @. δρ = ρ(Tⁿ, Sⁿ, pⁿ) - ρ₀
    @. ρⁿ = ρⁿ + δρ

    # Calculate the hydrostatic pressure anomaly pʰʸ′ by calculating the
    # effective weight of the water column above at every grid point, i.e. using
    # the reduced gravity g′ = g·δρ/ρ₀. Remember we are assuming the Boussinesq
    # approximation holds.
    @. g′ = g * δρ / ρ₀
    pʰʸ′ = -ρ₀ .* g′ .* repeat(reshape(z₀, 1, 1, Nᶻ), Nˣ, Nʸ, 1)

    # Store source terms from previous iteration.
    Gᵘⁿ⁻¹ = Gᵘⁿ; Gᵛⁿ⁻¹ = Gᵛⁿ; Gʷⁿ⁻¹ = Gʷⁿ; Gᵀⁿ⁻¹ = Gᵀⁿ; Gˢⁿ⁻¹ = Gˢⁿ;

    # Calculate source terms for the current time step.
    Gˢⁿ = -div_flux(uⁿ, vⁿ, wⁿ, Sⁿ) + laplacian_diffusion_zone(Sⁿ) + Fˢ
    Gᵀⁿ = -div_flux(uⁿ, vⁿ, wⁿ, Tⁿ) + laplacian_diffusion_zone(Tⁿ) + Fᵀ

    GTn_div_flux = -div_flux(uⁿ, vⁿ, wⁿ, Tⁿ)
    GTn_lap_diff = laplacian_diffusion_zone(Tⁿ)
    @info begin
      string("Temperature source term:\n",
            @sprintf("div_flux:  mean=%.4g, absmean=%.4g, std=%.4g\n", mean(GTn_div_flux), mean(abs.(GTn_div_flux)), std(GTn_div_flux)),
            @sprintf("lap_diff:  mean=%.4g, absmean=%.4g, std=%.4g\n", mean(GTn_lap_diff), mean(abs.(GTn_lap_diff)), std(GTn_lap_diff)),
            @sprintf("Fᵀ[:,:,1]: mean=%.4g, absmean=%.4g, std=%.4g\n", mean(Fᵀ[:, :, 1]), mean(abs.(Fᵀ[:, :, 1])), std(Fᵀ[:, :, 1]))
            )
    end

    Gᵘⁿ = -u_dot_u(uⁿ, vⁿ, wⁿ) .+ f.*vⁿ .+ laplacian_diffusion_face(uⁿ) .+ Fᵘ
    Gᵛⁿ = -u_dot_v(uⁿ, vⁿ, wⁿ) .- f.*uⁿ .+ laplacian_diffusion_face(vⁿ) .+ Fᵛ

    # Note that I call Gʷⁿ is actually \hat{G}_w from Eq. (43b) of Marshall
    # et al. (1997) so it includes the reduced gravity buoyancy term.
    Gʷⁿ = -u_dot_w(uⁿ, vⁿ, wⁿ) .- avgᶻ(g′) .+ laplacian_diffusion_face(wⁿ) .+ Fʷ

    # Calculate midpoint source terms using the Adams-Bashforth (AB2) method.
    @. begin
      Gᵘⁿ⁺ʰ = (3/2 + χ)*Gᵘⁿ - (1/2 + χ)*Gᵘⁿ⁻¹
      Gᵛⁿ⁺ʰ = (3/2 + χ)*Gᵛⁿ - (1/2 + χ)*Gᵛⁿ⁻¹
      Gʷⁿ⁺ʰ = (3/2 + χ)*Gʷⁿ - (1/2 + χ)*Gʷⁿ⁻¹
      Gᵀⁿ⁺ʰ = (3/2 + χ)*Gᵀⁿ - (1/2 + χ)*Gᵀⁿ⁻¹
      Gˢⁿ⁺ʰ = (3/2 + χ)*Gˢⁿ - (1/2 + χ)*Gˢⁿ⁻¹
    end

    # Calculate non-hydrostatic component of pressure. As we have built in the
    pⁿʰ = solve_for_pressure(Gᵘⁿ⁺ʰ, Gᵛⁿ⁺ʰ, Gʷⁿ⁺ʰ)

    # Calculate the full pressure field.
    @. pⁿ = p₀ + pʰʸ + pʰʸ′ + pⁿʰ

    uⁿ = uⁿ .+ (Gᵘⁿ⁺ʰ .- (Aˣ/V) .* (δˣ(pⁿ) ./ ρ₀)) .* Δt
    vⁿ = vⁿ .+ (Gᵛⁿ⁺ʰ .- (Aʸ/V) .* (δʸ(pⁿ) ./ ρ₀)) .* Δt
    wⁿ = wⁿ .+ (Gʷⁿ⁺ʰ .- (Aᶻ/V) .* (δᶻ(pⁿʰ) ./ ρ₀)) .* Δt
    # wⁿ = - (wⁿ .+ (Gʷⁿ⁺ʰ .- (Aᶻ/V).*δᶻ(pⁿ)) ./ Δt)  # Minus to account for the fact that z increases with depth.
    # wⁿ = wⁿ .+ (Gʷⁿ⁺ʰ .- (Aᶻ/V).*δᶻ(pⁿ)) ./ Δt
    @. Sⁿ = Sⁿ + (Gˢⁿ⁺ʰ * Δt)
    @. Tⁿ = Tⁿ + (Gᵀⁿ⁺ʰ * Δt)

    @info begin
      string("Time: $(n*Δt)\n",
             @sprintf("Tⁿ[50, 50, 1] = %.4g K\n", Tⁿ[50, 50, 1]),
             @sprintf("Tⁿ[50, 50, 2] = %.4g K\n", Tⁿ[50, 50, 2]),
             @sprintf("ΔT[50, 50, 1] = %.4g K\n", Tⁿ[50, 50, 1] - T_ref[1]),
             @sprintf("pʰʸ[1, 1, 1]  = %.4g kPa\n", pʰʸ[1, 1, 1] / 1000),
             @sprintf("pʰʸ[1, 1, 50] = %.4g kPa\n", pʰʸ[1, 1, 50] / 1000),
             @sprintf("uⁿ:   min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(uⁿ), maximum(uⁿ), mean(uⁿ), mean(abs.(uⁿ)), std(uⁿ)),
             @sprintf("vⁿ:   min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(vⁿ), maximum(vⁿ), mean(vⁿ), mean(abs.(vⁿ)), std(vⁿ)),
             @sprintf("wⁿ:   min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(wⁿ), maximum(wⁿ), mean(wⁿ), mean(abs.(wⁿ)), std(wⁿ)),
             @sprintf("Tⁿ:   min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(Tⁿ), maximum(Tⁿ), mean(Tⁿ), mean(abs.(Tⁿ)), std(Tⁿ)),
             @sprintf("Sⁿ:   min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(Sⁿ), maximum(Sⁿ), mean(Sⁿ), mean(abs.(Sⁿ)), std(Sⁿ)),
             @sprintf("pʰʸ:  min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(pʰʸ), maximum(pʰʸ), mean(pʰʸ), mean(abs.(pʰʸ)), std(pʰʸ)),
             @sprintf("pʰʸ′: min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(pʰʸ′), maximum(pʰʸ′), mean(pʰʸ′), mean(abs.(pʰʸ′)), std(pʰʸ′)),
             @sprintf("pⁿʰ:  min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(pⁿʰ), maximum(pⁿʰ), mean(pⁿʰ), mean(abs.(pⁿʰ)), std(pⁿʰ)),
             @sprintf("pⁿ:   min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(pⁿ), maximum(pⁿ), mean(pⁿ), mean(abs.(pⁿ)), std(pⁿ)),
             @sprintf("g′:   min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(g′), maximum(g′), mean(g′), mean(abs.(g′)), std(g′)),
             @sprintf("ρⁿ:   min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(ρⁿ), maximum(ρⁿ), mean(ρⁿ), mean(abs.(ρⁿ)), std(ρⁿ)),
             @sprintf("δρ:   min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(δρ), maximum(δρ), mean(δρ), mean(abs.(δρ)), std(δρ)),
             @sprintf("Gᵘⁿ⁺ʰ: min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(Gᵘⁿ⁺ʰ), maximum(Gᵘⁿ⁺ʰ), mean(Gᵘⁿ⁺ʰ), mean(abs.(Gᵘⁿ⁺ʰ)), std(Gᵘⁿ⁺ʰ)),
             @sprintf("Gᵛⁿ⁺ʰ: min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(Gᵛⁿ⁺ʰ), maximum(Gᵛⁿ⁺ʰ), mean(Gᵛⁿ⁺ʰ), mean(abs.(Gᵛⁿ⁺ʰ)), std(Gᵛⁿ⁺ʰ)),
             @sprintf("Gʷⁿ⁺ʰ: min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(Gʷⁿ⁺ʰ), maximum(Gʷⁿ⁺ʰ), mean(Gʷⁿ⁺ʰ), mean(abs.(Gʷⁿ⁺ʰ)), std(Gʷⁿ⁺ʰ)),
             @sprintf("Gᵀⁿ⁺ʰ: min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(Gᵀⁿ⁺ʰ), maximum(Gᵀⁿ⁺ʰ), mean(Gᵀⁿ⁺ʰ), mean(abs.(Gᵀⁿ⁺ʰ)), std(Gᵀⁿ⁺ʰ)),
             @sprintf("Gˢⁿ⁺ʰ: min=%.4g, max=%.4g, mean=%.4g, absmean=%.4g, std=%.4g\n", minimum(Gˢⁿ⁺ʰ), maximum(Gˢⁿ⁺ʰ), mean(Gˢⁿ⁺ʰ), mean(abs.(Gˢⁿ⁺ʰ)), std(Gˢⁿ⁺ʰ))
            )
    end  # @info
  end  # time stepping for loop

  # Makie.surface(1:Nˣ, 1:Nʸ, reshape(pⁿʰ[:, :, 1:1], (Nˣ, Nʸ)), algorithm = :mip)

  PyPlot.pcolormesh(collect(1:Nˣ), collect(1:Nʸ), reshape(pⁿʰ[:, :, 1:1], (Nˣ, Nʸ)), cmap="seismic")
  # PyPlot.pcolormesh(collect(1:Nˣ), collect(1:Nʸ), reshape(pⁿ[:, :, 25:25], (Nˣ, Nʸ)), cmap="seismic")
  # PyPlot.pcolormesh(collect(1:Nˣ), collect(1:Nᶻ), reshape(pⁿ[1:1, :, :], (Nᶻ, Nˣ)), cmap="seismic")

  PyPlot.colorbar()
end  # time_stepping function

time_stepping(uⁿ, vⁿ, wⁿ, Tⁿ, Sⁿ, pⁿ, pʰʸ, pʰʸ′, pⁿʰ, g′, ρⁿ, δρ, Gᵘⁿ, Gᵛⁿ, Gʷⁿ, Gᵀⁿ, Gˢⁿ, Gᵘⁿ⁻¹, Gᵛⁿ⁻¹, Gʷⁿ⁻¹, Gᵀⁿ⁻¹, Gˢⁿ⁻¹, Gᵘⁿ⁺ʰ, Gᵛⁿ⁺ʰ, Gʷⁿ⁺ʰ, Gᵀⁿ⁺ʰ, Gˢⁿ⁺ʰ)
