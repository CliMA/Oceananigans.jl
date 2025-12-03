# # Convection on Enceladus
#
# This example simulates convection in a liquid water ocean beneath Enceladus's ice shell.
# The simulation uses a LatitudeLongitudeGrid with Enceladus-specific parameters and
# bottom heating to drive convection.
#
# Enceladus parameters:
#   * Radius: ~252 km
#   * Surface gravity: ~0.113 m/s²
#   * Likely liquid water ocean beneath ice shell
#
# The simulation demonstrates:
#   * Convection driven by bottom heating
#   * Use of LatitudeLongitudeGrid for spherical geometry
#   * Visualization of convection patterns

using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials: TEOS10EquationOfState
using SeawaterPolynomials: thermal_expansion, haline_contraction, ρ
using CairoMakie
using Printf

# ## Enceladus parameters
#
# Enceladus is a small moon of Saturn with a subsurface ocean.
# We use appropriate physical parameters for Enceladus.

const R_enceladus = 252e3  # m, Enceladus radius (~252 km)
const g_enceladus = 0.113  # m/s², surface gravity (much weaker than Earth)
const Ω_enceladus = 2π / 32.9hours # 1/s, rotation rate of Enceladus

# ## Plotting thermal expansion to haline contraction ratio
# Create heatmaps of R = α/β where α is thermal expansion coefficient and β is haline contraction coefficient.

# Parameters
Nd, NT, NS = 256, 256, 256
z_min, z_max = -40e3, 0
T_min, T_max, S_min, S_max = -2, 20, -2, 40
S1 = 0.0  # psu, fixed salinity for R_TD
S2 = 10.0 # psu, fixed salinity for R_TD
S3 = 40.0 # psu, fixed salinity for R_TD
T1 = 0.0  # °C, fixed temperature for R_SD

eos = TEOS10EquationOfState()
z = reshape(range(z_min, z_max, length=Nd), Nd, 1)
T = reshape(range(T_min, T_max, length=NT), 1, NT)
S = reshape(range(S_min, S_max, length=NS), NS, 1)

# Compute R using broadcasting
R_TD1 = thermal_expansion.(T, S1, z, Ref(eos)) ./ haline_contraction.(T, S1, z, Ref(eos))
R_TD2 = thermal_expansion.(T, S2, z, Ref(eos)) ./ haline_contraction.(T, S2, z, Ref(eos))
R_TD3 = thermal_expansion.(T, S3, z, Ref(eos)) ./ haline_contraction.(T, S3, z, Ref(eos))
R_SD1 = thermal_expansion.(T1, S, z, Ref(eos)) ./ haline_contraction.(T1, S, z, Ref(eos))

# Create figure for heatmaps
fig = Figure(size = (1200, 600))
z_km = range(z_min, z_max, length=Nd) ./ 1e3

# First plot: R vs depth and temperature
ax1 = Axis(fig[1, 1], ylabel = "Depth (km)", xlabel = "Temperature (°C)", title = "S = $(S1) psu")
ax2 = Axis(fig[1, 2], ylabel = "Depth (km)", xlabel = "Salinity (psu)", title = "T = $(T1) °C")
ax3 = Axis(fig[2, 1], ylabel = "Depth (km)", xlabel = "Temperature (°C)", title = "S = $(S2) psu")
ax4 = Axis(fig[2, 2], ylabel = "Depth (km)", xlabel = "Temperature (°C)", title = "S = $(S3) psu")

Rlim = (-0.5, 0.5)
hm1 = heatmap!(ax1, T[:], z_km, R_TD1, colormap = :balance, colorrange = Rlim)
hm2 = heatmap!(ax2, S[:], z_km, R_SD1, colormap = :balance, colorrange = Rlim)
hm3 = heatmap!(ax3, S[:], z_km, R_TD2, colormap = :balance, colorrange = Rlim)
hm4 = heatmap!(ax4, S[:], z_km, R_TD3, colormap = :balance, colorrange = Rlim)
Colorbar(fig[3, 1:2], hm4, label = "R = α/β", vertical = false, tellwidth=false)

Label(fig[0, :], "R = α/β: Thermal Expansion / Haline Contraction Ratio", fontsize = 16)
save("enceladus_alpha_beta_ratio.png", fig)
@info "Saved alpha/beta ratio plots to enceladus_alpha_beta_ratio.png"

# Create separate figure for density plot
S_r = range(0, 40, length=256)

fig = Figure(size = (1000, 600))
ax1 = Axis(fig[1, 1], ylabel = "Ratio α / β", xlabel = "Salinity (psu)")
ax2 = Axis(fig[2, 1], ylabel = "Density (kg/m³)", xlabel = "Salinity (psu)")

for T in [0.0]
    #for z in [0.0, -30e3,-40e3]
    for z in [-30e3, -35e3, -40e3]
        α_S = thermal_expansion.(T, S_r, z, Ref(eos))
        β_S = haline_contraction.(T, S_r, z, Ref(eos))
        lines!(ax1, S_r, α_S ./ β_S, linewidth = 2, label = "T = $T °C, z = $z m")

        ρ_S = ρ.(T, S_r, z, Ref(eos))
        lines!(ax2, S_r, ρ_S, linewidth = 2, label = "T = $T °C, z = $z m")
    end
end

Legend(fig[3, 1], ax1, position = :rb, nbanks = 1, tellwidth = false)

Z_r = range(0, -45e3, length=256)
ax3 = Axis(fig[1, 2], ylabel = "Ratio α / β", xlabel = "Depth (km)")
ax4 = Axis(fig[2, 2], ylabel = "Density (kg/m³)", xlabel = "Depth (km)")

for T in [0.0]
    for S in [0.0, 10.0, 40.0]
        α_Z = thermal_expansion.(T, S, Z_r, Ref(eos))
        β_Z = haline_contraction.(T, S, Z_r, Ref(eos))
        lines!(ax3, Z_r ./ 1e3, α_Z ./ β_Z, linewidth = 2, label = "T = $T °C, S = $S psu")
        
        ρ_Z = ρ.(T, S, Z_r, Ref(eos))
        lines!(ax4, Z_r ./ 1e3, ρ_Z, linewidth = 2, label = "T = $T °C, S = $S psu")
    end
end

Legend(fig[3, 2], ax2, position = :rb, nbanks = 1, tellwidth = false)

ax5 = Axis(fig[1, 3], ylabel = "Ratio α / β", xlabel = "Temperature (°C)")
ax6 = Axis(fig[2, 3], ylabel = "Density (kg/m³)", xlabel = "Temperature (°C)")

T_r = range(-2, 20, length=256)

for S in [0.0]
    for z in [-30e3, -35e3, -40e3]
        α_T = thermal_expansion.(T_r, S, z, Ref(eos))
        β_T = haline_contraction.(T_r, S, z, Ref(eos))
        lines!(ax5, T_r, α_T ./ β_T, linewidth = 2, label = "S = $S psu, z = $z m")

        ρ_T = ρ.(T_r, S, z, Ref(eos))
        lines!(ax6, T_r, ρ_T, linewidth = 2, label = "S = $S psu, z = $z m")
    end
end

Legend(fig[3, 3], ax5, position = :rb, nbanks = 1, tellwidth = false)

save("enceladus_density.png", fig)
@info "Saved density plot to enceladus_density.png"

# ## Grid
#
# We use a LatitudeLongitudeGrid with low resolution for testing.
# The domain covers a sector of the sphere with depth representing
# the ocean beneath the ice shell.

# Low resolution for quick test
grid = LatitudeLongitudeGrid(size = (16, 16, 16),
                             longitude = (0, 360),      # degrees, 60° sector
                             latitude = (-85, 85),      # degrees, 60° sector
                             z = (-40e3, -30e3),            # m, 10 km deep ocean
                             radius = R_enceladus,
                             topology = (Periodic, Bounded, Bounded))

# ## Model
#
# We use a NonhydrostaticModel with buoyancy as a tracer.
# For Enceladus, we use a simple buoyancy formulation since
# we're simulating convection in a liquid water ocean.
using Oceananigans.Coriolis: SphericalCoriolis, NonhydrostaticFormulation
coriolis = SphericalCoriolis(rotation_rate=Ω_enceladus, formulation=NonhydrostaticFormulation())

model = NonhydrostaticModel(; grid, coriolis,
                            advection = WENO(order=5),
                            tracers = (:T, :S),
                            buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState()))

# ## Initial conditions
#
# We start with a stably stratified ocean (cold at top, warmer at bottom)
# with some small random perturbations to break symmetry.

# Start from rest with small perturbations
uᵢ(λ, φ, z) = 1e-4 * randn()
vᵢ(λ, φ, z) = 1e-4 * randn()
wᵢ(λ, φ, z) = 1e-4 * randn()

set!(model, u=uᵢ, v=vᵢ, w=wᵢ)

# ## Simulation
#
# Quick test: 100 iterations at low resolution

simulation = Simulation(model, Δt=100, stop_iteration=100)
conjure_time_step_wizard!(simulation, cfl=0.8, max_Δt=500.0)

# Progress callback
wall_clock = Ref(time_ns())

function print_progress(sim)
    u, v, w = sim.model.velocities
    progress = 100 * (iteration(sim) / sim.stop_iteration)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            progress, iteration(sim), prettytime(time(sim)), prettytime(elapsed),
            maximum(abs, u), maximum(abs, v), maximum(abs, w), prettytime(sim.Δt))

    wall_clock[] = time_ns()
    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, IterationInterval(10))

# ## Output
#
# Save fields for visualization

output_interval = IterationInterval(10)

u, v, w = model.velocities

fields_to_output = (; u, v, w)

filename = "enceladus_convection"

simulation.output_writers[:fields] =
    JLD2Writer(model, fields_to_output,
               schedule = output_interval,
               filename = filename * "_fields.jld2",
               overwrite_existing = true)

# ## Run simulation
#
@info "Running Enceladus convection simulation..."
@info "Grid: $(grid)"
# @info "Bottom buoyancy flux: $bottom_buoyancy_flux m²/s³"

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)
