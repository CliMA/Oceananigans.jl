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
using Printf

# ## Enceladus parameters
#
# Enceladus is a small moon of Saturn with a subsurface ocean.
# We use appropriate physical parameters for Enceladus.

const R_enceladus = 252e3  # m, Enceladus radius (~252 km)
const g_enceladus = 0.113  # m/s², surface gravity (much weaker than Earth)
const Ω_enceladus = 2π / 32.9hours # 1/s, rotation rate of Enceladus

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
