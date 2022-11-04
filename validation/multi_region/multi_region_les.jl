using Statistics
using Printf
using JLD2
using Oceananigans
using Oceananigans.Units

@inline passive_tracer_forcing(x, y, z, t, p) = p.μ⁺ * exp(-(z - p.z₀)^2 / (2 * p.λ^2)) - p.μ⁻

# using CUDA
# CUDA.device!(1)

# Defaults:
Nx, Ny, Nz = (256, 256, 256)
extent = (512meters, 512meters, 256meters)
arch = GPU()
f = 1e-4

buoyancy_flux = 1e-8
momentum_flux = -1e-4

thermocline_type = "linear"

surface_layer_depth = 48.0
thermocline_width   = 24.0
surface_temperature = 20.0



Lx, Ly, Lz = extent
slice_depth = 8.0
Qᵇ = buoyancy_flux
Qᵘ = momentum_flux
@info "Mapping grid..."

grid = RectilinearGrid(arch,
                       size=(Nx, Ny, Nz),
                       x=(0, Lx), y=(0, Ly), z=(-Lz, 0),
                       halo=(3, 3, 3))

mrg = MultiRegionGrid(grid, partition = XPartition(4), devices = (0, 1, 2, 3))

# Buoyancy and boundary conditions
@info "Enforcing boundary conditions..."

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(), constant_salinity=35.0)

N²_surface_layer = 2e-6
N²_thermocline   = 1e-5
N²_deep          = 2e-6

α = buoyancy.equation_of_state.thermal_expansion
g = buoyancy.gravitational_acceleration

Qᶿ = Qᵇ / (α * g)
dθdz_surface_layer = N²_surface_layer / (α * g)
dθdz_thermocline   = N²_thermocline   / (α * g)
dθdz_deep          = N²_deep          / (α * g)

θ_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᶿ),
                                bottom = GradientBoundaryCondition(dθdz_deep))

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

# Tracer forcing

@info "Forcing and sponging tracers..."

# # Initial condition and sponge layer

## Fiddle with indices to get a correct discrete profile
z = znodes(Center, grid)

k_transition = searchsortedfirst(z, -surface_layer_depth)
k_deep       = searchsortedfirst(z, -(surface_layer_depth + thermocline_width))

z_transition = z[k_transition]
z_deep = z[k_deep]

θ_surface    = surface_temperature
θ_transition = θ_surface + z_transition * dθdz_surface_layer
θ_deep       = θ_transition + (z_deep - z_transition) * dθdz_thermocline

λ = 4.0
μ⁺ = 1 / 6hour
μ₀ = √(2π) * λ / grid.Lz * μ⁺ / 2
μ∞ = √(2π) * λ / grid.Lz * μ⁺

c₀_forcing = Forcing(passive_tracer_forcing, parameters=(z₀=  0.0, λ=λ, μ⁺=μ⁺, μ⁻=μ₀))
c₁_forcing = Forcing(passive_tracer_forcing, parameters=(z₀=-48.0, λ=λ, μ⁺=μ⁺, μ⁻=μ∞))
c₂_forcing = Forcing(passive_tracer_forcing, parameters=(z₀=-96.0, λ=λ, μ⁺=μ⁺, μ⁻=μ∞))

# # LES Model

# Wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# Necessary to obtain a smooth temperature distribution.

@info "Building the model..."

tracers = (:T, :c₀, :c₁, :c₂) 

model = NonhydrostaticModel(; grid = mrg, buoyancy, tracers,
                            timestepper = :QuasiAdamsBashforth2,
                            advection = WENO(),
                            coriolis = FPlane(f=f),
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (T=θ_bcs, u=u_bcs),
                            forcing = (c₀=c₀_forcing, c₁=c₁_forcing, c₂=c₂_forcing))

# # Set Initial condition

@info "Setting initial conditions..."

## Noise with 8 m decay scale
Ξ(z) = rand() * exp(z / 8)

function thermocline_structure_function(thermocline_type, z_transition, θ_transition, z_deep,
                                        θ_deep, dθdz_surface_layer, dθdz_thermocline, dθdz_deep)

    if thermocline_type == "linear"
        return z -> θ_transition + dθdz_thermocline * (z - z_transition)

    elseif thermocline_type == "cubic"
        p1 = (z_transition, θ_transition)
        p2 = (z_deep, θ_deep)
        coeffs = fit_cubic(p1, p2, dθdz_surface_layer, dθdz_deep)
        return z -> poly(z, coeffs)

    else
        @error "Invalid thermocline type: $thermocline"
    end
end

θ_thermocline = thermocline_structure_function(thermocline_type, z_transition, θ_transition, z_deep, θ_deep,
                                                dθdz_surface_layer, dθdz_thermocline, dθdz_deep)

"""
    initial_temperature(x, y, z)

Returns a three-layer initial temperature distribution. The average temperature varies in z
and is augmented by three-dimensional, surface-concentrated random noise.
"""
function initial_temperature(x, y, z)
    noise = 1e-6 * Ξ(z) * dθdz_surface_layer * grid.Lz

    if z_transition < z <= 0
        return θ_surface + dθdz_surface_layer * z + noise
    elseif z_deep < z <= z_transition
        return θ_thermocline(z) + noise
    else
        return θ_deep + dθdz_deep * (z - z_deep) + noise
    end
end

set!(model, T = initial_temperature)

# # Prepare the simulation

@info "Conjuring the simulation..."

simulation = Simulation(model; Δt=1.0, stop_iteration=1000)

mutable struct SimulationProgressMessenger{T} <: Function
    wall_time₀ :: T  # Wall time at simulation start
    wall_time⁻ :: T  # Wall time at previous callback
    iteration⁻ :: Int  # Iteration at previous callback
end

SimulationProgressMessenger(Δt) =
    SimulationProgressMessenger(
                      1e-9 * time_ns(),
                      1e-9 * time_ns(),
                      0)

function (pm::SimulationProgressMessenger)(simulation)
    model = simulation.model

    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (t / simulation.stop_time)

    current_wall_time = 1e-9 * time_ns() - pm.wall_time₀
    time_since_last_callback = 1e-9 * time_ns() - pm.wall_time⁻
    iterations_since_last_callback = i - pm.iteration⁻
    wall_time_per_step = time_since_last_callback / iterations_since_last_callback

    pm.wall_time⁻ = 1e-9 * time_ns()
    pm.iteration⁻ = i

    @info @sprintf("[%06.2f%%] iteration: % 6d, time: % 10s, Δt: % 10s, wall time: % 8s (% 8s / time step)",
                    progress, i, prettytime(t), prettytime(simulation.Δt),
                    prettytime(current_wall_time), prettytime(wall_time_per_step))
    @info ""
    return nothing
end                    

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.8, max_change=1.1, min_Δt=0.01, max_Δt=30.0)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
simulation.callbacks[:progress] = Callback(SimulationProgressMessenger(wizard), IterationInterval(10))

# # Prepare Output

@info "Strapping on checkpointer..."

run!(simulation)