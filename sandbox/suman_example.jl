using Random
using Printf
using Oceananigans
using Oceananigans.Units: minute, minutes, hour
Lz=32
Nz=32
S = 1.1 # Stretching factor
hyperbolically_spaced_nodes(k) = -Lz-Lz*(tanh(S * ( (-(k-34) - 1) / Nz - 1)) / tanh(S))
computational_grid = VerticallyStretchedRectilinearGrid(size = (32, 32, 32), 
                                                               architecture = CPU(),
                                                               x = (0,64),
                                                               y = (0,64),
                                                               halo = (3, 3, 3),
                                                               z_faces = hyperbolically_spaced_nodes)
pause
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4, β=8e-4))
Qʰ = 200  # W m⁻², surface _heat_ flux
ρₒ = 1026 # kg m⁻³, average density at the surface of the world ocean
cᴾ = 3991 # J K⁻¹ s⁻¹, typical heat capacity for seawater
Qᵀ = Qʰ / (ρₒ * cᴾ) # K m⁻¹ s⁻¹, surface _temperature_ flux
dTdz = 0.01 # K m⁻¹
T_bcs = TracerBoundaryConditions(computational_grid,
                                 top = FluxBoundaryCondition(Qᵀ),
                                 bottom = GradientBoundaryCondition(dTdz))
u₁₀ = 10     # m s⁻¹, average wind velocity 10 meters above the ocean
cᴰ = 2.5e-3  # dimensionless drag coefficient
ρₐ = 1.225   # kg m⁻³, average density of air at sea-level
Qᵘ = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻²
u_bcs = UVelocityBoundaryConditions(computational_grid, top = FluxBoundaryCondition(Qᵘ))

@inline Qˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S
evaporation_rate = 1e-3 / hour
evaporation_bc = FluxBoundaryCondition(Qˢ, field_dependencies=:S, parameters=evaporation_rate)
S_bcs = TracerBoundaryConditions(computational_grid, top=evaporation_bc)
model = IncompressibleModel(architecture = CPU(),
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            grid = computational_grid,
                            coriolis = FPlane(f=1e-4),
                            buoyancy = buoyancy,
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))
# Random noise damped at top and bottom
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
# Temperature initial condition: a stable density gradient with random noise superposed.
Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-6 * Ξ(z)
# Velocity initial condition: random noise scaled by the friction velocity.
uᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)
# `set!` the `model` fields using functions or constants:
set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)
cfl=AdvectiveCFL(0.1)
start_time = time_ns() # so we can print the total elapsed wall time
# Print a progress message
progress_message(sim) =
    @printf("i: %04d, t: %s, cfl: %s, wmax = %.1e ms⁻¹, wall time: %s\n",
            sim.model.clock.iteration, prettytime(model.clock.time),
            cfl(model), maximum(abs, sim.model.velocities.w),
            prettytime((time_ns() - start_time) * 1e-9))
simulation = Simulation(model, Δt=0.1, stop_time=40minutes, iteration_interval=10,
                        progress=progress_message,parameters=(;cfl))
# Create a NamedTuple with eddy viscosity
eddy_viscosity = (νₑ = model.diffusivities.νₑ,)
simulation.output_writers[:slices] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                           prefix = "ocean_wind_mixing_and_convection2",
                     field_slicer = FieldSlicer(j=Int(computational_grid.Ny/2)),
                         schedule = TimeInterval(1minute),
                            force = true)
run!(simulation)
