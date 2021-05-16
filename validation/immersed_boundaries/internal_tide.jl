using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Plots

grid = RegularRectilinearGrid(size=(1024, 1024), x=(-10, 10), z=(0, 5), topology=(Periodic, Flat, Bounded))

# Gaussian bump of width "1"
bump(x, y, z) = z < exp(-x^2)

grid_with_bump = ImmersedBoundaryGrid(grid, GridFittedBoundary(bump))

# Tidal forcing
tidal_forcing(x, y, z, t) = 1e-4 * cos(t)

model = HydrostaticFreeSurfaceModel(architecture = GPU(),
                                    grid = grid_with_bump,
                                    momentum_advection = CenteredSecondOrder(),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=10),
                                    closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=sqrt(0.5)),
                                    forcing = (u = tidal_forcing,))

# Linear stratification
set!(model, b = (x, y, z) -> 10 * z)

progress(s) = @info @sprintf("Progress: %.2f \%, max|w|: %.2e",
                             s.model.clock.time / s.stop_time, maximum(abs, model.velocities.w))

gravity_wave_speed = sqrt(model.free_surface.gravitational_acceleration * grid.Lz)
Δt = 0.2 * grid.Δx / gravity_wave_speed
              
simulation = Simulation(model, Δt = Δt, stop_time = 10, progress = progress, iteration_interval = 100)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(0.02),
                                                      prefix = "internal_tide"
                                                      force = true)
                        
run!(simulation)

#=
xu, yu, zu = nodes(model.velocities.u)
xw, yw, zw = nodes(model.velocities.w)

u = interior(model.velocities.u)[:, 1, :]
w = interior(model.velocities.w)[:, 1, :]

umax = maximum(abs, u)
ulim = umax / 10
ulevels = vcat(-[umax], range(-ulim, stop=ulim, length=30), [umax]) 

xu2 = reshape(xu, grid.Nx, 1)
zu2 = reshape(zu, 1, grid.Nz)
u[bump.(xu2, 0, zu2)] .= NaN

#u_plot = contourf(xu, zu, u'; title = "x velocity", color = :balance, linewidth = 0, clims = (-ulim, ulim))
u_plot = heatmap(xu, zu, u'; title = "x velocity", color = :balance, linewidth = 0, clims = (-ulim, ulim))

display(u_plot)
=#
