using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Plots

grid = RegularRectilinearGrid(size=(256, 256), x=(-10, 10), z=(0, 5), topology=(Periodic, Flat, Bounded))

# Gaussian bump of width "1"
bump(x, y, z) = z < exp(-x^2)

grid_with_bump = ImmersedBoundaryGrid(grid, GridFittedBoundary(bump))

# Tidal forcing
tidal_forcing(x, y, z, t) = 1e-4 * cos(t)

model = HydrostaticFreeSurfaceModel(grid = grid_with_bump,
                                    momentum_advection = CenteredSecondOrder(),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=4),
                                    closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=sqrt(0.5)),
                                    forcing = (u = tidal_forcing,))

# Linear stratification
set!(model, b = (x, y, z) -> 2 * z)

progress(s) = @info "Time = $(s.model.clock.time), max|w|: $(maximum(abs, model.velocities.w))"
              
simulation = Simulation(model, Δt = 1e-3, stop_time = 2, progress = progress, iteration_interval = 10)
                        
run!(simulation)

xu, yu, zu = nodes(model.velocities.u)
u = interior(model.velocities.u)[:, 1, :]

umax = maximum(abs, u)
ulim = umax / 2
ulevels = vcat(-[umax], range(-ulim, stop=ulim, length=30), [umax]) 

xu2 = reshape(xu, grid.Nx, 1)
zu2 = reshape(zu, 1, grid.Nz)
u[bump.(xu2, 0, zu2)] .= NaN

u_plot = contourf(xu, zu, u'; title = "x velocity", color = :balance, linewidth = 0, clims = (-ulim, ulim))

display(u_plot)
