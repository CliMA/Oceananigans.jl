using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Plots

grid = RegularRectilinearGrid(size=(128, 256), x=(-10, 10), z=(0, 5), topology=(Periodic, Flat, Bounded))

# Gaussian bump of width "1"
bump(x, y, z) = z < exp(-x^2)

grid_with_bump = ImmersedBoundaryGrid(grid, GridFittedBoundary(bump))

# Tidal forcing
tidal_forcing(x, y, z, t) = 1e-4 * cos(t)

model = HydrostaticFreeSurfaceModel(grid = grid_with_bump,
                                    momentum_advection = CenteredSecondOrder(),
                                    #free_surface = ImplicitFreeSurface(gravitational_acceleration=10, tolerance=1e-7),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=4),
                                    closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=sqrt(0.5)),
                                    forcing = (u = tidal_forcing,))

@show model
                                    
# Linear stratification with N² = 2
set!(model, b = (x, y, z) -> 2 * z)

w = model.velocities.w

progress(s) = @info "Time = $(s.model.clock.time), max|w|: $(maximum(abs, w))"
              
# Calculate time-step
c = sqrt(model.free_surface.gravitational_acceleration * grid.Lz)
@show Δt = 0.1 * grid.Δx / c

simulation = Simulation(model, Δt = Δt, stop_time = 1,
                        progress = progress, iteration_interval = 10)

@show simulation

run!(simulation)

#=
xw, yw, zw = nodes(model.velocities.w)
w = interior(model.velocities.w)[:, 1, :]

wmax = maximum(abs, w)
wlim = wmax / 2
wlevels = vcat(-[wmax], range(-wlim, stop=wlim, length=30), [wmax]) 

xw2 = reshape(xw, grid.Nx, 1)
zw2 = reshape(zw, 1, grid.Nz+1)
w[bump.(xw2, 0, zw2)] .= NaN

w_plot = heatmap(xw, zw, w'; title = "vertical velocity", color = :balance, clims = (-wlim, wlim))

display(w_plot)
=#

xu, yu, zu = nodes(model.velocities.u)
u = interior(model.velocities.u)[:, 1, :]

umax = maximum(abs, u)
ulim = umax / 2
ulevels = vcat(-[umax], range(-ulim, stop=ulim, length=30), [umax]) 

xu2 = reshape(xu, grid.Nx, 1)
zu2 = reshape(zu, 1, grid.Nz)
u[bump.(xu2, 0, zu2)] .= NaN

u_plot = heatmap(xu, zu, u'; title = "x velocity", color = :balance, clims = (-ulim, ulim))

display(u_plot)

#=
w_plot = contourf(xw, zw, w'; title = "vertical velocity",
                  xlims = (-grid.Lx/2, grid.Lx/2), ylims = (0, grid.Lz),
                  aspectratio = :equal, clims = (-wlim, wlim), color = :balance, levels = wlevels,
                  linewidth = 0)
=#

