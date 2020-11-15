# # Hello, ocean!
#
# > How inappropriate to call this planet Earth, when it is quite clearly _Ocean_.
#
#   --Arthur C. Clark

using Oceananigans, Oceananigans.Grids, Plots

grid = RegularCartesianGrid(size = (1, 64, 64),
                            x = (-6, 6), y = (-6, 6), z = (-3, 3),
                            topology = (Periodic, Periodic, Bounded))

model = IncompressibleModel(grid = grid,
                            architecture = CPU(),
                            advection = Oceananigans.Advection.WENO5(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = IsotropicDiffusivity(ν=1e-3, κ=1e-3))

@info "Simulating the ocean with" model

set!(model,
     u = (x, y, z) -> 1 + tanh(z/2),
     b = (x, y, z) -> z + exp(-x^2) * (tanh(z) - 1))

run!(Simulation(model, Δt=0.1, stop_iteration=1))

# Analyze the data

b = model.tracers.b

@show maximum(abs, b.data)

plt = contourf(xnodes(b), znodes(b), interior(b)[:, 1, :]',
               xlabel = "x", ylabel = "z", title = "Hello, ocean!",
               #xlim = (grid.xF[1], grid.xF[end]), ylim = (grid.zF[1], grid.zF[end]),
               aspectratio = :equal, linewidth = 0)

display(plt) # hide
