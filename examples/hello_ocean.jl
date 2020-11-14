# # Hello, ocean!
#
# > How inappropriate to call this planet Earth, when it is quite clearly _Ocean_.
#
#   --Arthur C. Clark

using Oceananigans, Oceananigans.Grids, Plots

grid = RegularCartesianGrid(size = (1, 64, 64),
                            x = (0, 1), y = (-4, 4), z = (-4, 4),
                            topology = (Periodic, Bounded, Bounded))

model = IncompressibleModel(grid = grid,
                            architecture = CPU(),
                            advection = Oceananigans.Advection.WENO5(),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = IsotropicDiffusivity(ν=1e-3, κ=1e-3))
         
@info "Simulating a rising buoyant bubble with" model

set!(model, b = (x, y, z) -> exp(-y^2 - z^2))

run!(Simulation(model, Δt=0.01, stop_iteration=800))

# Analyze the data

b = model.tracers.b

plt = contourf(ynodes(b), znodes(b), interior(b)[1, :, :]',
               xlabel = "y", ylabel = "z", title = "Buoyancy",
               xlim = (grid.yF[1], grid.yF[end]), ylim = (grid.zF[1], grid.zF[end]),
               aspectratio = :equal, linewidth = 0)

display(plt) # hide
