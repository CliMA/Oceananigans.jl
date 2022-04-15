using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Utils: prettysummary
using Printf
using GLMakie

κ = ν = 1
Nz = 32
Lz = 1

underlying_grid = RectilinearGrid(size=Nz, z=(0, Lz), topology = (Flat, Flat, Bounded))

@inline bottom_height(x, y) = 0.1
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

b_immersed_bc = FluxBoundaryCondition(1)
u_drag_func(i, j, k, grid, clock, model_fields) = - 1e-3 * model_fields.u[i, j, k]^2
u_immersed_bc = ValueBoundaryCondition(0)
u_top_bc = ValueBoundaryCondition(1)

b_bcs = FieldBoundaryConditions(immersed=b_immersed_bc)
u_bcs = FieldBoundaryConditions(immersed=u_immersed_bc, top=u_top_bc)

model = NonhydrostaticModel(; grid,
                            advection = nothing,
                            timestepper = :RungeKutta3,
                            tracers = :c,
                            closure = VerticalScalarDiffusivity(; ν, κ),
                            boundary_conditions = (u=u_bcs, b=b_bcs))

set!(model, u = (x, y, z) -> z - 0.1)

simulation = Simulation(model, Δt=1e-4, stop_iteration=1000)

outputs = merge(model.velocities, model.tracers)
simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs,
                                                    filename = "immersed_couette_flow.jld2",
                                                    schedule = IterationInterval(10),
                                                    overwrite_existing = true)

progress(sim) = @info string("Iteration: ", iteration(sim), ", time: ", time(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

run!(simulation)

ct = FieldTimeSeries("immersed_couette_flow.jld2", "c")
ut = FieldTimeSeries("immersed_couette_flow.jld2", "u")
Nt = length(ct.times)

z = znodes(ut)

fig = Figure()
axc = Axis(fig[2, 1], xlabel="c", ylabel="z")
axu = Axis(fig[2, 2], xlabel="u", ylabel="z")
slider = Slider(fig[3, 1:2], range=1:Nt, startvalue=1)
n = slider.value

title = @lift string("Immersed Couette flow at t = ", prettysummary(ct.times[$n]))
Label(fig[1, 1:2], title)

cn = @lift interior(ct[$n], 1, 1, :)
un = @lift interior(ut[$n], 1, 1, :)

lines!(axc, cn, z)
lines!(axu, un, z)

display(fig)

