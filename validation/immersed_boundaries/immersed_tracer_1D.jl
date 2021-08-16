pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Statistics
using Printf
using Plots
using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

const κ = 1
const ν = 1
const nz = 3
const H = nz * 5
const topo = -H + 5

underlying_grid = RegularRectilinearGrid(size=(nz,), halo=(3,), z = (-H,0),
                                                 topology = (Flat, Flat, Bounded))

solid(x, y, z) = (z <= topo)

immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(solid))

model = NonhydrostaticModel(architecture = CPU(),
                               advection = CenteredSecondOrder(),
                             timestepper = :RungeKutta3,
                                    grid = immersed_grid,
                                 tracers = (:b),
                                 closure = IsotropicDiffusivity(ν=ν, κ=κ),
                                buoyancy = BuoyancyTracer())

set!(model, b=1.0, u=0, v=0, w=0)

wizard = TimeStepWizard(cfl=0.09, Δt=0.09 * underlying_grid.Δz, max_change=1.1, max_Δt=10.0, min_Δt=0.0001)

start_time = time_ns()
stop_time = 25.

progress_message(sim) =
           @printf("i: %04d, t: %s, Δt: %s, bmax = %.1e ms⁻¹, wall time: %s\n",
                   sim.model.clock.iteration, prettytime(model.clock.time),
                   prettytime(wizard.Δt), maximum(abs, sim.model.tracers.b),
                   prettytime((time_ns() - start_time) * 1e-9))

simulation = Simulation(model, Δt=wizard, stop_time=stop_time, iteration_interval=5, progress= progress_message)

outputs = merge(model.velocities, model.tracers)

data_path = "immersed_wall_tracer_test_1D"

simulation.output_writers[:fields] =
                   JLD2OutputWriter(model, outputs,
                                    schedule = TimeInterval(25.0),
                                    prefix = data_path,
                                    field_slicer = nothing,
                                    force = true)

run!(simulation)

filepath = data_path * ".jld2"

b_timeseries = FieldTimeSeries(filepath, "b")
xb, yb, zb = nodes(b_timeseries)

b1 = b_timeseries[1]
b1i = interior(b1)[1,1,:]

bend = b_timeseries[length(b_timeseries.times)]
bendi = interior(bend)[1,1,:]
Δb = maximum(abs.(b1i - bendi))

@info "Largest change in buoyancy is $(Δb)."


bplot = plot(b1i,ones(size(zb),).*topo, label = "", color = :black, lw = 2, xlabel = "b", ylabel = "z", title = @sprintf("Buoyancy, |Δb|ₘₐₓ = %.1f", Δb), legend = :top)
plot!(b1i,zb, lw = 3, label = "initial", color = :blue)
scatter!(b1i, zb, markersize = 4, color = :blue, markershape = :diamond, label = "")
plot!(bendi,zb, lw = 3, label = "final", color = :red, linestyle = :dash)
scatter!(bendi, zb, markersize = 3, color = :red, markershape = :circle, label = "")
savefig(bplot, "b_plot1D_tracertest")




