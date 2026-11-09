# A uniform, spatially-constant oscillating flow carries a tracer out through an open
# boundary and back in. Over a whole number of periods the net displacement is zero, so a
# uniform tracer should be unchanged by the boundary. This compares three `TracerReservoir`
# length scales against that expectation:
#
#   - memoryless (the default, `inflow_length_scale = 0`): the boundary halo is reset to the
#     exterior value on every inflow step, so the water that just left is not the water that
#     comes back.
#   - a finite length scale: the reservoir relaxes toward the exterior value over that
#     distance, so it partly remembers.
#   - frozen (`inflow_length_scale = Inf`): the reservoir is never touched on inflow, so it
#     returns exactly the water that left.

using Oceananigans
using Oceananigans.BoundaryConditions: TracerReservoir
using Oceananigans.Units
using CairoMakie
using Statistics: mean

const Nx = 100
const Lx = 200kilometers
const T  = 1day
const ω  = 2π / T
const U₀ = 0.5
const stop_time = 4T

u_prescribed(x, z, t) = U₀ * sin(ω * t)

function tracer_reservoir_simulation(scheme, filename)
    grid = RectilinearGrid(size = (Nx, 4), x = (0, Lx), z = (-100.0, 0),
                           halo = (5, 4), topology = (Bounded, Flat, Bounded))

    c_bcs = FieldBoundaryConditions(east = ValueBoundaryCondition(0; scheme))

    model = HydrostaticFreeSurfaceModel(grid;
        velocities = PrescribedVelocityFields(u = u_prescribed),
        momentum_advection = nothing,
        tracer_advection = WENO(order = 5),
        buoyancy = nothing,
        tracers = :c,
        boundary_conditions = (; c = c_bcs))

    set!(model, c = 1)

    simulation = Simulation(model; Δt = 60, stop_time)

    simulation.output_writers[:snaps] = JLD2Writer(model, (; c = model.tracers.c),
                                                   schedule = TimeInterval(T / 40),
                                                   filename = filename, overwrite_files = true)
    return simulation
end

runs = ("memoryless" => TracerReservoir(),
        "finite (40 km)" => TracerReservoir(inflow_length_scale = 40kilometers),
        "frozen" => TracerReservoir(inflow_length_scale = Inf))

function run_and_get_filepath(scheme, name)
    simulation = tracer_reservoir_simulation(scheme, "tracer_reservoir_$name")
    run!(simulation)
    return simulation.output_writers[:snaps].filepath
end

filepaths = [run_and_get_filepath(scheme, name) for (name, scheme) in runs]

fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1], xlabel = "time (days)", ylabel = "domain-mean c",
         title = "Mean tracer content: a uniform tracer should stay at 1")
for ((name, _), filepath) in zip(runs, filepaths)
    c_ts = FieldTimeSeries(filepath, "c")
    times = c_ts.times ./ day
    means = [mean(interior(c_ts[n])) for n in 1:length(times)]
    lines!(ax, times, means, label = name)
end
hlines!(ax, [1], color = (:black, 0.3), linestyle = :dash)
axislegend(ax, position = :lb)

resize_to_layout!(fig)
save("tracer_reservoir.png", fig)
