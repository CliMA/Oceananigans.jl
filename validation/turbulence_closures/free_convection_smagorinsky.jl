using Oceananigans
using Oceananigans.Units
using Statistics
using Printf
simname = "free_convection"

N = 16
L = 400meters
grid = RectilinearGrid(size=(N, N, N), extent=(L, L, L/4), topology=(Periodic, Periodic, Bounded))


function run_free_convection(closure; grid = grid, N² = 1e-6 / second, Qb = 4e-9meter^2/second^3)
    model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3, advection = UpwindBiasedFifthOrder(),
                                buoyancy = BuoyancyTracer(),
                                tracers = :b,
                                boundary_conditions = (; b = FieldBoundaryConditions(top=FluxBoundaryCondition(Qb))),
                                closure = closure)
    @show model

    noise(x, y, z) = 1e-3 * randn()
    set!(model, u=noise, v=noise, b = (x, y, z) -> N²*z)

    u, v, w = model.velocities
    b = model.tracers.b

    simulation = Simulation(model, Δt=10minutes, stop_time=2days)

    wizard = TimeStepWizard(cfl=0.7, max_change=1.1,)
    add_callback!(simulation, wizard, IterationInterval(10))

    progress(sim) = @printf("Iteration: %d, time: %s, max(u): %f m/s, Δt: %s\n",
                            iteration(sim), prettytime(sim), maximum(abs(u)), prettytime(sim.Δt))
    add_callback!(simulation, progress, IterationInterval(100))


    if closure isa ScaleInvariantSmagorinsky
        cₛ² = model.diffusivity_fields.LM_avg / model.diffusivity_fields.MM_avg
    else
        cₛ² = Field{Nothing, Nothing, Center}(grid)
        cₛ² .= model.closure.C^2
    end

    κₑ = @show diffusivity(model.closure, model.diffusivity_fields, Val(:b))
    qb = Average(κₑ * ∂z(b), dims=(1,2))
    outputs = (; cₛ², qb)

    filename = simname * "_" * string(nameof(typeof(closure)))
    simulation.output_writers[:fields] = NetCDFOutputWriter(model, merge(model.velocities, outputs),
                                                            schedule = TimeInterval(20minutes),
                                                            filename = filename * ".nc",
                                                            indices = (:, 1, :),
                                                            global_attributes = (; Qb, N²),
                                                            overwrite_existing = true)
    run!(simulation)
end

closures = [SmagorinskyLilly(), ScaleInvariantSmagorinsky(averaging=(1,2))]
for closure in closures
    @info "Running" closure
    run_free_convection(closure)
end



@info "Start plotting"
using CairoMakie
using NCDatasets
set_theme!(Theme(fontsize = 18))
fig = Figure(size = (800, 500))
ax1 = Axis(fig[2, 1]; xlabel = "cₛ", ylabel = "z", limits = ((0, 0.3), (-100, 0)))
ax2 = Axis(fig[2, 2]; xlabel = "z", ylabel = "U", limits = ((1e-3, 4e-1), (10, 20)), xscale = log10)
n = Observable(1)

colors = [:red, :blue]
for (i, closure) in enumerate(closures)
    closure_name = string(nameof(typeof(closure)))
    local filename = simname * "_" * closure_name
    @info "Plotting from " * filename
    ds = NCDataset(filename * ".nc", "r")

    xc, zc = ds["xC"], ds["zC"]

    cₛ² = @lift sqrt.(max.(ds["cₛ²"], 0))[:, $n]
    scatterlines!(ax1, cₛ², zc, color=colors[i], markercolor=colors[i], label=closure_name)

    global times = ds["time"]
end

axislegend(ax1, labelsize=10)
title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)
frames = 1:length(times)
@info "Making a neat animation of vorticity and speed..."
record(fig, simname * ".mp4", frames, framerate=8) do i
    n[] = i
end
