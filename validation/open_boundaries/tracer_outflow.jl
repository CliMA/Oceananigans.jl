using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using CairoMakie
using Printf

function tracer_outflow_simulation(; arch = CPU(),
                                     Nx = 200,
                                     Nz = 4,
                                     Lx = 10,
                                     Lz = 1,
                                     U = 1,
                                     blob_centre = 2,
                                     blob_width = 0.5,
                                     stop_time = 12,
                                     base_simulation_name = "tracer_outflow")

    grid = RectilinearGrid(arch; topology = (Bounded, Flat, Bounded),
                           size = (Nx, Nz), x = (0, Lx), z = (-Lz, 0), halo = (4, 4))

    scheme = PerturbationAdvection(inflow_timescale = 0, outflow_timescale = Inf)

    tracer_bcs(c̄) = FieldBoundaryConditions(west = OpenBoundaryCondition(c̄; scheme), east = OpenBoundaryCondition(c̄; scheme))

    boundary_conditions = (u     = FieldBoundaryConditions(west = OpenBoundaryCondition(U), east = OpenBoundaryCondition(U)),
                           cpos  = tracer_bcs( 1),
                           czero = tracer_bcs( 0),
                           cneg  = tracer_bcs(-1))

    # Centered advection is essential here: an upwind/WENO scheme effectively
    # ignores the halo at the outflow face (smoothness indicators downweight the
    # discontinuous halo cell), hiding the BC's behavior from the interior.
    model = NonhydrostaticModel(grid; tracers = (:cpos, :czero, :cneg),
                                advection = Centered(order = 4),
                                boundary_conditions)

    blob(x, z) = exp(-((x - blob_centre) / blob_width)^2)
    set!(model, u = U,
                cpos  = (x, z) -> blob(x, z) + 1,
                czero = (x, z) -> blob(x, z),
                cneg  = (x, z) -> blob(x, z) - 1)

    # NonhydrostaticModel fills tracer halos with fill_open_bcs=false, so the
    # PerturbationAdvection BC never fires without manually refilling — once here
    # for the initial state, and via the per-step callback below.
    fill_halo_regions!(model.tracers, model.clock, fields(model))

    Δt = 0.5 * minimum_xspacing(grid) / abs(U)
    simulation = Simulation(model; Δt, stop_time, verbose = false)

    fill_tracer_open_halos!(sim) =
        fill_halo_regions!(sim.model.tracers, sim.model.clock, fields(sim.model))
    add_callback!(simulation, fill_tracer_open_halos!, IterationInterval(1))

    function progress(sim)
        @printf("Iteration: %05d, time: %s, Δt: %s\n",
                sim.model.clock.iteration,
                prettytime(sim.model.clock.time),
                prettytime(sim.Δt))
    end
    add_callback!(simulation, progress, IterationInterval(100))

    outputs = (; model.tracers..., u = model.velocities.u)
    simulation.output_writers[:snaps] = JLD2Writer(model, outputs,
                                                   schedule = TimeInterval(stop_time / 80),
                                                   filename = base_simulation_name,
                                                   overwrite_existing = true,
                                                   with_halos = true)
    return simulation
end

function plot_tracer_outflow_animation(filepath; framerate = 12, compression = 20)
    @info "Plotting tracer outflow from $filepath"

    panels = (("c̄ = +1", FieldTimeSeries(filepath, "cpos"),   1),
              ("c̄ =  0", FieldTimeSeries(filepath, "czero"),  0),
              ("c̄ = -1", FieldTimeSeries(filepath, "cneg"),  -1))

    times = panels[1][2].times
    n = Observable(1)
    fig = Figure(size = (700, 600))

    title = @lift @sprintf("t = %.2f", times[$n])
    Label(fig[0, 1:2], title, fontsize = 16, tellwidth = false)

    for (i, (label, ts, c̄)) in enumerate(panels)
        ax = Axis(fig[i, 1], xlabel = "x", ylabel = "z", title = label,
                  width = 550, height = 130)
        c_plt = @lift ts[$n]
        hm = heatmap!(ax, c_plt, colorrange = (-2, 2), colormap = :balance)
        Colorbar(fig[i, 2], hm, tellwidth = false, height = Relative(0.8))
    end

    colsize!(fig.layout, 2, Fixed(30))
    resize_to_layout!(fig)

    frames = 1:length(times)
    animation_filename = "$filepath.mp4"
    CairoMakie.record(fig, animation_filename, frames; framerate, compression) do i
        n[] = i
        i % 10 == 0 && @info "  Frame $(i) of $(length(frames))"
    end

    @info "Saved animation to $animation_filename"
    return fig
end

simulation = tracer_outflow_simulation()
run!(simulation)
plot_tracer_outflow_animation(simulation.output_writers[:snaps].filepath)
