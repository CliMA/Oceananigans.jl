using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using CairoMakie
using Printf

# Minimal 1D outflow test: a Gaussian blob in three tracers (background values
# c̄ ∈ {+1, 0, −1}) is advected eastward by a constant U > 0 through the east open
# boundary. The west boundary is an inflow at u = U, so the upstream tracer relaxes to c̄.
# Centered(order = 4) advection is used because it leaves the boundary halo cell exposed
# in the interior stencil (upwind/WENO mask it), so any spurious mode injected at the
# boundary stays visible.
function tracer_outflow_simulation(; arch = CPU(),
                                     Nx = 200,
                                     Lx = 10,
                                     U = 1,
                                     blob_centre = 2,
                                     blob_width = 0.5,
                                     stop_time = 12,
                                     base_simulation_name = "tracer_outflow",
                                     scheme = PerturbationAdvection(inflow_timescale = 0,
                                                                    outflow_timescale = Inf))

    grid = RectilinearGrid(arch; topology = (Bounded, Flat, Flat), size = Nx, x = (0, Lx))

    tracer_bcs(c̄) = FieldBoundaryConditions(west = OpenBoundaryCondition(c̄; scheme),
                                            east = OpenBoundaryCondition(c̄; scheme))

    boundary_conditions = (u     = FieldBoundaryConditions(west = OpenBoundaryCondition(U),
                                                           east = OpenBoundaryCondition(U)),
                           cpos  = tracer_bcs( 1),
                           czero = tracer_bcs( 0),
                           cneg  = tracer_bcs(-1))

    model = NonhydrostaticModel(grid; tracers = (:cpos, :czero, :cneg),
                                advection = Centered(order = 4),
                                boundary_conditions)

    blob(x) = exp(-((x - blob_centre) / blob_width)^2)
    set!(model, u = U,
                cpos  = x -> blob(x) + 1,
                czero = x -> blob(x),
                cneg  = x -> blob(x) - 1)

    # NonhydrostaticModel fills tracer halos with fill_open_bcs=false, so the open BC
    # never fires automatically — fire it here for the initial state and once per
    # iteration via the callback below.
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

    times       = panels[1][2].times
    # x coordinates with halos so the OBC's boundary halo cells are visible alongside the
    # interior (JLD2Writer used with_halos = true, so parent(ts[n]) carries them).
    xs          = collect(xnodes(panels[1][2][1], with_halos = true))
    xs_interior = xnodes(panels[1][2][1])
    Δx          = minimum(diff(xs))
    domain_edges = [first(xs_interior) - 0.5Δx, last(xs_interior) + 0.5Δx]

    n = Observable(1)
    fig = Figure(size = (800, 700))

    title = @lift @sprintf("t = %.2f", times[$n])
    Label(fig[0, 1], title, fontsize = 16, tellwidth = false)

    for (i, (label, ts, c̄)) in enumerate(panels)
        ax = Axis(fig[i, 1], xlabel = "x", ylabel = "c", title = label,
                  width = 700, height = 180)
        c_plt = @lift Array(parent(ts[$n]))[:, 1, 1]
        lines!(ax, xs, c_plt, color = :dodgerblue, linewidth = 1.5)
        vlines!(ax, domain_edges, color = (:black, 0.4), linestyle = :dot)
        hlines!(ax, [c̄], color = (:black, 0.3), linestyle = :dash)
        ylims!(ax, c̄ - 1.5, c̄ + 1.5)
    end

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

# Combined Hovmöller (x–t) of all three tracers in one figure with a shared colorbar, so
# the panels are directly comparable across c̄. Boundary noise / pileup / wrong-branch
# behavior shows up as vertical striations or sharp colour jumps near the domain edges —
# easier to spot here than in single animation frames.
function plot_tracer_outflow_hovmollers(filepath)
    @info "Plotting Hovmöllers from $filepath"

    panels = (("c̄ = +1", FieldTimeSeries(filepath, "cpos"),   1),
              ("c̄ =  0", FieldTimeSeries(filepath, "czero"),  0),
              ("c̄ = -1", FieldTimeSeries(filepath, "cneg"),  -1))

    times       = panels[1][2].times
    xs          = collect(xnodes(panels[1][2][1], with_halos = true))
    xs_interior = xnodes(panels[1][2][1])
    Δx          = minimum(diff(xs))
    domain_edges = [first(xs_interior) - 0.5Δx, last(xs_interior) + 0.5Δx]

    fig = Figure(size = (800, 900))
    Label(fig[0, 1:2], "Hovmöller — $(basename(filepath))", fontsize = 16, tellwidth = false)

    hm = nothing
    for (i, (label, ts, c̄)) in enumerate(panels)
        field_xt = stack([Array(parent(ts[n]))[:, 1, 1] for n in 1:length(times)]; dims = 2)
        ax = Axis(fig[i, 1], xlabel = "x", ylabel = "t", title = label)
        hm = heatmap!(ax, xs, times, field_xt; colormap = :balance, colorrange = (-1.8, 1.8))
        vlines!(ax, domain_edges, color = (:black, 0.6), linestyle = :dash)
    end
    Colorbar(fig[1:length(panels), 2], hm)

    out = "$filepath" * "_hovmoller.png"
    save(out, fig)
    @info "Saved $out"
    return fig
end

simulation = tracer_outflow_simulation()
run!(simulation)
filepath = simulation.output_writers[:snaps].filepath
plot_tracer_outflow_animation(filepath)
plot_tracer_outflow_hovmollers(filepath)
