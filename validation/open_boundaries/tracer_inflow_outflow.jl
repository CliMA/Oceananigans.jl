using Oceananigans
using CairoMakie
using Printf

# Minimal 1D inflow + outflow test: a Gaussian blob in three tracers (background values
# c̄ ∈ {+1, 0, −1}) starts outside the domain, enters through the west open boundary
# carried by a constant U > 0, transits, and exits through the east open boundary.
# The boundary values are the analytic solution c(x, t) = c̄ + blob(x − Ut) evaluated at
# the boundary, so the truth is known everywhere at all times: a clean entrance tests the
# inflow relaxation, a clean exit tests the radiation.
# Centered(order = 4) advection is used because it leaves the boundary halo cell exposed
# in the interior stencil (upwind/WENO mask it), so any spurious mode injected at the
# boundary stays visible.
function tracer_outflow_simulation(; arch = CPU(),
                                     Nx = 200,
                                     Lx = 10,
                                     U = 1,
                                     blob_centre = -3,
                                     blob_width = 0.5,
                                     stop_time = 16,
                                     base_simulation_name = "tracer_inflow_outflow",
                                     scheme = Radiation(inflow_timescale = 0,
                                                        outflow_timescale = Inf))

    grid = RectilinearGrid(arch; topology = (Bounded, Flat, Flat), size = Nx, x = (0, Lx))

    blob(x, t) = exp(-((x - U * t - blob_centre) / blob_width)^2)
    cp(x, t) = blob(x, t) + 1
    cm(x, t) = blob(x, t) - 1
    co(x, t) = blob(x, t)

    # The boundary value is the analytic solution evaluated at the boundary position;
    # the discrete form gives access to the clock (continuous-form boundary functions
    # receive only tangential coordinates, not the boundary-normal position).
    tracer_bcs(c) = FieldBoundaryConditions(west = ValueBoundaryCondition((j, k, grid, clock, fields) -> c(0,  clock.time); scheme, discrete_form = true),
                                            east = ValueBoundaryCondition((j, k, grid, clock, fields) -> c(Lx, clock.time); scheme, discrete_form = true))

    boundary_conditions = (u     = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(U),
                                                           east = NormalFlowBoundaryCondition(U)),
                           cpos  = tracer_bcs(cp),
                           czero = tracer_bcs(co),
                           cneg  = tracer_bcs(cm))

    model = NonhydrostaticModel(grid; tracers = (:cpos, :czero, :cneg),
                                advection = Centered(order = 4),
                                boundary_conditions)

    set!(model, u = U,
                cpos  = x -> cp(x, 0),
                czero = x -> co(x, 0),
                cneg  = x -> cm(x, 0))

    Δt = 0.5 * minimum_xspacing(grid) / abs(U)
    simulation = Simulation(model; Δt, stop_time, verbose = false)

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

function plot_tracer_outflow_animation(filepaths; framerate = 12, compression = 20)
    @info "Plotting tracer outflow from $(filepaths)"

    panels = (("c̄ = +1, PA", FieldTimeSeries(filepaths[1], "cpos"),   1),
              ("c̄ =  0, PA", FieldTimeSeries(filepaths[1], "czero"),  0),
              ("c̄ = -1, PA", FieldTimeSeries(filepaths[1], "cneg"),  -1),
              ("c̄ = +1, RD", FieldTimeSeries(filepaths[2], "cpos"),   1),
              ("c̄ =  0, RD", FieldTimeSeries(filepaths[2], "czero"),  0),
              ("c̄ = -1, RD", FieldTimeSeries(filepaths[2], "cneg"),  -1))

    times = panels[1][2].times

    n = Observable(1)
    fig = Figure(size = (800, 700))

    title = @lift @sprintf("t = %.2f", times[$n])
    Label(fig[0, 1:2], title, fontsize = 16, tellwidth = false)

    for (i, (label, ts, c̄)) in enumerate(panels)
        ax = Axis(fig[mod(i-1, 3)+1, ((i-1)÷3)+1], xlabel = "x", ylabel = "c", title = label, width = 700, height = 180)
        c_plt = @lift ts[$n]
        lines!(ax, c_plt, color = :dodgerblue, linewidth = 1.5)
        hlines!(ax, [c̄], color = (:black, 0.3), linestyle = :dash)
        ylims!(ax, c̄ - 1.5, c̄ + 1.5)
    end

    resize_to_layout!(fig)

    frames = 1:length(times)
    animation_filename = "tracer_inflow_outflow.mp4"
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
function plot_tracer_outflow_hovmollers(filepaths)
    @info "Plotting Hovmöllers from $filepath"

    panels = (("c̄ = +1, PA", FieldTimeSeries(filepaths[1], "cpos"),   1),
              ("c̄ =  0, PA", FieldTimeSeries(filepaths[1], "czero"),  0),
              ("c̄ = -1, PA", FieldTimeSeries(filepaths[1], "cneg"),  -1),
              ("c̄ = +1, RD", FieldTimeSeries(filepaths[2], "cpos"),   1),
              ("c̄ =  0, RD", FieldTimeSeries(filepaths[2], "czero"),  0),
              ("c̄ = -1, RD", FieldTimeSeries(filepaths[2], "cneg"),  -1))

    times = panels[1][2].times
    xs    = collect(xnodes(panels[1][2][1]))

    fig = Figure(size = (800, 900))
    Label(fig[0, 1:2], "Hovmöller — tracer_inflow_outflow", fontsize = 16, tellwidth = false)

    hm = nothing
    for (i, (label, ts, c̄)) in enumerate(panels)
        field_xt = stack([Array(interior(ts[n]))[:, 1, 1] for n in 1:length(times)]; dims = 2)
        ax = Axis(fig[mod(i-1, 3), ((i-1)÷3)+1], xlabel = "x", ylabel = "t", title = label)
        hm = heatmap!(ax, xs, times, field_xt; colormap = :balance, colorrange = (-1.8, 1.8))
    end
    Colorbar(fig[1:length(panels), 2], hm)

    out = "tracer_inflow_outflow_hovmoller.png"
    save(out, fig)
    @info "Saved $out"
    return fig
end

simPA = tracer_outflow_simulation(scheme=PerturbationAdvection(), base_simulation_name = "tracer_inflow_outflow_pa")
simRD = tracer_outflow_simulation(scheme=Radiation(), base_simulation_name = "tracer_inflow_outflow_ra")

run!(simPA)
run!(simRD)

filepathPA = simPA.output_writers[:snaps].filepath
filepathRD = simRD.output_writers[:snaps].filepath

plot_tracer_outflow_animation([filepathPA, filepathRD])
plot_tracer_outflow_hovmollers([filepathPA, filepathRD])
