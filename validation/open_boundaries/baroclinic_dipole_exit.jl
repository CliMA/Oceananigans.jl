using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: FlatherBoundaryCondition
using Oceananigans.Fields: FunctionField
using CairoMakie
using Printf

# A baroclinic dipole (two opposite-signed, surface-intensified buoyancy lenses in
# thermal-wind balance) self-propagates through an open boundary of a stratified,
# rotating 3D domain. The run tests the full open-boundary stack:
#
#   - the residual adjustment of the discretely-imbalanced initial state radiates a
#     small barotropic pulse that exits through the Flather conditions on (U, V);
#   - the vortex dipole swims toward the `heading` boundary by mutual advection,
#     carrying velocity, a buoyancy anomaly, and a passive tracer seeded in its cores
#     through the open boundary conditions on (u, v, b, c).
#
# All four lateral boundaries are open, so `heading` can be any of :west, :east, :south,
# :north to test each boundary in turn. Verdicts:
#
#   - kinetic energy decays to a small fraction of its peak after the transit time
#     (the residual fraction measures reflection),
#   - tracer mass exits with the dipole,
#   - no pileup or grid-scale noise at the boundaries in the animation.
#
# The +b lens is an surface-intensified anticyclone (warm-core ring, f > 0), so the pair is
# oriented with the +b lens to the right of the heading.

heading_vector(heading) = heading === :west  ? (-1,  0) :
                          heading === :east  ? ( 1,  0) :
                          heading === :south ? ( 0, -1) :
                          heading === :north ? ( 0,  1) :
                          throw(ArgumentError("heading must be :west, :east, :south or :north, got :$heading"))

# The lens must be statically stable: max|∂b′/∂z| = lens_amplitude / lens_e_folding
# must stay below N² (defaults have a factor 2 margin). The barotropic substep must
# satisfy √(g Lz) (2Δt / substeps) / Δx < 1 (defaults give ≈ 0.4).
# The defaults are a small, fast configuration (~minutes on CPU) with the same grid
# spacing — and therefore the same boundary physics — as the production configuration:
# size = (96, 96, 20), extent = (300km, 300km, 1km), start_distance = 50km, stop_time = 20days.
function dipole_exit_simulation(; arch = CPU(),
                                  size = (48, 48, 12),
                                  extent = (150kilometers, 150kilometers, 1kilometers),
                                  heading = :west,
                                  latitude = 45,
                                  N² = 2e-5,                  # background stratification [s⁻²]
                                  lens_amplitude = 3e-3,      # buoyancy anomaly [m s⁻²]
                                  lens_radius = 20kilometers,
                                  lens_separation = 25kilometers,
                                  lens_e_folding = 300,          # vertical decay scale [m]
                                  start_distance = 40kilometers, # pair centre to heading boundary
                                  Δt = 10minutes,
                                  substeps = 60,
                                  stop_time = 16days,
                                  inflow_timescale = 0,
                                  outflow_timescale = Inf,
                                  model_type = :hydrostatic,
                                  base_simulation_name = "dipole_exit_$(heading)_$(model_type)")

    Lx, Ly, Lz = extent
    grid = RectilinearGrid(arch; topology = (Bounded, Bounded, Bounded),
                           size, x = (0, Lx), y = (0, Ly), z = (-Lz, 0))

    radiation = Radiation(; inflow_timescale, outflow_timescale)

    buoyancy_target(ξ, z, t) = N² * z   # same signature on x- and y-boundaries: (y, z, t) / (x, z, t)

    u_bcs = FieldBoundaryConditions(west  = NormalFlowBoundaryCondition(0; scheme = radiation),
                                    east  = NormalFlowBoundaryCondition(0; scheme = radiation),
                                    south = ValueBoundaryCondition(0; scheme = radiation),
                                    north = ValueBoundaryCondition(0; scheme = radiation))

    v_bcs = FieldBoundaryConditions(west  = ValueBoundaryCondition(0; scheme = radiation),
                                    east  = ValueBoundaryCondition(0; scheme = radiation),
                                    south = NormalFlowBoundaryCondition(0; scheme = radiation),
                                    north = NormalFlowBoundaryCondition(0; scheme = radiation))

    b_bcs = FieldBoundaryConditions(west  = ValueBoundaryCondition(buoyancy_target; scheme = radiation),
                                    east  = ValueBoundaryCondition(buoyancy_target; scheme = radiation),
                                    south = ValueBoundaryCondition(buoyancy_target; scheme = radiation),
                                    north = ValueBoundaryCondition(buoyancy_target; scheme = radiation))

    c_bcs = FieldBoundaryConditions(west  = ValueBoundaryCondition(0; scheme = radiation),
                                    east  = ValueBoundaryCondition(0; scheme = radiation),
                                    south = ValueBoundaryCondition(0; scheme = radiation),
                                    north = ValueBoundaryCondition(0; scheme = radiation))

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                    west = FlatherBoundaryCondition(0, 0),
                                    east = FlatherBoundaryCondition(0, 0))

    V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
                                    south = FlatherBoundaryCondition(0, 0),
                                    north = FlatherBoundaryCondition(0, 0))

    model = if model_type === :hydrostatic
        HydrostaticFreeSurfaceModel(grid;
                                    coriolis = FPlane(; latitude),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = (:b, :c),
                                    timestepper = :SplitRungeKutta3,
                                    momentum_advection = WENO(order = 5, minimum_buffer_upwind_order = 1),
                                    tracer_advection = WENO(order = 5, minimum_buffer_upwind_order = 1),
                                    free_surface = SplitExplicitFreeSurface(grid; substeps),
                                    boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs,
                                                           c = c_bcs, U = U_bcs, V = V_bcs))
    elseif model_type === :nonhydrostatic
        NonhydrostaticModel(grid;
                            coriolis = FPlane(; latitude),
                            buoyancy = BuoyancyTracer(),
                            tracers = (:b, :c),
                            timestepper = :RungeKutta3,
                            advection = WENO(order = 5, minimum_buffer_upwind_order = 1),
                            boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs, c = c_bcs))
    else
        throw(ArgumentError("model_type must be :hydrostatic or :nonhydrostatic, got :$model_type"))
    end

    # Lens pair perpendicular to the heading, +b lens to the right (f > 0).
    # The lenses are surface-intensified so the thermal-wind columns are single-signed
    # and the pair translates coherently by mutual advection: a mid-depth lens carries
    # opposite circulation above and below itself, has zero column-integrated dipole
    # moment, and does not self-propagate.
    # The pair starts `start_distance` from the heading boundary: at finite Rossby
    # number the cyclone/anticyclone asymmetry curves the trajectory, so the transit
    # is kept short.
    dx̂, dŷ = heading_vector(heading)
    x₀ = Lx/2 + dx̂ * (Lx/2 - start_distance)
    y₀ = Ly/2 + dŷ * (Ly/2 - start_distance)
    x₊, y₊ = x₀ + lens_separation/2 * dŷ, y₀ - lens_separation/2 * dx̂
    x₋, y₋ = x₀ - lens_separation/2 * dŷ, y₀ + lens_separation/2 * dx̂

    f = model.coriolis.f
    A, σ, h = lens_amplitude, lens_radius, lens_e_folding

    G₊(x, y) = exp(-((x - x₊)^2 + (y - y₊)^2) / 2σ^2)
    G₋(x, y) = exp(-((x - x₋)^2 + (y - y₋)^2) / 2σ^2)
    F(z) = exp(z / h)
    W(z) = h * (exp(z / h) - exp(-Lz / h))   # ∫ F dz from a quiescent bottom

    # Thermal-wind balance about a quiescent bottom: p = A (G₊ − G₋) W(z).
    bᵢ(x, y, z) = N² * z + A * (G₊(x, y) - G₋(x, y)) * F(z)
    cᵢ(x, y, z) = (G₊(x, y) + G₋(x, y)) * F(z)
    uᵢ(x, y, z) = A * W(z) / (f * σ^2) * ((y - y₊) * G₊(x, y) - (y - y₋) * G₋(x, y))
    vᵢ(x, y, z) = -A * W(z) / (f * σ^2) * ((x - x₊) * G₊(x, y) - (x - x₋) * G₋(x, y))

    if model_type === :hydrostatic
        # The free surface carries the lid pressure: g η = p(z = 0). The rigid-lid
        # nonhydrostatic model recovers it through the pressure projection instead.
        g = model.free_surface.gravitational_acceleration
        ηᵢ(x, y, z) = A * (G₊(x, y) - G₋(x, y)) * W(0) / g
        set!(model, b = bᵢ, c = cᵢ, u = uᵢ, v = vᵢ, η = ηᵢ)
    else
        set!(model, b = bᵢ, c = cᵢ, u = uᵢ, v = vᵢ)
    end

    simulation = Simulation(model; Δt, stop_time)

    wall_clock = Ref(time_ns())
    function progress(sim)
        m = sim.model
        @printf("i: %05d, t: %s, wall: %s, max|u|: %.3f, max|v|: %.3f\n",
                m.clock.iteration, prettytime(m.clock.time),
                prettytime(1e-9 * (time_ns() - wall_clock[])),
                maximum(abs, m.velocities.u), maximum(abs, m.velocities.v))
        wall_clock[] = time_ns()
        return nothing
    end
    add_callback!(simulation, progress, IterationInterval(144))

    u, v, w = model.velocities
    c = model.tracers.c

    # Flow visualization: vorticity and tracer at the surface (the lenses are
    # surface-intensified)
    ζ = Field(∂x(v) - ∂y(u))

    @info base_simulation_name
    simulation.output_writers[:slices] = JLD2Writer(model, (; ζ, c, u, v);
                                                    indices = (:, :, Base.size(grid, 3)),
                                                    schedule = TimeInterval(3hours),
                                                    filename = base_simulation_name * "_slices",
                                                    overwrite_existing = true)

    # Metrics: volume-integrated kinetic energy, tracer mass, and tracer first moments
    # (the dipole trajectory is the centroid Cx / C, Cy / C)
    x_field = FunctionField{Center, Center, Center}((x, y, z) -> x, grid)
    y_field = FunctionField{Center, Center, Center}((x, y, z) -> y, grid)

    KE = Field(Integral((u^2 + v^2) / 2))
    C  = Field(Integral(c))
    Cx = Field(Integral(c * x_field))
    Cy = Field(Integral(c * y_field))

    simulation.output_writers[:metrics] = JLD2Writer(model, (; KE, C, Cx, Cy);
                                                     schedule = TimeInterval(1hours),
                                                     filename = base_simulation_name * "_metrics",
                                                     overwrite_existing = true)

    return simulation
end

function plot_dipole_exit_animation(name; framerate = 12)
    @info "Plotting dipole exit animation from $name"

    ζ_ts = FieldTimeSeries(name * "_slices.jld2", "ζ")
    c_ts = FieldTimeSeries(name * "_slices.jld2", "c")
    KE_ts = FieldTimeSeries(name * "_metrics.jld2", "KE")
    C_ts  = FieldTimeSeries(name * "_metrics.jld2", "C")

    slice_times = ζ_ts.times
    metric_times = KE_ts.times

    KE = [KE_ts[n][1, 1, 1] for n in 1:length(metric_times)]
    C  = [C_ts[n][1, 1, 1]  for n in 1:length(metric_times)]

    n = Observable(1)
    fig = Figure(size = (1500, 550))

    title = @lift @sprintf("t = %.1f days", slice_times[$n] / day)
    Label(fig[0, 1:3], title, fontsize = 18, tellwidth = false)

    ζmax = maximum(abs, ζ_ts[length(slice_times) ÷ 3])
    ax_ζ = Axis(fig[1, 1], title = "ζ at the surface", xlabel = "x [km]", ylabel = "y [km]", aspect = 1)
    ζn = @lift interior(ζ_ts[$n], :, :, 1) ./ ζmax
    heatmap!(ax_ζ, ζn, colormap = :balance, colorrange = (-1, 1))

    ax_c = Axis(fig[1, 2], title = "tracer at the surface", xlabel = "x [km]", aspect = 1)
    cn = @lift interior(c_ts[$n], :, :, 1)
    heatmap!(ax_c, cn, colormap = :amp, colorrange = (0, 1))

    ax_m = Axis(fig[1, 3], title = "metrics", xlabel = "t [days]", limits = (0, slice_times[end] / day, 0, 1.1))
    lines!(ax_m, metric_times ./ day, KE ./ maximum(KE), label = "KE / max(KE)")
    lines!(ax_m, metric_times ./ day, C ./ C[1], label = "tracer mass / initial")
    cursor = @lift slice_times[$n] / day
    vlines!(ax_m, cursor, color = (:black, 0.4), linestyle = :dash)
    axislegend(ax_m, position = :lb)

    CairoMakie.record(fig, name * ".mp4", 1:length(slice_times); framerate) do i
        n[] = i
        i % 10 == 0 && @info "  Frame $i of $(length(slice_times))"
    end

    @info "Saved $(name).mp4"
    return fig
end

function plot_dipole_exit_trajectory(name; extent = (150kilometers, 150kilometers))
    KE_ts = FieldTimeSeries(name * "_metrics.jld2", "KE")
    C_ts  = FieldTimeSeries(name * "_metrics.jld2", "C")
    Cx_ts = FieldTimeSeries(name * "_metrics.jld2", "Cx")
    Cy_ts = FieldTimeSeries(name * "_metrics.jld2", "Cy")
    t = KE_ts.times ./ day

    KE = [KE_ts[n][1, 1, 1] for n in 1:length(t)]
    C  = [C_ts[n][1, 1, 1]  for n in 1:length(t)]
    x_centroid = [Cx_ts[n][1, 1, 1] for n in 1:length(t)] ./ C ./ 1e3
    y_centroid = [Cy_ts[n][1, 1, 1] for n in 1:length(t)] ./ C ./ 1e3

    fig = Figure(size = (1000, 450))

    ax = Axis(fig[1, 1], xlabel = "t [days]", title = "metrics — $name", limits = (0, t[end], 0, 1.1))
    lines!(ax, t, KE ./ maximum(KE), label = "KE / max(KE)", linewidth = 2)
    lines!(ax, t, C ./ C[1], label = "tracer mass / initial", linewidth = 2)
    axislegend(ax, position = :lb)

    ax2 = Axis(fig[1, 2], xlabel = "x [km]", ylabel = "y [km]", title = "tracer centroid trajectory",
               limits = (0, extent[1]/1e3, 0, extent[2]/1e3), aspect = 1)
    lines!(ax2, x_centroid, y_centroid, linewidth = 2)
    scatter!(ax2, [x_centroid[1]], [y_centroid[1]], marker = :circle, label = "start")
    scatter!(ax2, [x_centroid[end]], [y_centroid[end]], marker = :star5, label = "end")
    axislegend(ax2, position = :rt)

    out = name * "_trajectory.png"
    save(out, fig)
    @info "Saved $out — final KE fraction $(round(KE[end] / maximum(KE), digits = 4)), " *
          "final mass fraction $(round(C[end] / C[1], digits = 4))"
    return fig
end

heading = :north
model_type = :hydrostatic

simulation = dipole_exit_simulation(; heading, model_type)
run!(simulation)

name = "dipole_exit_$(heading)_$(model_type)"
plot_dipole_exit_animation(name)
