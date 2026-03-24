using Oceananigans
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using CairoMakie
using Printf

function flow_over_hill_simulation(; scheme = PerturbationAdvection(),
                                     arch = CPU(),
                                     model_type = :nonhydrostatic,
                                     Nz = 16,
                                     hill_height = 1,
                                     hill_width = 10,
                                     Lz = 3 * hill_height,
                                     Lx = 10 * hill_width,
                                     U = 1,
                                     pressure_solver_constructor = nothing,
                                     cycle_periods = 5,
                                     cfl = 0.7,
                                     cell_aspect_ratio = 4,
                                     debug = false,
                                     base_simulation_name = "flow_over_hill")

    # Grid definition
    Nx = ceil(Int, Nz * Lx / (Lz * cell_aspect_ratio))
    x₀ = Lx / 3

    # Determine vertical coordinate based on model type
    if (model_type == :nonhydrostatic)
        if !isnothing(pressure_solver_constructor)
            z = (-Lz, 0)  # Default for nonhydrostatic without free surface
        else
            z = MutableVerticalDiscretization(-Lz:(Lz/Nz):0)
        end
    elseif model_type == :hydrostatic_with_implicit_surface
        z = MutableVerticalDiscretization(-Lz:(Lz/Nz):0)
    else
        error("Unknown model_type: $model_type. Expected :nonhydrostatic or :hydrostatic_with_implicit_surface")
    end

    # Create grid first
    grid_base = RectilinearGrid(arch; topology = (Bounded, Flat, Bounded), size = (Nx, Nz), x = (0, Lx), z, halo = (8, 8))
    hill(x) = hill_height * exp(-((x - x₀)/hill_width)^2) - Lz
    grid = ImmersedBoundaryGrid(grid_base, PartialCellBottom(hill))

    # Model kwargs
    u_boundaries = FieldBoundaryConditions(west = OpenBoundaryCondition(U), # No scheme here for a perfectly barotropic inflow
                                           east = OpenBoundaryCondition(U; scheme))
    boundary_conditions = (u = u_boundaries,)
    advection = WENO(; order=5, minimum_buffer_upwind_order=1)

    model_kwargs = (; boundary_conditions)

    if model_type == :nonhydrostatic
        # atm [2026-03-12] ConjugateGradientPoissonSolver doesn't work with the free surface boundary condition
        pressure_solver = isnothing(pressure_solver_constructor) ? nothing : pressure_solver_constructor(grid)
        free_surface = isnothing(pressure_solver) ? ImplicitFreeSurface() : nothing

        model_kwargs = merge(model_kwargs, (; free_surface, pressure_solver, advection))
        model_constructor = NonhydrostaticModel
    elseif model_type == :hydrostatic_with_implicit_surface
        free_surface = ImplicitFreeSurface()

        model_kwargs = merge(model_kwargs, (; free_surface, momentum_advection = advection, tracer_advection = advection, vertical_coordinate = ZStarCoordinate()))
        model_constructor = HydrostaticFreeSurfaceModel
    end

    model = model_constructor(grid; model_kwargs...)

    set!(model, u=U)

    stop_time = cycle_periods * Lx / U
    simulation = Simulation(model; Δt = 0.1 * minimum_xspacing(grid) / abs(U), stop_time, verbose = debug)
    conjure_time_step_wizard!(simulation, IterationInterval(1); cfl)

    if debug
        function progress(simulation)
            iter = simulation.model.clock.iteration
            time = simulation.model.clock.time
            dt = simulation.Δt
            progress_pct = 100 * time / stop_time

            @printf("Iteration: %05d, time: %s, Δt: %s, progress: %5.1f%%\n",
                    iter, prettytime(time), prettytime(dt), progress_pct)
        end

        add_callback!(simulation, progress, IterationInterval(100))
    end

    u, v, w = model.velocities
    ω = ∂z(u) - ∂x(w)
    outputs = (; ω, model.velocities...)

    if !isnothing(model.free_surface)
        outputs = merge(outputs, (; η = model.free_surface.displacement))
    end

    simname = "$(base_simulation_name)_$(string(model_type))_Nz=$Nz"
    simulation.output_writers[:snaps] = JLD2Writer(model, outputs,
                                                   schedule = TimeInterval(simulation.stop_time / 100),
                                                   filename = simname,
                                                   overwrite_existing = true,
                                                   with_halos = true)

    return simulation
end

function plot_flow_over_hill_animation(filepath;
                                       model_type = :nonhydrostatic,
                                       U = 1,
                                       hill_height = 1,
                                       framerate = 16,
                                       compression = 20)
    @info "Plotting flow over hill from $filepath"

    # Load results
    u_ts = FieldTimeSeries(filepath, "u")
    w_ts = FieldTimeSeries(filepath, "w")
    ω_ts = FieldTimeSeries(filepath, "ω")
    grid = u_ts.grid
    @info "Loaded results"


    # Create visualization
    n = Observable(1)
    fig = Figure(size = (600, 800))

    # Top panel: free surface elevation (1D line plot) - only if η exists in the file
    try
        η_ts = FieldTimeSeries(filepath, "η")
        η_plt = @lift η_ts[$n]
        ax_η = Axis(fig[1, 1], xlabel = "x", ylabel = "η (m)", title = "Free surface elevation", width = 500, height = 150)
        lines!(ax_η, η_plt, linewidth = 2, color = :blue)
        η_max = maximum(abs, η_ts)
        ylims!(ax_η, -η_max, η_max)
    catch e
        @info "η not found in file, skipping free surface elevation panel"
        # Create blank axis as placeholder
        ax_blank = Axis(fig[1, 1], xlabel = "", ylabel = "", title = "")
        hidedecorations!(ax_blank)
        hidespines!(ax_blank)
    end

    # Second panel: always plot vorticity (2D heatmap)
    ω_plt = @lift ω_ts[$n]
    ax_ω = Axis(fig[2, 1], xlabel = "x", ylabel = "z", title = "ω (vorticity)", width = 500, height = 150)
    hm_ω = heatmap!(ax_ω, ω_plt, colorrange = (-12 * U / hill_height, 12 * U / hill_height), colormap = :curl)
    Colorbar(fig[2, 2], hm_ω, tellwidth = false, height = Relative(0.5))

    # Third panel: u velocity
    u_plt = @lift u_ts[$n]
    ax_u = Axis(fig[3, 1], xlabel = "x", ylabel = "z", title = "u (m/s)", width = 500, height = 150)
    hm_u = heatmap!(ax_u, u_plt, colorrange = (-1.5 * U, 1.5 * U), colormap = :balance)
    Colorbar(fig[3, 2], hm_u, tellwidth = false, height = Relative(0.5))

    # Fourth panel: w velocity
    w_plt = @lift w_ts[$n]
    ax_w = Axis(fig[4, 1], xlabel = "x", ylabel = "z", title = "w (m/s)", width = 500, height = 150)
    hm_w = heatmap!(ax_w, w_plt, colorrange = (-0.5 * U, 0.5 * U), colormap = :balance)
    Colorbar(fig[4, 2], hm_w, tellwidth = false, height = Relative(0.5))

    colsize!(fig.layout, 2, Fixed(30))
    resize_to_layout!(fig)

    frames = 1:length(u_ts.times)

    @info "Recording animation"
    animation_filename = "$filepath.mp4"
    CairoMakie.record(fig, animation_filename, frames; framerate, compression) do i
        n[] = i
        i % 10 == 0 && @info "  Frame $(i) of $(length(frames))"
    end

    @info "Saved animation to $animation_filename"

    return fig
end

# Run and plot an approximately hydrostatic flow over a very flat hill using a nonhydrostatic model with an implicit free surface
hydrostatic_physics_options = (; cell_aspect_ratio=100, hill_width=100, Nz=16, base_simulation_name = "flow_over_flat_hill")
model_type = :nonhydrostatic
nh_simulation = flow_over_hill_simulation(; model_type, hydrostatic_physics_options..., debug=false)
run!(nh_simulation)
plot_flow_over_hill_animation(nh_simulation.output_writers[:snaps].filepath; model_type)

# Run and plot the same flow using a hydrostatic model with an implicit free surface
model_type = :hydrostatic_with_implicit_surface
hs_simulation = flow_over_hill_simulation(; model_type, hydrostatic_physics_options..., debug=true)
run!(hs_simulation)
plot_flow_over_hill_animation(hs_simulation.output_writers[:snaps].filepath; model_type)

# Run and plot a fully nonhydrostatic flow over a steep hill using a nonhydrostatic model with an implicit free surface
model_type = :nonhydrostatic
nh_simulation2 = flow_over_hill_simulation(; model_type, cell_aspect_ratio=4, hill_width=2, Nz=32, pressure_solver_constructor=ConjugateGradientPoissonSolver, debug=true, base_simulation_name = "flow_over_steep_hill")
run!(nh_simulation2)
plot_flow_over_hill_animation(nh_simulation2.output_writers[:snaps].filepath; model_type)
