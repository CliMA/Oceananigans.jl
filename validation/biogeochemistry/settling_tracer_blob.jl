using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputWriters: NetCDFWriter
using Oceananigans.Solvers: ConjugateGradientPoissonSolver

using NCDatasets
using CairoMakie

function build_grid(use_immersed_grid::Bool;
    Nx = 16,
    Nz = 16,
    Lx = 1,
    Lz = 1,
    )
    underlying_grid = RectilinearGrid(CPU();
        topology = (Bounded, Flat, Bounded),
        size = (Nx, Nz),
        x = (0, Lx),
        z = (-Lz, 0),
        halo = (5, 5)
    )

    if use_immersed_grid
        return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-3Lz/4))
    else
        return underlying_grid
    end
end

# Build the model and simulation (not the grid)
function build_simulation(grid;
    w₀ = -0.02meter/second,
    output_filename::String="nonhydrostatic_tracer_circle_2d.nc",
    progress_label::String="")

    c_forcing = AdvectiveForcing(w = w₀; grid, open_boundaries=false)

    model = NonhydrostaticModel(
        grid = grid,
        advection = WENO(order=5),
        tracers = (:c,),
        forcing = (c = c_forcing,),
        pressure_solver = ConjugateGradientPoissonSolver(grid)
    )

    # Circular initial condition for tracer c in the center of the domain
    x₀ = grid.Lx / 2
    z₀ = -grid.Lz / 2
    r = min(grid.Lx, grid.Lz) / 6
    circle(x, z; x₀, z₀, r) = (x - x₀)^2 + (z - z₀)^2 <= r^2
    c_initial(x, z) = circle(x, z; x₀, z₀, r) ? 1.0 : 0.0
    set!(model; c = c_initial)

    # Simple simulation configuration
    simulation = Simulation(model; Δt=w₀ / minimum_zspacing(grid), stop_time = 50seconds)

    wizard = TimeStepWizard(
        cfl = 0.7,
        diffusive_cfl = 0.2,
        min_Δt = 0.01seconds,
        max_change = 1.2,
    )
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

    c_integral = Integral(model.tracers.c) |> Field
    function progress(sim)
        it = iteration(sim)
        t = time(sim)
        cmax = maximum(abs, sim.model.tracers.c)
        compute!(c_integral)
        @info "$(progress_label) it=$(it) t=$(t) max(|c|)=$(cmax) ∫c dV=$(c_integral[])"
    end
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

    simulation.output_writers[:netcdf] = NetCDFWriter(model,
        (c = model.tracers.c,),
        schedule = TimeInterval(0.5seconds),
        filename = output_filename,
        overwrite_existing = true,
    )

    return simulation
end

# Build two grids and simulations: immersed vs plain
grid_immersed = build_grid(true)
grid_plain    = build_grid(false)

immersed_filename = "settling_tracer_blob_immersed.nc"
plain_filename = "settling_tracer_blob_plain.nc"
sim_immersed = build_simulation(grid_immersed; output_filename=immersed_filename, progress_label="[Immersed]")
sim_plain    = build_simulation(grid_plain;    output_filename=plain_filename,    progress_label="[Plain]")

@info "Starting simulation with immersed grid"
run!(sim_immersed)
@info "Starting simulation with plain grid"
run!(sim_plain)

@info "Both simulations completed."

#+++ Create side-by-side comparison animation from NetCDF outputs
let adv_file = immersed_filename, fun_file = plain_filename
    @info "Rendering comparison animation"

    dsa = NCDataset(adv_file); cvara = dsa["c"]; times_a = dsa["time"][:]
    x = dsa["x_caa"][:]
    z = dsa["z_aac"][:]
    close(dsa)

    dsf = NCDataset(fun_file); cvarf = dsf["c"]; times_f = dsf["time"][:]; close(dsf)
    nframes = min(length(times_a), length(times_f))

    fig = Figure(resolution = (1200, 500))
    ax1 = Axis(fig[1, 1], xlabel = "x (m)", ylabel = "z (m)", title = "AdvectiveForcing")
    ax2 = Axis(fig[1, 2], xlabel = "x (m)", ylabel = "z (m)", title = "Custom Forcing")
    heat1 = heatmap!(ax1, x, z, zeros(length(x), length(z)); colorrange = (0.0, 1.0))
    heat2 = heatmap!(ax2, x, z, zeros(length(x), length(z)); colorrange = (0.0, 1.0))
    Colorbar(fig[1, 3], heat1, label = "c")

    outfile = "nonhydrostatic_tracer_circle_2d_compare.mp4"
    record(fig, outfile, 1:nframes) do it
        dsa = NCDataset(adv_file); cvara = dsa["c"]; frame_a = cvara[:, :, it]; close(dsa)
        dsf = NCDataset(fun_file); cvarf = dsf["c"]; frame_f = cvarf[:, :, it]; close(dsf)
        heat1[3] = frame_a
        heat2[3] = frame_f
        ax1.title = "AdvectiveForcing t = $(round(times_a[it]; digits=2)) s"
        ax2.title = "Custom Forcing t = $(round(times_f[it]; digits=2)) s"
    end
    @info "Saved comparison animation to $(outfile)"
end
#---
