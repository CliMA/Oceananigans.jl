using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition, OpenBoundaryCondition
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver, DiagonallyDominantPreconditioner, fft_poisson_solver

# using GLMakie
using GLMakie: Figure, heatmap!, Colorbar, Axis, recordframe!, VideoStream, save
using Printf
using CUDA

function create_mass_conservation_simulation(; 
    use_open_boundary_condition = true,
    stratification = nothing,
    immersed_bottom = nothing,
    arch = CPU(),
    N = 32,
    Lx = 1.0,
    Lz = 1.0,
    U₀ = 1.0,
    stop_time = 1,
    inflow_timescale = 0,
    outflow_timescale = Inf,
    add_progress_messenger = false,
    poisson_solver = nothing,
    timestepper = :QuasiAdamsBashforth2,
    animation = false,
    animation_framerate = 12)

    #+++ Choose grid and pressure solver
    underlying_grid = RectilinearGrid(arch, topology = (Bounded, Flat, Bounded),
                                      size = (N, N),
                                      extent = (Lx, Lz),
                                      halo = (4, 4))
    grid = immersed_bottom isa Nothing ? underlying_grid : ImmersedBoundaryGrid(underlying_grid, immersed_bottom)

    if poisson_solver isa Nothing
        pressure_solver = grid isa ImmersedBoundaryGrid ? ConjugateGradientPoissonSolver(grid) : nothing
    elseif poisson_solver == :fft
        if grid isa ImmersedBoundaryGrid
            pressure_solver = FFTBasedPoissonSolver(grid.underlying_grid)
        else
            pressure_solver = FFTBasedPoissonSolver(grid)
        end
    elseif poisson_solver == :conjugate_gradient_with_diagonally_dominant_preconditioner
        pressure_solver = ConjugateGradientPoissonSolver(grid, preconditioner=DiagonallyDominantPreconditioner())
    elseif poisson_solver == :conjugate_gradient_with_fft_preconditioner
        pressure_solver = ConjugateGradientPoissonSolver(grid, preconditioner=fft_poisson_solver(grid.underlying_grid))
    else
        error("Unknown poisson_solver option: $poisson_solver")
    end
    #---

    #+++ Set boundary conditions
    if use_open_boundary_condition
        u_boundary_conditions = FieldBoundaryConditions(
            west = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale),
            east = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale)
        )
        boundary_conditions = (; u = u_boundary_conditions)
    else
        boundary_conditions = NamedTuple()
    end
    #---

    #+++ Create model and simulation
    uᵢ(x, z) = U₀ * (1 + 1e-2 * rand())
    if stratification === nothing
        buoyancy = nothing
        bᵢ = 0
    else
        buoyancy = BuoyancyTracer()
        bᵢ(x, z) = stratification * z
    end
    model = NonhydrostaticModel(; grid, boundary_conditions, pressure_solver, timestepper,
                                  advection = WENO(order=5), tracers = :b, buoyancy)
    set!(model, u=uᵢ, b=bᵢ)

    # Calculate time step
    Δt = 0.1 * minimum_zspacing(grid) / abs(maximum(model.velocities.u))
    simulation = Simulation(model; Δt, stop_time, verbose=false)
    conjure_time_step_wizard!(simulation, IterationInterval(1), cfl=0.1)
    #---

    #+++ Progress Messenger
    if add_progress_messenger
        # Set up progress monitoring
        u, v, w = model.velocities
        ∫∇u = Field(Average(Field(∂x(u) + ∂z(w))))

        function progress(sim, u_critical)
            u, v, w = model.velocities
            compute!(∫∇u)
            it = sim.model.pressure_solver isa ConjugateGradientPoissonSolver ? iteration(sim.model.pressure_solver) : 0
            max_u = maximum(abs, u)
            @info @sprintf("time: %s, max|u|: %.3f, Net flux: %.4e, iteration: %d",
                           prettytime(time(sim)), max_u, maximum(∫∇u), it)

            if max_u > u_critical
                @warn "max|u| > $u_critical, stopping simulation"
                stop_time(sim) = time(sim)
                sim.running = false
            end
        end
        add_callback!(simulation, progress, IterationInterval(5); parameters = 20*U₀)
    end
    #---

    #+++ Animation
    if animation
        # Create figure and axes in 2x2 grid
        global fig = Figure(size = (1000, 800))
        global io = VideoStream(fig; framerate = animation_framerate)

        ax1 = Axis(fig[1, 1], title = "u-velocity", xlabel = "x", ylabel = "z")
        ax2 = Axis(fig[1, 2], title = "Divergence", xlabel = "x", ylabel = "z")
        ax3 = Axis(fig[3, 1], title = "y-Vorticity", xlabel = "x", ylabel = "z")
        ax4 = Axis(fig[3, 2], title = "Buoyancy", xlabel = "x", ylabel = "z")

        # Get fields for visualization
        u, v, w = model.velocities
        b = model.tracers.b
        ∇u = Field(∂x(u) + ∂z(w))
        ωy = Field(∂z(u) - ∂x(w))  # y-direction vorticity

        # Define update function for animation
        function update_plot(sim)
            # Compute divergence and vorticity
            compute!(∇u)
            compute!(ωy)

            hm1 = heatmap!(ax1, u, colormap = :balance, colorrange = (-5U₀, 5U₀))
            hm2 = heatmap!(ax2, ∇u, colormap = :balance, colorrange = (-1e-9, 1e-9))
            hm3 = heatmap!(ax3, ωy, colormap = :balance, colorrange = (-U₀/10, U₀/10))
            hm4 = heatmap!(ax4, b, colormap = :thermal)
            if time(sim) == 0
                Colorbar(fig[2, 1], hm1, vertical=false)
                Colorbar(fig[2, 2], hm2, vertical=false)
                Colorbar(fig[4, 1], hm3, vertical=false)
                Colorbar(fig[4, 2], hm4, vertical=false)
            end
            recordframe!(io)
        end

        update_plot(simulation)
        add_callback!(simulation, update_plot, TimeInterval(5minutes))
    end
    #---

    return simulation
end

# using BenchmarkTools
# simulation = create_mass_conservation_simulation(; use_open_boundary_condition = true, common_kwargs...);
# b1 = @benchmark time_step!(simulation)

# simulation = create_mass_conservation_simulation(; use_open_boundary_condition = false, common_kwargs...);
# b2 = @benchmark time_step!(simulation)

Lx = 2700meters
Lz = 850meters
bottom(x) = -(3Lz/4) + (Lz/4) * sin(2π * x / (Lx/3))
common_kwargs = (; arch = CPU(),
                   immersed_bottom = GridFittedBottom(bottom),
                   Lx,
                   Lz,
                   stop_time = 0.5day,
                   U₀ = 0.1,
                   inflow_timescale = 0,
                   outflow_timescale = Inf,
                   poisson_solver = :conjugate_gradient_with_fft_preconditioner,
                   stratification = 1e-6/seconds^2,
                   add_progress_messenger = true,
                   timestepper = :RungeKutta3)

simulation = create_mass_conservation_simulation(; use_open_boundary_condition = true, animation = true, common_kwargs...);
run!(simulation)
save("2d_flow_over_bathymetry.mp4", io)