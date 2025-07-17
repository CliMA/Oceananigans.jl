using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition
using Oceananigans.Diagnostics: AdvectiveCFL
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.AbstractOperations: ∂x, ∂z, Average

# using GLMakie
using GLMakie: Observable, save, recordframe!, VideoStream
using Printf
using Test
using BenchmarkTools
using Random: seed!
seed!(156)

function create_mass_conservation_simulation(; 
    use_open_boundary_condition = true,
    immersed_bottom = nothing,
    arch = CPU(),
    N = 32,
    Lx = 1.0,
    Lz = 1.0,
    U₀ = 1.0,
    stop_time = 1,
    inflow_timescale = 1e-1,
    outflow_timescale = Inf,
    add_progress_messenger = false,
    poisson_solver = nothing,
    timestepper = :QuasiAdamsBashforth2,
    animation = false,
    animation_framerate = 12)
    # Create underlying grid
    underlying_grid = RectilinearGrid(arch, topology = (Bounded, Flat, Bounded),
                                      size = (N, N),
                                      extent = (Lx, Lz),
                                      halo = (4, 4))

    # Choose grid type based on immersed_bottom parameter
    grid = immersed_bottom isa Nothing ? underlying_grid : ImmersedBoundaryGrid(underlying_grid, immersed_bottom)

    if poisson_solver isa Nothing
        pressure_solver = grid isa ImmersedBoundaryGrid ? ConjugateGradientPoissonSolver(grid) : nothing
    elseif (poisson_solver == FFTBasedPoissonSolver) && (grid isa ImmersedBoundaryGrid)
        pressure_solver = poisson_solver(grid.underlying_grid)
    else
        pressure_solver = poisson_solver(grid)
    end

    # Set boundary conditions based on boolean flag
    if use_open_boundary_condition
        u_boundary_conditions = FieldBoundaryConditions(
            west = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale),
            east = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale)
        )
        boundary_conditions = (; u = u_boundary_conditions)
    else
        boundary_conditions = NamedTuple()
    end

    model = NonhydrostaticModel(; grid, boundary_conditions, pressure_solver, timestepper)
    uᵢ(x, z) = U₀ + 1e-2 * rand()
    set!(model, u=uᵢ)

    # Calculate time step
    Δt = 0.1 * minimum_zspacing(grid) / abs(maximum(model.velocities.u))
    simulation = Simulation(model; Δt, stop_time, verbose=false)
    conjure_time_step_wizard!(simulation, IterationInterval(1), cfl=0.1)

    #+++ Progress Messenger
    if add_progress_messenger
        # Set up progress monitoring
        u, v, w = model.velocities
        ∫∇u = Field(Average(Field(∂x(u) + ∂z(w))))

        function progress(sim)
            u, v, w = model.velocities
            compute!(∫∇u)
            max_u = maximum(abs, u)
            @info @sprintf("time: %s, max|u|: %.3f, Net flux: %.4e",
                           prettytime(time(sim)), max_u, maximum(∫∇u))
            u_critical = 2
            if max_u > u_critical
                @warn "max|u| > $u_critical, stopping simulation"
                stop_time(sim) = time(sim)
                sim.running = false
            end
        end
        add_callback!(simulation, progress, IterationInterval(5))
    end
    #---

    #+++ Animation
    if animation
        # Create figure and axes
        global fig = Figure(size = (1500, 400))
        global io = VideoStream(fig; framerate = animation_framerate)
        time_str = @sprintf("t = %s", prettytime(time(simulation)))
        #fig[0, 1:3] = Label(fig, time_str, fontsize=18, tellwidth=false)
        ax1 = Axis(fig[1, 1], title = "u-velocity", xlabel = "x", ylabel = "z")
        ax2 = Axis(fig[1, 2], title = "Divergence", xlabel = "x", ylabel = "z")
        ax3 = Axis(fig[1, 3], title = "y-Vorticity", xlabel = "x", ylabel = "z")

        # Get fields for visualization
        u, v, w = model.velocities
        ∇u = Field(∂x(u) + ∂z(w))
        ωy = Field(∂z(u) - ∂x(w))  # y-direction vorticity

        # Define update function for animation
        function update_plot(sim)
            # Compute divergence and vorticity
            compute!(∇u)
            compute!(ωy)

            hm1 = heatmap!(ax1, u, colormap = :balance, colorrange = (-5U₀, 5U₀))
            Colorbar(fig[2, 1], hm1, vertical=false)

            hm2 = heatmap!(ax2, ∇u, colormap = :balance, colorrange = (-1e-10, 1e-10))
            Colorbar(fig[2, 2], hm2, vertical=false)

            hm3 = heatmap!(ax3, ωy, colormap = :balance, colorrange = (-U₀, U₀))
            Colorbar(fig[2, 3], hm3, vertical=false)
            recordframe!(io)
        end
        update_plot(simulation)
        add_callback!(simulation, update_plot, TimeInterval(5minutes))
    end
    #---

    return simulation
end

Lx = 700meters
Lz = 600meters
bottom(x) = -500meters + (Lz/4) * exp(-(x-Lx/2)^2 / (2 * (Lx/10))^2)
common_kwargs = (; arch = CPU(),
                   immersed_bottom = GridFittedBottom(bottom),
                   Lx,
                   Lz,
                   stop_time = 1day,
                   U₀ = 0.1,
                   add_progress_messenger = true,
                   timestepper = :RungeKutta3)

simulation = create_mass_conservation_simulation(; use_open_boundary_condition = true, animation = true, common_kwargs...);
run!(simulation)
save("mass_conservation.mp4", io)