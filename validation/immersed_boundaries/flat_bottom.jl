using Oceananigans
using Printf
using Statistics
using CairoMakie

using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Utils: prettysummary
using Oceananigans.Solvers: ConjugateGradientPoissonSolver

#+++ Create simulation
function create_flat_bottom_simulation(; use_immersed_boundary = false,
                                        Δz = 0.05,
                                        U_constant = 1.0,
                                        stop_time = 2.0,
                                        save_interval = 0.1,
                                        architecture = CPU(),
                                        filename = "flat_bottom",
                                        immersed_pressure_solver = nothing)

    # Calculate Nx and Nz from Δz
    # Domain x: 0 → 2, so Nx = 2 / Δx where Δx ≈ Δz for roughly square cells
    Δx = Δz
    Nx = round(Int, 2 / Δx)

    if use_immersed_boundary
        # Domain from -1/2 to 1 with flat bottom at z = 0
        # Total height = 1.5, so Nz = 1.5 / Δz
        total_height = 1.5
        Nz = round(Int, total_height / Δz)

        underlying_grid = RectilinearGrid(architecture, size = (Nx, Nz), halo = (4, 4),
                                          x = (0, 2), z = (-0.5, 1.0),
                                          topology = (Bounded, Flat, Bounded))

        # Create flat bottom at z = 0
        flat_bottom(x) = 0.0
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(flat_bottom))
        
        # Check that z = 0 is aligned with a grid face
        @assert 0 ∈ znodes(grid, Face()) "Δz is such that the immersed boundary does not exactly align with the bottom of the non-immersed domain"

        @info "Using immersed boundary grid with flat bottom at z = 0"
        @info "Grid size: Nx = $Nx, Nz = $Nz, Δz = $Δz"

        # Use user-provided pressure solver or default ConjugateGradient solver for immersed boundaries
        if immersed_pressure_solver === nothing
            pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=100, reltol=1e-8, abstol=1e-10)
            @info "Using default ConjugateGradientPoissonSolver for immersed boundary"
        else
            pressure_solver = immersed_pressure_solver
            @info "Using user-provided pressure solver for immersed boundary"
        end
    else
        # Regular domain from 0 to 1, no immersed boundary
        # Total height = 1.0, so Nz = 1.0 / Δz
        total_height = 1.0
        Nz = round(Int, total_height / Δz)

        grid = RectilinearGrid(architecture, size = (Nx, Nz), halo = (4, 4),
                               x = (0, 2), z = (0, 1),
                               topology = (Bounded, Flat, Bounded))

        @info "Using regular grid from z = 0 to 1"
        @info "Grid size: Nx = $Nx, Nz = $Nz, Δz = $Δz"

        # Use default FFT solver for regular grids (faster)
        pressure_solver = nothing
        @info "Using default FFT pressure solver"
    end

    # Set constant velocity boundary conditions at west and east boundaries
    u_constant_bc = OpenBoundaryCondition(U_constant)
    u_bcs = FieldBoundaryConditions(west = u_constant_bc, east = u_constant_bc)
    boundary_conditions = (; u = u_bcs,)

    model = NonhydrostaticModel(; grid, boundary_conditions, pressure_solver,
                                advection = UpwindBiased(order=3),
                                hydrostatic_pressure_anomaly = CenterField(grid),
                                timestepper = :RungeKutta3,)

    # Initial conditions with small perturbations
    uᵢ(x, z) = U_constant + 1e-2 * sin(x) * cos(π * z)
    wᵢ(x, z) = 1e-3 * cos(x) * sin(π * z)

    set!(model, u=uᵢ, w=wᵢ)

    # Time stepping
    Δt = 0.01 * Δx / U_constant
    simulation = Simulation(model; Δt, stop_time)

    # Progress callback
    wall_clock = Ref(time_ns())
    function progress(sim)
        u, v, w = model.velocities
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        @info @sprintf("Iter: %d, time: %.3f, max|u|: %.3f, max|w|: %.3f, wall time: %s",
                       iteration(sim), time(sim), maximum(abs, u), maximum(abs, w), prettytime(elapsed))
        wall_clock[] = time_ns()
        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

    # Adaptive time stepping
    wizard = TimeStepWizard(cfl=0.3)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    # Compute pressure and divergence for output
    u, v, w = model.velocities
    p = sum(model.pressures)

    # Compute divergence: ∇⋅u = ∂u/∂x + ∂w/∂z
    divergence = ∂x(u) + ∂z(w)

    # Output writer
    simulation.output_writers[:fields] =
        JLD2Writer(model, (; u, w, p, divergence);
                   schedule = TimeInterval(save_interval),
                   with_halos = true,
                   filename,
                   overwrite_existing = true)

    @info "Created simulation with grid:"
    @show model.grid
    @info "Boundary conditions:"
    @show model.velocities.u.boundary_conditions

    @info "Pressure solver:"
    @show pressure_solver

    return simulation
end
#---

#+++ Animate output
function create_visualization(regular_filename, immersed_filename)
    @info "Loading results for visualization..."

    # Load results from both simulations
    u_regular = FieldTimeSeries(regular_filename * ".jld2", "u")
    p_regular = FieldTimeSeries(regular_filename * ".jld2", "p")
    div_regular = FieldTimeSeries(regular_filename * ".jld2", "divergence")

    u_immersed = FieldTimeSeries(immersed_filename * ".jld2", "u")
    p_immersed = FieldTimeSeries(immersed_filename * ".jld2", "p")
    div_immersed = FieldTimeSeries(immersed_filename * ".jld2", "divergence")

    # Get coordinates
    x_reg, y_reg, z_reg = nodes(u_regular)
    x_imm, y_imm, z_imm = nodes(u_immersed)

    # Determine number of time steps
    Nt = min(length(u_regular.times), length(u_immersed.times))
    times = u_regular.times[1:Nt]

    @info "Creating visualization with $Nt time steps..."

    # Calculate consistent color ranges across both simulations
    u_max = max(maximum(abs, u_regular), maximum(abs, u_immersed))
    p_max = max(maximum(abs, p_regular), maximum(abs, p_immersed))
    div_max = max(maximum(abs, div_regular), maximum(abs, div_immersed))

    # Create figure
    fig = Figure(size = (1600, 1200))

    # Observable for animation
    n = Observable(1)

    # Title
    title = @lift @sprintf("Flat Bottom Comparison - t = %.2f", times[$n])
    Label(fig[1, 1:4], title, fontsize = 24)

    # Column labels
    Label(fig[2, 1:2], "Regular Grid (z: 0 → 1)", fontsize = 20)
    Label(fig[2, 3:4], "Immersed Boundary (z: -0.5 → 1, flat bottom at z=0)", fontsize = 20)

    # Consistent axis kwargs with y limits from -0.5 to 1 for both columns
    axis_kwargs = (xlabel = "x", ylabel = "z", limits = ((0, 2), (-0.5, 1.0)))

    # Create axes for each variable and simulation
    ax_u_reg = Axis(fig[3, 1]; title = "u velocity", axis_kwargs...)
    ax_u_imm = Axis(fig[3, 3]; title = "u velocity", axis_kwargs...)

    ax_p_reg = Axis(fig[4, 1]; title = "pressure", axis_kwargs...)
    ax_p_imm = Axis(fig[4, 3]; title = "pressure", axis_kwargs...)

    ax_div_reg = Axis(fig[5, 1]; title = "divergence", axis_kwargs...)
    ax_div_imm = Axis(fig[5, 3]; title = "divergence", axis_kwargs...)

    # Create observables for data (simplified using FieldTimeSeries directly)
    u_reg_plot = @lift u_regular[$n]
    u_imm_plot = @lift u_immersed[$n]

    p_reg_plot = @lift p_regular[$n]
    p_imm_plot = @lift p_immersed[$n]

    div_reg_plot = @lift div_regular[$n]
    div_imm_plot = @lift div_immersed[$n]
   
    # Create heatmaps with consistent color ranges
    hm_u_reg = heatmap!(ax_u_reg, u_reg_plot; colorrange = (-u_max, u_max), colormap = :balance)
    hm_u_imm = heatmap!(ax_u_imm, u_imm_plot; colorrange = (-u_max, u_max), colormap = :balance)
        
    hm_p_reg = heatmap!(ax_p_reg, p_reg_plot; colorrange = (-p_max, p_max), colormap = :balance)
    hm_p_imm = heatmap!(ax_p_imm, p_imm_plot; colorrange = (-p_max, p_max), colormap = :balance)
       
    hm_div_reg = heatmap!(ax_div_reg, div_reg_plot; colorrange = (-div_max, div_max), colormap = :balance)
    hm_div_imm = heatmap!(ax_div_imm, div_imm_plot; colorrange = (-div_max, div_max), colormap = :balance)

    # Add colorbars (single columns for wider panels)
    Colorbar(fig[3, 2], hm_u_reg; label = "u velocity")
    Colorbar(fig[3, 4], hm_u_imm; label = "u velocity")
    Colorbar(fig[4, 2], hm_p_reg; label = "pressure")
    Colorbar(fig[4, 4], hm_p_imm; label = "pressure")
    Colorbar(fig[5, 2], hm_div_reg; label = "∇⋅u")
    Colorbar(fig[5, 4], hm_div_imm; label = "∇⋅u")

    # Add flat bottom line to immersed boundary plots
    hlines!(ax_u_imm, 0.0; color = :black, linewidth = 3)
    hlines!(ax_p_imm, 0.0; color = :black, linewidth = 3)
    hlines!(ax_div_imm, 0.0; color = :black, linewidth = 3)

    # Time slider
    slider = Slider(fig[6, 1:4]; range = 1:Nt, startvalue = 1)
    n = slider.value

    display(fig)

    # Create movie
    moviename = "flat_bottom_comparison.mp4"
    @info "Recording movie: $moviename"
    
    frames = 1:Nt
    record(fig, moviename, frames; framerate = 8) do i
        n[] = i
        i % 10 == 0 && @info "Frame $i of $Nt"
    end

    @info "Movie saved as: $moviename"
    return fig
end
#---

#####
##### Main execution
#####

@info "Starting flat bottom comparison simulations..."

# Run both simulations with the same Δz
Δz = 0.05  # Grid spacing

# Run both simulations
@info "Running regular grid simulation..."
#regular_sim = create_flat_bottom_simulation(use_immersed_boundary = false,
#                                            filename = "regular_grid",
#                                            Δz = Δz,
#                                            stop_time = 2.0)
#run!(relar_sim)
#
#@info "nng immersed boundary simulation..."
#immersed_sim create_flat_bottom_simulation(use_immersed_boundary = true,
#                                           filename = "immersed_boundary",
#                                           Δz = Δz,
#                                           stop_time = 2.0)
#run!(immersed_sim)

@info "Creating visualization..."
fig = create_visualization("regular_grid", "immersed_boundary")

@info "Done!"

