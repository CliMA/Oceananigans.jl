#=
# Flat Bottom Immersed Boundary Validation

This script validates immersed boundary implementations by comparing flow simulations
with and without immersed boundaries under various boundary condition scenarios.

## Purpose
- Test the accuracy of immersed boundary methods against regular grid solutions
- Validate that flow physics are correctly captured near immersed boundaries
- Compare different boundary condition implementations and their effects

## Test Setup
The script runs two similar simulations side-by-side for comparison:

1. **Regular Grid Simulation**:
   - Domain: z ∈ [0, 1]
   - No immersed boundaries

2. **Immersed Boundary Simulation**:
   - Domain: z ∈ [-0.5, 1]
   - Flat bottom immersed boundary at z = 0
=#

using Oceananigans
using Printf
using Statistics
using CairoMakie
using OrderedCollections

using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Utils: prettysummary
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition

#+++ Create simulation
function create_flat_bottom_simulation(; use_immersed_boundary = false,
                                        Δz = 0.05,
                                        U₀ = 1.0,
                                        stop_time = 2.0,
                                        save_interval = 0.01,
                                        architecture = CPU(),
                                        filename = "flat_bottom",
                                        immersed_pressure_solver = nothing,
                                        u_boundary_conditions = nothing,
                                        verbose = false)

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

        @info "    Using immersed boundary grid with flat bottom at z = 0"
        @info "    Grid size: Nx = $Nx, Nz = $Nz, Δz = $Δz"

        # Use user-provided pressure solver or default ConjugateGradient solver for immersed boundaries
        if immersed_pressure_solver === nothing
            pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=100)
            @info "    Using default ConjugateGradientPoissonSolver for immersed boundary"
        else
            pressure_solver = immersed_pressure_solver
            @info "    Using user-provided pressure solver for immersed boundary"
        end
    else
        # Regular domain from 0 to 1, no immersed boundary
        # Total height = 1.0, so Nz = 1.0 / Δz
        total_height = 1.0
        Nz = round(Int, total_height / Δz)

        grid = RectilinearGrid(architecture, size = (Nx, Nz), halo = (4, 4),
                               x = (0, 2), z = (0, 1),
                               topology = (Bounded, Flat, Bounded))

        @info "    Using regular grid from z = 0 to 1"
        @info "    Grid size: Nx = $Nx, Nz = $Nz, Δz = $Δz"

        # Use default FFT solver for regular grids (faster)
        pressure_solver = nothing
        @info "    Using default FFT pressure solver"
    end

    # Set u boundary conditions
    if u_boundary_conditions === nothing
        # Default: constant velocity boundary conditions at west and east boundaries
        u_constant_bc = OpenBoundaryCondition(U₀)
        u_boundary_conditions = FieldBoundaryConditions(west = u_constant_bc, east = u_constant_bc)
    end

    boundary_conditions = (; u = u_boundary_conditions,)

    model = NonhydrostaticModel(; grid, boundary_conditions, pressure_solver,
                                advection = UpwindBiased(order=3),
                                hydrostatic_pressure_anomaly = CenterField(grid),
                                timestepper = :RungeKutta3,)
    @info "Using $(summary(model.pressure_solver))"

    # Initial conditions with small perturbations
    uᵢ(x, z) = U₀ + 1e-2 * sin(x) * cos(π * z)
    set!(model, u=uᵢ)

    # Time stepping
    Δt = 0.1 * Δx / U₀
    simulation = Simulation(model; Δt, stop_time, verbose, minimum_relative_step = 1e-10)

    # Progress callback
    if verbose
        wall_clock = Ref(time_ns())
        function progress(sim)
            u, v, w = model.velocities
            elapsed = 1e-9 * (time_ns() - wall_clock[])
            @info @sprintf("Iter: %d, time: %.3f, max|u|: %.3f, max|w|: %.3f, wall time: %s",
                           iteration(sim), time(sim), maximum(abs, u), maximum(abs, w), prettytime(elapsed))
            wall_clock[] = time_ns()
            return nothing
        end
        add_callback!(simulation, progress, IterationInterval(50); name = :progress)
    end

    # Adaptive time stepping
    conjure_time_step_wizard!(simulation, IterationInterval(5), cfl=0.8)

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

    return simulation
end
#---

#+++ Animate output
function create_visualization(regular_filename, immersed_filename; suffix = "")
    @info "Loading results for visualization using files $regular_filename and $immersed_filename..."

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
    div_max = maximum(abs, div_regular)

    # Create figure (wider for 3 columns)
    fig = Figure(size = (2400, 1200))

    # Observable for animation
    n = Observable(1)

    # Title
    title = @lift @sprintf("Flat Bottom Comparison - t = %.2f", times[$n])
    Label(fig[1, 1:6], title, fontsize = 24)

    # Column labels
    Label(fig[2, 1:2], "Regular Grid", fontsize = 18)
    Label(fig[2, 3:4], "Immersed Boundary", fontsize = 18)
    Label(fig[2, 5:6], "Difference (Regular - Immersed)", fontsize = 18)

    # Consistent axis kwargs with y limits from -0.5 to 1 for both columns
    axis_kwargs = (xlabel = "x", ylabel = "z", limits = ((0, 2), (-0.5, 1.0)))

    # Create axes for each variable and simulation
    ax_u_reg = Axis(fig[3, 1]; title = "u velocity", axis_kwargs...)
    ax_u_imm = Axis(fig[3, 3]; title = "u velocity", axis_kwargs...)
    ax_u_diff = Axis(fig[3, 5]; title = "u difference", axis_kwargs...)

    ax_p_reg = Axis(fig[4, 1]; title = "pressure", axis_kwargs...)
    ax_p_imm = Axis(fig[4, 3]; title = "pressure", axis_kwargs...)
    ax_p_diff = Axis(fig[4, 5]; title = "pressure difference", axis_kwargs...)

    ax_div_reg = Axis(fig[5, 1]; title = "divergence", axis_kwargs...)
    ax_div_imm = Axis(fig[5, 3]; title = "divergence", axis_kwargs...)
    ax_div_diff = Axis(fig[5, 5]; title = "divergence difference", axis_kwargs...)

    # Create observables for data (simplified using FieldTimeSeries directly)
    for (n, time) in enumerate(u_immersed.times)
        mask_immersed_field!(u_immersed[n], NaN)
        mask_immersed_field!(p_immersed[n], NaN)
        mask_immersed_field!(div_immersed[n], NaN)
    end

    u_reg_plot = @lift u_regular[$n]
    u_imm_plot = @lift u_immersed[$n]

    p_reg_plot = @lift p_regular[$n]
    p_imm_plot = @lift p_immersed[$n]

    div_reg_plot = @lift div_regular[$n]
    div_imm_plot = @lift div_immersed[$n]

    # Create difference plots (interpolating immersed to regular grid and computing difference)
    u_diff_plot = @lift begin
        # Get interior data for comparison in overlapping region (z > 0)
        u_reg_interior = interior(u_regular[$n], :, 1, :)
        u_imm_interior = interior(u_immersed[$n], :, 1, :)

        # Find the overlapping z range (z > 0 for both grids)
        z_reg_range = findall(z_reg .>= 0)
        z_imm_range = findall(z_imm .>= 0)

        # Create difference array initialized with NaN
        u_diff = fill(NaN, size(u_reg_interior))

        # Compute difference in overlapping region
        if !isempty(z_reg_range) && !isempty(z_imm_range)
            # Use the minimum overlap
            nz_overlap = min(length(z_reg_range), length(z_imm_range))
            u_diff[:, z_reg_range[1:nz_overlap]] = u_reg_interior[:, z_reg_range[1:nz_overlap]] - u_imm_interior[:, z_imm_range[1:nz_overlap]]
        end

        u_diff
    end

    p_diff_plot = @lift begin
        p_reg_interior = interior(p_regular[$n], :, 1, :)
        p_imm_interior = interior(p_immersed[$n], :, 1, :)

        z_reg_range = findall(z_reg .>= 0)
        z_imm_range = findall(z_imm .>= 0)

        p_diff = fill(NaN, size(p_reg_interior))

        if !isempty(z_reg_range) && !isempty(z_imm_range)
            nz_overlap = min(length(z_reg_range), length(z_imm_range))
            p_diff[:, z_reg_range[1:nz_overlap]] = p_reg_interior[:, z_reg_range[1:nz_overlap]] - p_imm_interior[:, z_imm_range[1:nz_overlap]]
        end

        p_diff
    end

    div_diff_plot = @lift begin
        div_reg_interior = interior(div_regular[$n], :, 1, :)
        div_imm_interior = interior(div_immersed[$n], :, 1, :)

        z_reg_range = findall(z_reg .>= 0)
        z_imm_range = findall(z_imm .>= 0)

        div_diff = fill(NaN, size(div_reg_interior))

        if !isempty(z_reg_range) && !isempty(z_imm_range)
            nz_overlap = min(length(z_reg_range), length(z_imm_range))
            div_diff[:, z_reg_range[1:nz_overlap]] = div_reg_interior[:, z_reg_range[1:nz_overlap]] - div_imm_interior[:, z_imm_range[1:nz_overlap]]
        end

        div_diff
    end

    # Create heatmaps with consistent color ranges
    hm_u_reg = heatmap!(ax_u_reg, u_reg_plot; colorrange = (-u_max, u_max), colormap = :balance)
    hm_u_imm = heatmap!(ax_u_imm, u_imm_plot; colorrange = (-u_max, u_max), colormap = :balance, nan_color=:lightgray)
    hm_u_diff = heatmap!(ax_u_diff, x_reg, z_reg, u_diff_plot; colorrange = (-u_max/10000, u_max/10000), colormap = :balance, nan_color=:lightgray)

    hm_p_reg = heatmap!(ax_p_reg, p_reg_plot; colorrange = (-p_max, p_max), colormap = :balance)
    hm_p_imm = heatmap!(ax_p_imm, p_imm_plot; colorrange = (-p_max, p_max), colormap = :balance, nan_color=:lightgray)
    hm_p_diff = heatmap!(ax_p_diff, x_reg, z_reg, p_diff_plot; colorrange = (-p_max/10000, p_max/10000), colormap = :balance, nan_color=:lightgray)

    hm_div_reg = heatmap!(ax_div_reg, div_reg_plot; colorrange = (-div_max, div_max), colormap = :balance)
    hm_div_imm = heatmap!(ax_div_imm, div_imm_plot; colorrange = (-div_max, div_max), colormap = :balance, nan_color=:lightgray)
    hm_div_diff = heatmap!(ax_div_diff, x_reg, z_reg, div_diff_plot; colorrange = (-div_max, div_max), colormap = :balance, nan_color=:lightgray)

    # Add colorbars (single columns for wider panels)
    Colorbar(fig[3, 2], hm_u_reg; label = "u velocity")
    Colorbar(fig[3, 4], hm_u_imm; label = "u velocity")
    Colorbar(fig[3, 6], hm_u_diff; label = "u difference")

    Colorbar(fig[4, 2], hm_p_reg; label = "pressure")
    Colorbar(fig[4, 4], hm_p_imm; label = "pressure")
    Colorbar(fig[4, 6], hm_p_diff; label = "pressure diff")

    Colorbar(fig[5, 2], hm_div_reg; label = "∇⋅u")
    Colorbar(fig[5, 4], hm_div_imm; label = "∇⋅u")
    Colorbar(fig[5, 6], hm_div_diff; label = "∇⋅u difference")

    # Create movie
    moviename = "flat_bottom_comparison$suffix.mp4"
    @info "Recording movie: $moviename"

    frames = 1:Nt
    record(fig, moviename, frames; framerate = 14) do i
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

# Simulation parameters
Δz = 0.05  # Grid spacing
stop_time = 1
U₀ = 1.0
inflow_timescale = 1e-4
outflow_timescale = Inf
frequency = 100

boundary_cfl = U₀ / (frequency * Δz)
@info "Boundary CFL is $boundary_cfl"

@inline u₀(t) = U₀ * sin(2π * t * frequency)
@inline u₀(z, t) = u₀(t)

# Define different boundary condition cases
boundary_condition_cases = OrderedDict(
#    "constant_velocity" => FieldBoundaryConditions(
#        west = OpenBoundaryCondition(U₀),
#        east = OpenBoundaryCondition(U₀)
#    ),
    "sine_velocity" => FieldBoundaryConditions(
        west = OpenBoundaryCondition(u₀),
        east = OpenBoundaryCondition(u₀)
    ),
#    "constant_velocity_pad" => FieldBoundaryConditions(
#        west = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale),
#        east = PerturbationAdvectionOpenBoundaryCondition(U₀; inflow_timescale, outflow_timescale)
#    ),
#    "sine_velocity_pad" => FieldBoundaryConditions(
#        west = PerturbationAdvectionOpenBoundaryCondition(u₀; inflow_timescale, outflow_timescale),
#        east = PerturbationAdvectionOpenBoundaryCondition(u₀; inflow_timescale, outflow_timescale)
#    ),
)

for (bc_name, u_bcs) in boundary_condition_cases
    println()
    @info "Running simulations with boundary condition: $bc_name"
    regular_filename = "regular_grid_$(bc_name)"
    immersed_filename = "immersed_boundary_$(bc_name)"

    # Run regular grid simulation
    @info "  Running regular grid simulation..."
    regular_sim = create_flat_bottom_simulation(
        use_immersed_boundary = false,
        filename = regular_filename,
        u_boundary_conditions = u_bcs;
        Δz, stop_time, U₀
    )
    run!(regular_sim)

    # Run immersed boundary simulation
    @info "  Running immersed boundary simulation..."
    immersed_sim = create_flat_bottom_simulation(
        use_immersed_boundary = true,
        filename = immersed_filename,
        u_boundary_conditions = u_bcs;
        Δz, stop_time, U₀
    )
    run!(immersed_sim)

    # Create visualization for this boundary condition case
    @info "  Creating visualization for $bc_name..."
    fig = create_visualization(regular_filename, immersed_filename, suffix = "_$bc_name");
end
