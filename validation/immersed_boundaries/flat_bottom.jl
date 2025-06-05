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
        @assert 0 ∈ znodes(immersed_sim.model.grid, Face()) "Δz is such that the immersed boundary does not exactly align with the bottom of the non-immersed domain"
        
        # Create flat bottom at z = 0
        flat_bottom(x) = 0.0
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(flat_bottom))
        
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
    w_regular = FieldTimeSeries(regular_filename * ".jld2", "w") 
    p_regular = FieldTimeSeries(regular_filename * ".jld2", "p")
    div_regular = FieldTimeSeries(regular_filename * ".jld2", "divergence")

    u_immersed = FieldTimeSeries(immersed_filename * ".jld2", "u")
    w_immersed = FieldTimeSeries(immersed_filename * ".jld2", "w")
    p_immersed = FieldTimeSeries(immersed_filename * ".jld2", "p")
    div_immersed = FieldTimeSeries(immersed_filename * ".jld2", "divergence")

    # Get coordinates
    x_reg, y_reg, z_reg = nodes(u_regular)
    x_imm, y_imm, z_imm = nodes(u_immersed)

    # Determine number of time steps
    Nt = min(length(u_regular.times), length(u_immersed.times))
    times = u_regular.times[1:Nt]

    @info "Creating visualization with $Nt time steps..."

    # Create figure
    fig = Figure(size = (1600, 1200))

    # Time slider
    slider = Slider(fig[7, 1:6], range = 1:Nt, startvalue = 1)
    n = slider.value

    # Title
    title = @lift string("Comparison at t = ", @sprintf("%.2f", times[$n]))
    Label(fig[1, 1:6], title, textsize = 24)

    # Column labels
    Label(fig[2, 1:3], "Regular Grid (z: 0 → 1)", textsize = 20)
    Label(fig[2, 4:6], "Immersed Boundary (z: -0.5 → 1, flat bottom at z=0)", textsize = 20)

    # Create axes for each variable and simulation
    ax_u_reg = Axis(fig[3, 1], aspect = DataAspect(), title = "u velocity", xlabel = "x", ylabel = "z")
    ax_u_imm = Axis(fig[3, 4], aspect = DataAspect(), title = "u velocity", xlabel = "x", ylabel = "z")

    ax_p_reg = Axis(fig[4, 1], aspect = DataAspect(), title = "pressure", xlabel = "x", ylabel = "z")
    ax_p_imm = Axis(fig[4, 4], aspect = DataAspect(), title = "pressure", xlabel = "x", ylabel = "z")

    ax_div_reg = Axis(fig[5, 1], aspect = DataAspect(), title = "divergence", xlabel = "x", ylabel = "z")
    ax_div_imm = Axis(fig[5, 4], aspect = DataAspect(), title = "divergence", xlabel = "x", ylabel = "z")

    # Create observables for the data
    u_reg_data = @lift begin
        un = u_regular[$n]
        interior(un, :, 1, :)
    end

    u_imm_data = @lift begin 
        un = u_immersed[$n]
        mask_immersed_field!(un, NaN)
        interior(un, :, 1, :)
    end

    p_reg_data = @lift begin
        pn = p_regular[$n]
        interior(pn, :, 1, :)
    end

    p_imm_data = @lift begin
        pn = p_immersed[$n]
        mask_immersed_field!(pn, NaN)
        interior(pn, :, 1, :)
    end

    div_reg_data = @lift begin
        dn = div_regular[$n]
        interior(dn, :, 1, :)
    end

    div_imm_data = @lift begin
        dn = div_immersed[$n]
        mask_immersed_field!(dn, NaN)
        interior(dn, :, 1, :)
    end

    # Create heatmaps
    u_max = max(maximum(abs, u_regular), maximum(abs, u_immersed))
    p_max = max(maximum(abs, p_regular), maximum(abs, p_immersed))
    div_max = max(maximum(abs, div_regular), maximum(abs, div_immersed))

    hm_u_reg = heatmap!(ax_u_reg, x_reg, z_reg, u_reg_data, colorrange = (-u_max, u_max), colormap = :balance)
    hm_u_imm = heatmap!(ax_u_imm, x_imm, z_imm, u_imm_data, colorrange = (-u_max, u_max), colormap = :balance)

    hm_p_reg = heatmap!(ax_p_reg, x_reg, z_reg, p_reg_data, colorrange = (-p_max, p_max), colormap = :balance)  
    hm_p_imm = heatmap!(ax_p_imm, x_imm, z_imm, p_imm_data, colorrange = (-p_max, p_max), colormap = :balance)

    hm_div_reg = heatmap!(ax_div_reg, x_reg, z_reg, div_reg_data, colorrange = (-div_max, div_max), colormap = :balance)
    hm_div_imm = heatmap!(ax_div_imm, x_imm, z_imm, div_imm_data, colorrange = (-div_max, div_max), colormap = :balance)

    # Add colorbars
    Colorbar(fig[3, 2], hm_u_reg, label = "u")
    Colorbar(fig[3, 5], hm_u_imm, label = "u")
    Colorbar(fig[4, 2], hm_p_reg, label = "p") 
    Colorbar(fig[4, 5], hm_p_imm, label = "p")
    Colorbar(fig[5, 2], hm_div_reg, label = "∇⋅u")
    Colorbar(fig[5, 5], hm_div_imm, label = "∇⋅u")

    # Add flat bottom line to immersed boundary plots
    hlines!(ax_u_imm, 0.0, color = :black, linewidth = 3, label = "flat bottom")
    hlines!(ax_p_imm, 0.0, color = :black, linewidth = 3, label = "flat bottom") 
    hlines!(ax_div_imm, 0.0, color = :black, linewidth = 3, label = "flat bottom")

    display(fig)

    # Create movie
    moviename = "flat_bottom_comparison.mp4"
    @info "Recording movie: $moviename"
    record(fig, moviename, 1:Nt, framerate = 8) do i
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
regular_sim = create_flat_bottom_simulation(use_immersed_boundary = false, 
                                            filename = "regular_grid",
                                            Δz = Δz,
                                            stop_time = 2.0)
run!(regular_sim)

@info "Running immersed boundary simulation..."
immersed_sim = create_flat_bottom_simulation(use_immersed_boundary = true,
                                             filename = "immersed_boundary", 
                                             Δz = Δz,
                                             stop_time = 2.0)
run!(immersed_sim)

@info "Creating visualization..."
fig = create_visualization("regular_grid", "immersed_boundary")

@info "Done!"

