using Oceananigans
using MPI
using Printf
using Statistics
using Oceananigans.BoundaryConditions
using Oceananigans.DistributedComputations
using Oceananigans.Grids
using Random
using JLD2
using GLMakie
using MPI
MPI.Init()

# Run with
#
# ```julia
#   mpiexec -n 4 julia --project distributed_hydrostatic_turbulence.jl
# ```

function run_simulation(nx, ny, arch; topology = (Periodic, Periodic, Bounded))
    grid = RectilinearGrid(arch; 
                           topology, 
                           size=(Nx, Ny, 1), 
                           x=(0, 4π),
                           y=(0, 4π),
                           z=MutableVerticalDiscretization((-0.5, 0)), 
                           halo=(8, 8, 8))

    # bottom(x, y) = (x > π && x < 3π/2 && y > π/2 && y < 3π/2) ? 1.0 : - grid.Lz - 1.0
    # grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom); active_cells_map = true)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        momentum_advection = WENOVectorInvariant(),
                                        free_surface = SplitExplicitFreeSurface(grid, substeps=10),
                                        tracer_advection = WENO(),
                                        # timestepper = :SplitRungeKutta3,
                                        buoyancy = nothing,
                                        coriolis = FPlane(f = 1),
                                        tracers = (:c, :constant))

    # Scale seed with rank to avoid symmetry
    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    Random.seed!(1234 * (local_rank + 1))
    mask(x, y, z) = x > 3π/2 && x < 5π/2 && y > 3π/2 && y < 5π/2

    set!(model, u=(x, y, z)->1-2rand(), v=(x, y, z)->1-2rand(), c=mask, constant=1)

    U, V = model.free_surface.barotropic_velocities
    set!(U, model.velocities.u * 0.5)
    set!(V, model.velocities.v * 0.5)

    u, v, _ = model.velocities
    η = model.free_surface.η
    outputs = merge(model.velocities, model.tracers, (η=η, U=U, V=V))

    progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time), Δt: $(sim.Δt), extrema c: $(extrema(model.tracers.constant) .- 1)"
    simulation = Simulation(model, Δt=0.02, stop_time=100.0)

    wizard = TimeStepWizard(cfl = 0.2, max_change = 1.1)

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
    simulation.callbacks[:wizard]   = Callback(wizard,   IterationInterval(10))

    filepath = "mpi_hydrostatic_turbulence"
    simulation.output_writers[:fields] =
        JLD2Writer(model, outputs, filename=filepath, schedule=TimeInterval(0.1),
                   overwrite_existing=true)

    run!(simulation)

    MPI.Barrier(MPI.COMM_WORLD)
end

Nx = 64
Ny = 64

ranks = 2
arch  = Distributed(CPU(), partition = Partition(1, ranks))

# # Run the simulation
run_simulation(Nx, Ny, arch)

# Visualize the plane
# Produce a video for variable `var`
function visualize_simulation(var)
    iter = Observable(1)

    v = Vector(undef, ranks)
    V = Vector(undef, ranks)
    x = Vector(undef, ranks)
    y = Vector(undef, ranks)

    for r in 1:ranks
        v[r] = FieldTimeSeries("mpi_hydrostatic_turbulence_rank$(r-1).jld2", var)
        nx, ny, _ = size(v[r])
        V[r] = @lift(interior(v[r][$iter], 1:nx, 1:ny, 1))

        x[r] = xnodes(v[r])
        y[r] = ynodes(v[r])
    end

    fig = Figure()
    ax = Axis(fig[1, 1])
    for r in 1:ranks
        heatmap!(ax, x[r], y[r], V[r])
    end

    GLMakie.record(fig, "hydrostatic_test_" * var * ".mp4", 1:length(v[1].times), framerate = 11) do i
        @info "step $i";
        iter[] = i;
    end
end

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    visualize_simulation("u")
    visualize_simulation("v")
    visualize_simulation("c")
    # visualize_simulation("U")
    # visualize_simulation("V")
    visualize_simulation("constant")
end

MPI.Barrier(MPI.COMM_WORLD)
