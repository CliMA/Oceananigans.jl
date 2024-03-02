using Oceananigans
using MPI
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField
using Printf
using Statistics
using Oceananigans.BoundaryConditions
using Oceananigans.DistributedComputations    
using Random
using JLD2
using Oceananigans.ImmersedBoundaries: ActiveCellsIBG, active_interior_map

# Run with 
#
# ```julia 
#   mpiexec -n 4 julia --project distributed_hydrostatic_turbulence.jl
# ```

function run_simulation(nx, ny, arch; topology = (Periodic, Periodic, Bounded))
    grid = RectilinearGrid(arch; topology, size = (Nx, Ny, 10), extent=(4π, 4π, 0.5), halo=(8, 8, 8))
    
    bottom(x, y) = (x > π && x < 3π/2 && y > π/2 && y < 3π/2) ? 1.0 : - grid.Lz - 1.0
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom); active_cells_map = true)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        momentum_advection = VectorInvariant(vorticity_scheme=WENO(order=9)),
                                        free_surface = SplitExplicitFreeSurface(substeps=10),
                                        tracer_advection = WENO(),
                                        buoyancy = nothing,
                                        coriolis = FPlane(f = 1),
                                        tracers = :c)

    # Scale seed with rank to avoid symmetry
    local_rank = MPI.Comm_rank(arch.communicator)
    Random.seed!(1234 * (local_rank + 1))

    set!(model, u = (x, y, z) -> 1-2rand(), v = (x, y, z) -> 1-2rand())
    
    mask(x, y, z) = x > 3π/2 && x < 5π/2 && y > 3π/2 && y < 5π/2
    c = model.tracers.c

    set!(c, mask)

    u, v, _ = model.velocities
    # ζ = VerticalVorticityField(model)
    η = model.free_surface.η
    outputs = merge(model.velocities, model.tracers)

    progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time), Δt: $(sim.Δt)"
    simulation = Simulation(model, Δt=0.02, stop_time=100.0)

    wizard = TimeStepWizard(cfl = 0.2, max_change = 1.1)

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
    simulation.callbacks[:wizard]   = Callback(wizard,   IterationInterval(10))

    filepath = "mpi_hydrostatic_turbulence_rank$(local_rank)"
    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, outputs, filename=filepath, schedule=TimeInterval(0.1),
                         overwrite_existing=true)

    run!(simulation)

    MPI.Barrier(arch.communicator)
end

Nx = 32
Ny = 32

arch = Distributed(CPU(), partition = Partition(2, 2)) 

# Run the simulation
run_simulation(Nx, Ny, arch)

# Visualize the plane
# Produce a video for variable `var`
try 
    using GLMakie

    function visualize_simulation(var)
        iter = Observable(1)

        v = Vector(undef, 4)
        V = Vector(undef, 4)
        x = Vector(undef, 4)
        y = Vector(undef, 4)

        for r in 1:4
            v[r] = FieldTimeSeries("mpi_hydrostatic_turbulence_rank$(r-1).jld2", var)
            nx, ny, _ = size(v[r])
            V[r] = @lift(interior(v[r][$iter], 1:nx, 1:ny, 1))

            x[r] = xnodes(v[r])
            y[r] = ynodes(v[r])
        end

        fig = Figure()
        ax = Axis(fig[1, 1])
        for r in 1:4
            heatmap!(ax, x[r], y[r], V[r], colorrange = (-1.0, 1.0))
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
    end
catch err
    @info err
end

MPI.Barrier(arch.communicator)

