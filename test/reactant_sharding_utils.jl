using JLD2
using MPI
using Oceananigans.DistributedComputations: reconstruct_global_field, reconstruct_global_grid
using Oceananigans.Units
using Reactant
using Random

include("dependencies_for_runtests.jl")

function distributed_child_architecture()
    reactant_test = get(ENV, "REACTANT_TEST", "false") == "true"
    return reactant_test ? Oceananigans.Architectures.ReactantState() : CPU() 
end

# Run the distributed grid simulation and save down reconstructed results
function run_distributed_latitude_longitude_grid(arch, filename)
    Random.seed!(1234)
    bottom_height = rand(40, 40, 1)

    distributed_grid = LatitudeLongitudeGrid(size=(40, 40, 1), longitude=(0, 360), latitude=(-10, 10), z=(-1000, 0), halo=(5, 5, 5))    
    distributed_grid = ImmersedBoundaryGrid(distributed_grid, GridFittedBottom(bottom_height))
    simulation       = run_latitude_longitude_simulation(distributed_grid)

    η = reconstruct_global_field(simulation.model.free_surface.η)
    u = reconstruct_global_field(simulation.model.velocities.u)
    v = reconstruct_global_field(simulation.model.velocities.v)
    c = reconstruct_global_field(simulation.model.tracers.c)

    if arch.local_rank == 0
        jldsave(filename; u = Array(interior(u, :, :, 1)),
                          v = Array(interior(v, :, :, 1)), 
                          c = Array(interior(c, :, :, 1)),
                          η = Array(interior(η, :, :, 1))) 
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()

    return nothing
end

# Just a random simulation on a tripolar grid
function run_latitude_longitude_simulation(grid)

    model = HydrostaticFreeSurfaceModel(; grid = grid,
                                          free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                          tracers = :c,
                                          buoyancy = nothing, 
                                          tracer_advection = WENO(), 
                                          momentum_advection = WENOVectorInvariant(order=3),
                                          coriolis = HydrostaticSphericalCoriolis())

    # Setup the model with a gaussian sea surface height
    # near the physical north poles and one near the equator
    ηᵢ(λ, φ, z) = exp(- (φ - 90)^2 / 10^2) + exp(- φ^2 / 10^2)

    set!(model, c = ηᵢ, η = ηᵢ)

    simulation = Simulation(model, Δt = 5minutes, stop_iteration = 100)
    
    run!(simulation)

    return simulation
end