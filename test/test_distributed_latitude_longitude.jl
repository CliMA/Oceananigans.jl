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

# The serial version of the TripolarGrid substitutes the second half of the last row of the grid
# This is not done in the distributed version, so we need to undo this substitution if we want to
# compare the results. Otherwise very tiny differences caused by finite precision compuations
# will appear in the last row of the grid.

# Run the distributed grid simulation and save down reconstructed results
function run_latitude_longitude_simulation(arch, filename)
    Random.seed!(1234)
    bottom_height = rand(40, 40, 1)

    distributed_grid = LatitudeLongitudeGrid(arch; size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5))    
    distributed_grid = ImmersedBoundaryGrid(distributed_grid, GridFittedBottom(bottom_height))
    simulation       = run_latitude_longitude_simulation(distributed_grid)

    η  = reconstruct_global_field(simulation.model.free_surface.η)
    u  = reconstruct_global_field(simulation.model.velocities.u)
    v  = reconstruct_global_field(simulation.model.velocities.v)
    c  = reconstruct_global_field(simulation.model.tracers.c)

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

run_yslab_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tripolar_tests_utils.jl")
    child_arch = distributed_child_architecture()
    arch = Distributed(child_arch, partition = Partition(1, 4))
    run_distributed_latitude_longitude_grid(arch, "distributed_yslab_llg.jld2")
"""

run_xslab_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tripolar_tests_utils.jl")
    child_arch = distributed_child_architecture()
    arch = Distributed(child_arch, partition = Partition(4, 1))
    run_distributed_latitude_longitude_grid(arch, "distributed_xslab_llg.jld2")
"""

run_pencil_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tripolar_tests_utils.jl")
    child_arch = distributed_child_architecture()
    arch = Distributed(child_arch, partition = Partition(2, 2))
    run_distributed_latitude_longitude_grid(arch, "distributed_pencil_llg.jld2")
"""

function distributed_child_architecture()
    reactant_test = get(ENV, "REACTANT_TEST", "false") == "true"
    return reactant_test ? Oceananigans.Architectures.ReactantState() : CPU() 
end

@testset "Test distributed LatitudeLongitudeGrid simulations..." begin
    # Run the serial computation    
    Random.seed!(1234)
    bottom_height = rand(40, 40, 1)

    grid = LatitudeLongitudeGrid(arch; size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5))    
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

    simulation = run_tripolar_simulation(grid)

    # Retrieve Serial quantities
    us, vs, ws = simulation.model.velocities
    cs = simulation.model.tracers.c
    ηs = simulation.model.free_surface.η

    us = interior(us, :, :, 1)
    vs = interior(vs, :, :, 1)
    cs = interior(cs, :, :, 1)
    # Run the distributed grid simulation with a slab configuration
    write("distributed_yslab_llg_tests.jl", run_yslab_distributed_grid)
    run(`$(mpiexec()) -n 4 julia --project -O0 distributed_yslab_llg_tests.jl`)
    rm("distributed_yslab_llg_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_yslab_llg.jld2")["u"]
    vp = jldopen("distributed_yslab_llg.jld2")["v"]
    cp = jldopen("distributed_yslab_llg.jld2")["c"]
    ηp = jldopen("distributed_yslab_llg.jld2")["η"]

    rm("distributed_yslab_llg.jld2")

    # Test slab partitioning
    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)

    # Run the distributed grid simulation with a pencil configuration
    write("distributed_xslab_llg_tests.jl", run_pencil_distributed_grid)
    run(`$(mpiexec()) -n 4 julia --project -O0 distributed_xslab_llg_tests.jl`)
    rm("distributed_xslab_llg_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_xslab_llg.jld2")["u"]
    vp = jldopen("distributed_xslab_llg.jld2")["v"]
    ηp = jldopen("distributed_xslab_llg.jld2")["η"]
    cp = jldopen("distributed_xslab_llg.jld2")["c"]

    rm("distributed_xslab_llg.jld2")
    
    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)
    
    child_arch = distributed_child_architecture()

    # We try now with more ranks in the x-direction. This is not a trivial
    # test as we are now splitting, not only where the singularities are, but
    # also in the middle of the north fold. This is a more challenging test
    write("distributed_pencil_llg_tests.jl", run_large_pencil_distributed_grid)
    run(`$(mpiexec()) -n 8 julia --project -O0 distributed_pencil_llg_tests.jl`)
    rm("distributed_pencil_llg_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_pencil_llg.jld2")["u"]
    vp = jldopen("distributed_pencil_llg.jld2")["v"]
    ηp = jldopen("distributed_pencil_llg.jld2")["η"]
    cp = jldopen("distributed_pencil_llg.jld2")["c"]

    rm("distributed_pencil_llg.jld2")

    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)
end