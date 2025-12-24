include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

# We need to initiate MPI for sharding because we are using a multi-host implementation:
# i.e. we are launching the tests with `mpiexec` and on Github actions the default MPI
# implementation is MPICH which requires calling MPI.Init(). In the case of OpenMPI,
# MPI.Init() is not necessary.

run_slab_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    Reactant.Distributed.initialize(; single_gpu_per_process=false)
    arch = Distributed(ReactantState(), partition = Partition(1, 4)) #, synchronized_communication=true)
    run_distributed_tripolar_grid(arch, "distributed_yslab_tripolar.jld2")
"""

run_pencil_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    Reactant.Distributed.initialize(; single_gpu_per_process=false)
    arch = Distributed(ReactantState(), partition = Partition(2, 2))
    run_distributed_tripolar_grid(arch, "distributed_pencil_tripolar.jld2")
"""

@testset "Test distributed TripolarGrid simulations..." begin
    # Run the serial computation
    grid  = TripolarGrid(size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5))
    grid  = analytical_immersed_tripolar_grid(grid)
    model = run_distributed_simulation(grid)

    # Retrieve Serial quantities
    us, vs, ws = model.velocities
    cs = model.tracers.c
    ηs = model.free_surface.η

    us = interior(us, :, :, 1)
    vs = interior(vs, :, :, 1)
    cs = interior(cs, :, :, 1)
    # Run the distributed grid simulation with a slab configuration
    write("distributed_slab_tests.jl", run_slab_distributed_grid)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project -O0 distributed_slab_tests.jl`)
    rm("distributed_slab_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_yslab_tripolar.jld2")["u"]
    vp = jldopen("distributed_yslab_tripolar.jld2")["v"]
    cp = jldopen("distributed_yslab_tripolar.jld2")["c"]
    ηp = jldopen("distributed_yslab_tripolar.jld2")["η"]

    rm("distributed_yslab_tripolar.jld2")

    # Test slab partitioning
    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)

    # Run the distributed grid simulation with a pencil configuration
    write("distributed_tests.jl", run_pencil_distributed_grid)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project -O0 distributed_tests.jl`)
    rm("distributed_tests.jl")

    # Retrieve Parallel quantities
    up = jldopen("distributed_pencil_tripolar.jld2")["u"]
    vp = jldopen("distributed_pencil_tripolar.jld2")["v"]
    ηp = jldopen("distributed_pencil_tripolar.jld2")["η"]
    cp = jldopen("distributed_pencil_tripolar.jld2")["c"]

    rm("distributed_pencil_tripolar.jld2")

    @test all(us .≈ up)
    @test all(vs .≈ vp)
    @test all(cs .≈ cp)
    @test all(ηs .≈ ηp)
end