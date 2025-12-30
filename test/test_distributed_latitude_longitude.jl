using JLD2
using MPI
using Oceananigans
using Oceananigans.DistributedComputations: reconstruct_global_field, reconstruct_global_grid
using Oceananigans.Units
using Reactant
using Random
using Test

include("distributed_tests_utils.jl")

run_xslab_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(4, 1))
    run_distributed_latitude_longitude_grid(arch, "distributed_xslab_llg.jld2")
"""

run_yslab_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(1, 4))
    run_distributed_latitude_longitude_grid(arch, "distributed_yslab_llg.jld2")
"""

run_pencil_distributed_grid = """
    using MPI
    MPI.Init()
    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(2, 2))
    run_distributed_latitude_longitude_grid(arch, "distributed_pencil_llg.jld2")
"""

@testset "Test distributed LatitudeLongitudeGrid simulations..." begin
    # Run the serial computation
    Random.seed!(1234)
    bottom_height = - rand(40, 40, 1) .* 500 .- 500

    grid = LatitudeLongitudeGrid(size=(40, 40, 10),
                                 longitude=(0, 360),
                                 latitude=(-10, 10),
                                 z=(-1000, 0),
                                 halo=(5, 5, 5))

    grid  = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
    model = run_distributed_simulation(grid)

    # Retrieve Serial quantities
    us, vs, ws = model.velocities
    cs = model.tracers.c
    ηs = model.free_surface.η

    us = interior(us, :, :, 1)
    vs = interior(vs, :, :, 1)
    cs = interior(cs, :, :, 1)

    # Run the distributed grid simulation with a pencil configuration
    write("distributed_xslab_llg_tests.jl", run_xslab_distributed_grid)
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

    # We try now with more ranks in the x-direction. This is not a trivial
    # test as we are now splitting, not only where the singularities are, but
    # also in the middle of the north fold. This is a more challenging test
    write("distributed_pencil_llg_tests.jl", run_pencil_distributed_grid)
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
