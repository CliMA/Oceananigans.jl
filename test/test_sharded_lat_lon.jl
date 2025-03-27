include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

@testset "Test distributed TripolarGrid simulations..." begin
    # Run the serial computation    
    Random.seed!(1234)
    bottom_height = - rand(40, 40, 1) .* 500 .- 500

    grid  = LatitudeLongitudeGrid(size=(40, 40, 10), longitude=(0, 360), latitude=(-10, 10), z=(-1000, 0), halo=(5, 5, 5))    
    grid  = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
    model = run_distributed_simulation(grid)

    # Retrieve Serial quantities
    us, vs, ws = model.velocities
    cs = model.tracers.c
    ηs = model.free_surface.η

    us = interior(us, :, :, 10)
    vs = interior(vs, :, :, 10)
    cs = interior(cs, :, :, 10)
    ηs = interior(ηs, :, :, 1)

    # Run the distributed grid simulations in all the configurations
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project -O0 distributed_slab_tests.jl "latlon"`)

    # Retrieve Parallel quantities
    up1 = jldopen("distributed_xslab_llg.jld2")["u"]
    vp1 = jldopen("distributed_xslab_llg.jld2")["v"]
    cp1 = jldopen("distributed_xslab_llg.jld2")["c"]
    ηp1 = jldopen("distributed_xslab_llg.jld2")["η"]

    vp2 = jldopen("distributed_yslab_llg.jld2")["v"]
    up2 = jldopen("distributed_yslab_llg.jld2")["u"]
    cp2 = jldopen("distributed_yslab_llg.jld2")["c"]
    ηp2 = jldopen("distributed_yslab_llg.jld2")["η"]

    vp3 = jldopen("distributed_pencil_llg.jld2")["v"]
    up3 = jldopen("distributed_pencil_llg.jld2")["u"]
    cp3 = jldopen("distributed_pencil_llg.jld2")["c"]
    ηp3 = jldopen("distributed_pencil_llg.jld2")["η"]

    # Test xslab partitioning
    @test all(us .≈ up1)
    @test all(vs .≈ vp1)
    @test all(cs .≈ cp1)
    @test all(ηs .≈ ηp1)

    # Test yslab partitioning
    @test all(us .≈ up2)
    @test all(vs .≈ vp2)
    @test all(cs .≈ cp2)
    @test all(ηs .≈ ηp2)

    # Test pencil partitioning
    @test all(us .≈ up3)
    @test all(vs .≈ vp3)
    @test all(cs .≈ cp3)
    @test all(ηs .≈ ηp3)
end