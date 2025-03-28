include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

Nhosts = 1

@testset "Test sharded LatitudeLongitudeGrid simulations..." begin
    # Run the serial computation    
    grid  = LatitudeLongitudeGrid(size=(40, 40, 10), longitude=(0, 360), latitude=(-10, 10), z=(-1000, 0), halo=(5, 5, 5))    
    model = run_distributed_simulation(grid)

    # Retrieve Serial quantities
    us, vs, ws = model.velocities
    cs = model.tracers.c
    ηs = model.free_surface.η

    us = interior(us, :, :, 1)
    vs = interior(vs, :, :, 1)
    cs = interior(cs, :, :, 1)
    ηs = interior(ηs, :, :, 1)

    # Run the distributed grid simulations in all the configurations
    run(`$(mpiexec()) -n $(Nhosts) $(Base.julia_cmd()) --project -O0 run_sharding_tests.jl "latlon"`)

    # Retrieve Parallel quantities
    up1 = jldopen("distributed_xslab_llg.jld2")["u"]
    vp1 = jldopen("distributed_xslab_llg.jld2")["v"]
    cp1 = jldopen("distributed_xslab_llg.jld2")["c"]
    ηp1 = jldopen("distributed_xslab_llg.jld2")["η"]

    up2 = jldopen("distributed_yslab_llg.jld2")["u"]
    vp2 = jldopen("distributed_yslab_llg.jld2")["v"]
    cp2 = jldopen("distributed_yslab_llg.jld2")["c"]
    ηp2 = jldopen("distributed_yslab_llg.jld2")["η"]

    up3 = jldopen("distributed_pencil_llg.jld2")["u"]
    vp3 = jldopen("distributed_pencil_llg.jld2")["v"]
    cp3 = jldopen("distributed_pencil_llg.jld2")["c"]
    ηp3 = jldopen("distributed_pencil_llg.jld2")["η"]

    @info "Testing xslab partitioning..."
    @test all(us .≈ up1)
    @test all(vs .≈ vp1)
    @test all(cs .≈ cp1)
    @test all(ηs .≈ ηp1)

    @info "Testing yslab partitioning..."
    @test all(us .≈ up2)
    @test all(vs .≈ vp2)
    @test all(cs .≈ cp2)
    @test all(ηs .≈ ηp2)

    @info "Testing pencil partitioning..."
    @test all(us .≈ up3)
    @test all(vs .≈ vp3)
    @test all(cs .≈ cp3)
    @test all(ηs .≈ ηp3)
end