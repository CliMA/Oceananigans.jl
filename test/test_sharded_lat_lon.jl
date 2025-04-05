include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

Nhosts = 1

@testset "Test sharded LatitudeLongitudeGrid simulations..." begin
    # Run the serial computation  
    Random.seed!(1234)
    bottom_height = - 500 .* rand(40, 40, 1) .- 500
    
    grid  = LatitudeLongitudeGrid(size=(40, 40, 10), 
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

    us = interior(us, :, :, 10)
    vs = interior(vs, :, :, 10)
    cs = interior(cs, :, :, 10)
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

    # What does correctness mean in this case? Probably sqrt(ϵ)?
    ϵ = sqrt(eps(Float64))

    @info "Testing xslab partitioning..."
    @test all(isapprox.(us, up1; atol=ϵ))
    @test all(isapprox.(vs, vp1; atol=ϵ))
    @test all(isapprox.(cs, cp1; atol=ϵ))
    @test all(isapprox.(ηs, ηp1; atol=ϵ))

    @info "Testing yslab partitioning..."
    @test all(isapprox.(us, up2; atol=ϵ))
    @test all(isapprox.(vs, vp2; atol=ϵ))
    @test all(isapprox.(cs, cp2; atol=ϵ))
    @test all(isapprox.(ηs, ηp2; atol=ϵ))

    @info "Testing pencil partitioning..."
    @test all(isapprox.(us, up2; atol=ϵ))
    @test all(isapprox.(vs, vp2; atol=ϵ))
    @test all(isapprox.(cs, cp2; atol=ϵ))
    @test all(isapprox.(ηs, ηp2; atol=ϵ))
end