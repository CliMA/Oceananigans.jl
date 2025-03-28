include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

Nhosts = 1

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
    ηs = interior(ηs, :, :, 1)
    cs = interior(cs, :, :, 1)
    bs = parent(bs)[:, :, 1]

    # Run the distributed grid simulations in all the configurations
    run(`$(mpiexec()) -n $(Nhosts) $(Base.julia_cmd()) --project -O0 run_sharding_tests.jl "tripolar"`)

    # Retrieve Parallel quantities
    bp1 = jldopen("distributed_xslab_trg.jld2")["b"]
    up1 = jldopen("distributed_xslab_trg.jld2")["u"]
    vp1 = jldopen("distributed_xslab_trg.jld2")["v"]
    cp1 = jldopen("distributed_xslab_trg.jld2")["c"]
    ηp1 = jldopen("distributed_xslab_trg.jld2")["η"]

    bp2 = jldopen("distributed_yslab_trg.jld2")["b"]
    vp2 = jldopen("distributed_yslab_trg.jld2")["v"]
    up2 = jldopen("distributed_yslab_trg.jld2")["u"]
    cp2 = jldopen("distributed_yslab_trg.jld2")["c"]
    ηp2 = jldopen("distributed_yslab_trg.jld2")["η"]

    bp3 = jldopen("distributed_pencil_trg.jld2")["b"]
    vp3 = jldopen("distributed_pencil_trg.jld2")["v"]
    up3 = jldopen("distributed_pencil_trg.jld2")["u"]
    cp3 = jldopen("distributed_pencil_trg.jld2")["c"]
    ηp3 = jldopen("distributed_pencil_trg.jld2")["η"]

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