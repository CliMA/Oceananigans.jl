include("sharding_test_utils.jl")
include("dependencies_for_runtests.jl")

@testset "Test sharded LatitudeLongitudeGrid simulations..." begin
    model = sharding_test_model(CPU())
    Δt = model.clock.last_Δt

    @info "Running serial first time step..."
    first_time_step!(model, Δt)
    @info "Running serial time steps..."
    for N in 2:100
        time_step!(model, Δt)
    end

    # Retrieve serial quantities
    us = interior(model.velocities.u, :, :, 10)
    vs = interior(model.velocities.v, :, :, 10)
    cs = interior(model.tracers.c, :, :, 10)
    ηs = interior(model.free_surface.displacement, :, :, 1)

    # Run the sharded Reactant simulation in a subprocess
    run(`$(Base.julia_cmd()) --project=$(Base.active_project()) --check-bounds=auto -O0 run_sharding_tests.jl`)

    # Retrieve sharded quantities
    up = jldopen("distributed_pencil_llg.jld2")["u"]
    vp = jldopen("distributed_pencil_llg.jld2")["v"]
    cp = jldopen("distributed_pencil_llg.jld2")["c"]
    ηp = jldopen("distributed_pencil_llg.jld2")["η"]

    ϵ = sqrt(eps(Float64))

    @info "Testing pencil partitioning..."
    @test all(isapprox.(us, up; atol=ϵ))
    @test all(isapprox.(vs, vp; atol=ϵ))
    @test all(isapprox.(cs, cp; atol=ϵ))
    @test all(isapprox.(ηs, ηp; atol=ϵ))
end
