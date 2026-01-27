function test_minimal_restore(arch, FT, pickup_method, model_type)
    N = 16
    L = 50

    grid = RectilinearGrid(arch, FT,
                           size = (N, N, N),
                           topology = (Periodic, Bounded, Bounded),
                           extent = (L, L, L))

    if model_type == :nonhydrostatic
        model = NonhydrostaticModel(grid)
    elseif model_type == :hydrostatic
        model = HydrostaticFreeSurfaceModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    end

    simulation = Simulation(model; Δt=1.0, stop_time=3.0)

    prefix = "mwe_checkpointer_$(model_type)_$(typeof(arch))_$(FT)"

    checkpointer = Checkpointer(model;
                                schedule = TimeInterval(1.0),
                                prefix = prefix,
                                cleanup = false,
                                verbose = true)

    simulation.output_writers[:checkpointer] = checkpointer

    @test_nowarn run!(simulation)

    @test isfile("$(prefix)_iteration0.jld2")
    @test isfile("$(prefix)_iteration1.jld2")
    @test isfile("$(prefix)_iteration2.jld2")
    @test isfile("$(prefix)_iteration3.jld2")

    grid = nothing
    model = nothing
    simulation = nothing
    checkpointer = nothing

    new_grid = RectilinearGrid(arch, FT,
                               size = (N, N, N),
                               topology = (Periodic, Bounded, Bounded),
                               extent = (L, L, L))

    if model_type == :nonhydrostatic
        new_model = NonhydrostaticModel(new_grid)
    elseif model_type == :hydrostatic
        new_model = HydrostaticFreeSurfaceModel(new_grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    end

    new_simulation = Simulation(new_model; Δt=1.0, stop_time=3.0)

    new_checkpointer = Checkpointer(new_model;
                                    schedule = TimeInterval(1.0),
                                    prefix = prefix,
                                    cleanup = false,
                                    verbose = true)

    new_simulation.output_writers[:checkpointer] = new_checkpointer

    if pickup_method == :boolean
        @test_nowarn set!(new_simulation; checkpoint=:latest)
    elseif pickup_method == :iteration
        @test_nowarn set!(new_simulation; iteration=3)
    elseif pickup_method == :filepath
        @test_nowarn set!(new_simulation; checkpoint="$(prefix)_iteration3.jld2")
    end

    @test iteration(new_simulation) == 3
    @test time(new_simulation) == 3.0

    @test new_checkpointer.schedule.actuations == 3

    rm.(glob("$(prefix)_iteration*.jld2"))

    return nothing
end
