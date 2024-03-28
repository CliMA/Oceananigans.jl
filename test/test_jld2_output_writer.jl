include("dependencies_for_runtests.jl")

using Oceananigans.Fields: FunctionField

#####
##### JLD2OutputWriter tests
#####

function jld2_sliced_field_output(model, outputs=model.velocities)

    model.clock.iteration = 0
    model.clock.time = 0.0

    set!(model, u = (x, y, z) -> rand(),
                v = (x, y, z) -> rand(),
                w = (x, y, z) -> rand())

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)

    simulation.output_writers[:velocities] =
        JLD2OutputWriter(model, outputs,
                         schedule = TimeInterval(1),
                         indices = (1:2, 1:4, :),
                         with_halos = false,
                         dir = ".",
                         filename = "test.jld2",
                         overwrite_existing = true)

    run!(simulation)

    file = jldopen("test.jld2")

    u₁ = file["timeseries/u/0"]
    v₁ = file["timeseries/v/0"]
    w₁ = file["timeseries/w/0"]

    close(file)

    rm("test.jld2")

    return size(u₁) == (2, 2, 4) && size(v₁) == (2, 2, 4) && size(w₁) == (2, 2, 5)
end

function test_jld2_file_splitting_size(arch)
    grid = RectilinearGrid(arch, size=(16, 16, 16), extent=(1, 1, 1), halo=(1, 1, 1))
    model = NonhydrostaticModel(; grid, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    simulation = Simulation(model, Δt=1, stop_iteration=10)

    function fake_bc_init(file, model)
        file["boundary_conditions/fake"] = π
    end
    ow = JLD2OutputWriter(model, (; u=model.velocities.u);
                          dir = ".",
                          filename = "test.jld2",
                          schedule = IterationInterval(1),
                          init = fake_bc_init,
                          including = [:grid],
                          array_type = Array{Float64},
                          with_halos = true,
                          file_splitting = FileSizeLimit(200KiB),
                          overwrite_existing = true)

    push!(simulation.output_writers, ow)

    # 531 KiB of output will be written which should get split into 3 files.
    run!(simulation)

    # Test that files has been split according to size as expected.
    @test filesize("test_part1.jld2") > 200KiB
    @test filesize("test_part2.jld2") > 200KiB
    @test filesize("test_part3.jld2") < 200KiB
    @test !isfile("test_part4.jld2")

    for n in string.(1:3)
        filename = "test_part$n.jld2"
        jldopen(filename, "r") do file
            # Test to make sure all files contain structs from `including`.
            @test file["grid/Nx"] == 16

            # Test to make sure all files contain info from `init` function.
            @test file["boundary_conditions/fake"] == π
        end

        # Leave test directory clean.
        rm(filename)
    end

    return nothing
end

function test_jld2_file_splitting_time(arch)
    grid = RectilinearGrid(arch, size=(16, 16, 16), extent=(1, 1, 1), halo=(1, 1, 1))
    model = NonhydrostaticModel(; grid, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    simulation = Simulation(model, Δt=1, stop_iteration=10)

    function fake_bc_init(file, model)
        file["boundary_conditions/fake"] = π
    end
    ow = JLD2OutputWriter(model, (; u=model.velocities.u);
                          dir = ".",
                          filename = "test.jld2",
                          schedule = IterationInterval(1),
                          init = fake_bc_init,
                          including = [:grid],
                          array_type = Array{Float64},
                          with_halos = true,
                          file_splitting = TimeInterval(3seconds),
                          overwrite_existing = true)

    push!(simulation.output_writers, ow)

    run!(simulation)

    for n in string.(1:3)
        filename = "test_part$n.jld2"
        jldopen(filename, "r") do file
            # Test to make sure all files contain structs from `including`.
            @test file["grid/Nx"] == 16

            # Test to make sure all files contain the same number of snapshots.
            dimlength = length(file["timeseries/t"])
            @test dimlength == 3

            # Test to make sure all files contain info from `init` function.
            @test file["boundary_conditions/fake"] == π
        end

        # Leave test directory clean.
        rm(filename)
    end
    rm("test_part4.jld2")

    return nothing
end

function test_jld2_time_averaging_of_horizontal_averages(model)

    model.clock.iteration = 0
    model.clock.time = 0.0

    u, v, w = model.velocities
    T = model.tracers.T

    u .= 1
    v .= 2
    w .= 0
    T .= 4

    Δt = 0.1
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    average_fluxes = (wu = Field(Average(w * u, dims=(1, 2))),
                      uv = Field(Average(u * v, dims=(1, 2))),
                      wT = Field(Average(w * T, dims=(1, 2))))

    simulation.output_writers[:fluxes] = JLD2OutputWriter(model, average_fluxes,
                                                          schedule = AveragedTimeInterval(4Δt, window=2Δt),
                                                          dir = ".",
                                                          filename = "jld2_time_averaging_test.jld2",
                                                          overwrite_existing = true)

    run!(simulation)

    test_file_name = "jld2_time_averaging_test.jld2"
    file = jldopen(test_file_name)

    # Data is saved without halos by default
    wu = file["timeseries/wu/4"][1, 1, 3]
    uv = file["timeseries/uv/4"][1, 1, 3]
    wT = file["timeseries/wT/4"][1, 1, 3]

    close(file)

    rm(test_file_name)

    FT = eltype(model.grid)

    @test wu == zero(FT) 
    @test wT == zero(FT) 
    @test uv == FT(2)

    return nothing
end

for arch in archs
    # Some tests can reuse this same grid and model.
    topo =(Periodic, Periodic, Bounded)
    grid = RectilinearGrid(arch, topology=topo, size=(4, 4, 4), extent=(1, 1, 1))
    background_u = BackgroundField((x, y, z, t) -> 0)
    model = NonhydrostaticModel(grid=grid, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S), background_fields=(u=background_u,))

    @testset "JLD2 output writer [$(typeof(arch))]" begin
        @info "  Testing JLD2 output writer [$(typeof(arch))]..."

        set!(model, u = (x, y, z) -> rand(),
                    v = (x, y, z) -> rand(),
                    w = (x, y, z) -> rand())

        simulation = Simulation(model, Δt=1.0, stop_iteration=1)

        # Flavors of output: functions, AbstractOperations, FunctionFields
        clock = model.clock
        α = 0.12
        test_function_field = FunctionField{Center, Center, Center}((x, y, z, t, α) -> α * t, grid; clock, parameters=α)
        function_and_background_fields = (; αt = test_function_field, background_u = model.background_fields.velocities.u)

        u, v, w = model.velocities
        operation_outputs= (u_op = 1 * u, v_op = 1 * v, w_op = 1 * w)

        vanilla_outputs = merge(model.velocities, function_and_background_fields, operation_outputs)

        simulation.output_writers[:velocities] = JLD2OutputWriter(model, vanilla_outputs,
                                                                  schedule = IterationInterval(1),
                                                                  dir = ".",
                                                                  filename = "vanilla_jld2_test.jld2",
                                                                  indices = (:, :, :),
                                                                  with_halos = false,
                                                                  overwrite_existing = true)

        simulation.output_writers[:sliced] = JLD2OutputWriter(model, model.velocities,
                                                              schedule = TimeInterval(1),
                                                              indices = (1:2, 1:4, :),
                                                              with_halos = false,
                                                              dir = ".",
                                                              filename = "sliced_jld2_test.jld2",
                                                              overwrite_existing = true)

        func_outputs = (u = model -> u, v = model -> v, w = model -> w)
        
        simulation.output_writers[:sliced_funcs] = JLD2OutputWriter(model, func_outputs,
                                                                    schedule = TimeInterval(1),
                                                                    indices = (1:2, 1:4, :),
                                                                    with_halos = false,
                                                                    dir = ".",
                                                                    filename = "sliced_funcs_jld2_test.jld2",
                                                                    overwrite_existing = true)


        simulation.output_writers[:sliced_func_fields] = JLD2OutputWriter(model, function_and_background_fields,
                                                                          schedule = TimeInterval(1),
                                                                          indices = (1:2, 1:4, :),
                                                                          with_halos = false,
                                                                          dir = ".",
                                                                          filename = "sliced_func_fields_jld2_test.jld2",
                                                                          overwrite_existing = true)



        u₀ = CUDA.@allowscalar model.velocities.u[3, 3, 3]
        v₀ = CUDA.@allowscalar model.velocities.v[3, 3, 3]
        w₀ = CUDA.@allowscalar model.velocities.w[3, 3, 3]

        run!(simulation)

        #####
        ##### Stuff was outputted
        #####

        file = jldopen("vanilla_jld2_test.jld2")

        # Data is saved without halos by default
        u₁ = file["timeseries/u/0"][3, 3, 3]
        v₁ = file["timeseries/v/0"][3, 3, 3]
        w₁ = file["timeseries/w/0"][3, 3, 3]

        # Operations
        u₁_op = file["timeseries/u_op/0"][3, 3, 3]
        v₁_op = file["timeseries/v_op/0"][3, 3, 3]
        w₁_op = file["timeseries/w_op/0"][3, 3, 3]

        # FunctionField
        αt₀ = file["timeseries/αt/0"][3, 3, 3]
        αt₁ = file["timeseries/αt/1"][3, 3, 3]
        t₀ = file["timeseries/t/0"]
        t₁ = file["timeseries/t/1"]

        close(file)

        rm("vanilla_jld2_test.jld2")

        FT = typeof(u₁)

        @test FT(u₀) == u₁
        @test FT(v₀) == v₁
        @test FT(w₀) == w₁

        @test FT(u₀) == u₁_op
        @test FT(v₀) == v₁_op
        @test FT(w₀) == w₁_op

        @test FT(αt₀) == α * t₀
        @test FT(αt₁) == α * t₁

        #####
        ##### Field slicing
        #####

        function test_field_slicing(test_file_name, variables, sizes...)
            file = jldopen(test_file_name)

            for (i, variable) in enumerate(variables)
                var₁ = file["timeseries/$variable/0"]
                @test size(var₁) == sizes[i]
            end

            close(file)
            rm(test_file_name)
        end

        test_field_slicing("sliced_jld2_test.jld2", ("u", "v", "w"), (2, 4, 4), (2, 4, 4), (2, 4, 5))
        test_field_slicing("sliced_funcs_jld2_test.jld2", ("u", "v", "w"), (4, 4, 4), (4, 4, 4), (4, 4, 5))
        test_field_slicing("sliced_func_fields_jld2_test.jld2", ("αt", "background_u"), (2, 4, 4), (2, 4, 4))
        
        ####
        #### File splitting
        ####

        test_jld2_size_file_splitting(arch)
        test_jld2_time_file_splitting(arch)

        #####
        ##### Time-averaging
        #####

        test_jld2_time_averaging_of_horizontal_averages(model)
    end
end
