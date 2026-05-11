include("dependencies_for_runtests.jl")

using Oceananigans.Fields: FunctionField

#####
##### JLD2Writer tests
#####

function jld2_sliced_field_output(model, outputs=model.velocities)

    model.clock.iteration = 0
    model.clock.time = 0.0

    set!(model, u = (x, y, z) -> rand(),
                v = (x, y, z) -> rand(),
                w = (x, y, z) -> rand())

    simulation = Simulation(model, Δt=1, stop_iteration=1)

    simulation.output_writers[:velocities] = JLD2Writer(model, outputs,
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

function test_jld2_size_file_splitting(arch)
    grid = RectilinearGrid(arch, size=(16, 16, 16), extent=(1, 1, 1), halo=(1, 1, 1))
    model = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    simulation = Simulation(model, Δt=1, stop_iteration=10)

    function fake_bc_init(file, model)
        file["boundary_conditions/fake"] = π
    end

    simulation.output_writers[:ow1] = JLD2Writer(model, (; u=model.velocities.u);
                                                 dir = ".",
                                                 filename = "test.jld2",
                                                 schedule = IterationInterval(1),
                                                 init = fake_bc_init,
                                                 including = [:grid],
                                                 array_type = Array{Float64},
                                                 with_halos = true,
                                                 file_splitting = FileSizeLimit(200KiB),
                                                 overwrite_existing = true)

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

function test_jld2_async_io_matches_sync(arch)
    # Verify that async output is byte-identical to synchronous output for the same simulation.
    function build_sim(asynchronous, filename)
        grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid; tracers=:c)
        set!(model,
             u = (x, y, z) -> sin(2π * x) * cos(2π * y),
             v = (x, y, z) -> cos(2π * x) * sin(2π * y),
             w = (x, y, z) -> 0.0,
             c = (x, y, z) -> z)

        sim = Simulation(model, Δt=0.001, stop_iteration=5)
        sim.output_writers[:fields] = JLD2Writer(model,
                                                 merge(model.velocities, (; c=model.tracers.c));
                                                 dir = ".",
                                                 filename = filename,
                                                 schedule = IterationInterval(1),
                                                 overwrite_existing = true,
                                                 asynchronous = asynchronous)
        return sim
    end

    function read_timeseries_data(filepath)
        return jldopen(filepath, "r") do f
            ts = f["timeseries"]
            result = Dict{String, Dict{String, Any}}()
            for name in keys(ts)
                result[name] = Dict{String, Any}()
                for key in keys(ts[name])
                    key == "serialized" && continue
                    result[name][key] = f["timeseries/$name/$key"]
                end
            end
            return result
        end
    end

    sync_file = "test_async_io_sync.jld2"
    async_file = "test_async_io_async.jld2"
    isfile(sync_file) && rm(sync_file)
    isfile(async_file) && rm(async_file)

    sim_sync = build_sim(false, sync_file)
    sim_async = build_sim(true, async_file)

    # Type-parameter dispatch: both are JLD2Writer; only the async one matches AsyncOutputWriter.
    @test sim_sync.output_writers[:fields] isa JLD2Writer
    @test sim_async.output_writers[:fields] isa JLD2Writer
    @test sim_sync.output_writers[:fields] isa SyncOutputWriter
    @test sim_async.output_writers[:fields] isa AsyncOutputWriter
    @test !is_asynchronous(sim_sync.output_writers[:fields])
    @test  is_asynchronous(sim_async.output_writers[:fields])

    run!(sim_sync)
    run!(sim_async)

    # `run!` should have flushed pending async writes.
    @test isfile(sync_file)
    @test isfile(async_file)

    sync_data = read_timeseries_data(sync_file)
    async_data = read_timeseries_data(async_file)

    @test sort(collect(keys(sync_data))) == sort(collect(keys(async_data)))
    for name in keys(sync_data)
        @test sort(collect(keys(sync_data[name]))) == sort(collect(keys(async_data[name])))
        for it in keys(sync_data[name])
            @test sync_data[name][it] == async_data[name][it]
        end
    end

    # Calling `wait_for_async_writes!` after a run is a no-op (no pending tasks).
    wait_for_async_writes!(sim_async)
    wait_for_async_writes!(sim_async)

    rm(sync_file)
    rm(async_file)

    return nothing
end

function test_jld2_time_file_splitting(arch)
    grid = RectilinearGrid(arch, size=(16, 16, 16), extent=(1, 1, 1), halo=(1, 1, 1))
    model = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    simulation = Simulation(model, Δt=1, stop_iteration=10)

    function fake_bc_init(file, model)
        file["boundary_conditions/fake"] = π
    end

    simulation.output_writers[:ow1] = JLD2Writer(model, (; u=model.velocities.u);
                                                 dir = ".",
                                                 filename = "test",
                                                 schedule = IterationInterval(1),
                                                 init = fake_bc_init,
                                                 including = [:grid],
                                                 array_type = Array{Float64},
                                                 with_halos = true,
                                                 file_splitting = TimeInterval(3seconds),
                                                 overwrite_existing = true)

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

    simulation.output_writers[:fluxes] = JLD2Writer(model, average_fluxes,
                                                    schedule = AveragedTimeInterval(4Δt, window=2Δt),
                                                    dir = ".",
                                                    with_halos = false,
                                                    filename = "jld2_time_averaging_test.jld2",
                                                    overwrite_existing = true)

    run!(simulation)

    test_file_name = "jld2_time_averaging_test.jld2"
    file = jldopen(test_file_name)

    # Data is saved without halos, and with singleton dimensions dropped
    wu = file["timeseries/wu/4"][3]
    uv = file["timeseries/uv/4"][3]
    wT = file["timeseries/wT/4"][3]

    close(file)

    rm(test_file_name)

    FT = eltype(model.grid)

    # Note: w is not identically 0 because T = 4 introduces a buoyancy term that is
    # subsequently cancelled by a large scale pressure field.
    @test abs(wu) < eps(FT)
    @test abs(wT) < eps(FT)
    @test uv == FT(2)

    return nothing
end

function test_jld2_time_averaging(arch)
    # Test for both "nice" floating point number and one that is more susceptible
    # to rounding errors
    for Δt in (1/64, 0.01)
        # Results should be very close (rtol < 1e-5) for stride = 1.
        # stride > 2 is currently not robust and can give inconsistent
        # results due to floating number errors that can result in vanishingly
        # small timesteps, which essentially decouples the clock time from
        # the iteration number.
        # Can add stride > 1 cases to the following line to test them.
        for (stride, rtol) in zip((1), (1.e-5))
            @info "  Testing time-averaging of NetCDF outputs [$(typeof(arch))] with stride of $(stride) and relative tolerance of $(rtol)"
            topo = (Periodic, Periodic, Periodic)
            domain = (x=(0, 1), y=(0, 1), z=(0, 1))
            grid = RectilinearGrid(arch, topology=topo, size=(4, 4, 4); domain...)

            λ1(x, y, z) = x + (1 - y)^2 + tanh(z)
            λ2(x, y, z) = x + (1 - y)^2 + tanh(4z)

            Fc1(x, y, z, t, c1) = - λ1(x, y, z) * c1
            Fc2(x, y, z, t, c2) = - λ2(x, y, z) * c2

            c1_forcing = Forcing(Fc1, field_dependencies=:c1)
            c2_forcing = Forcing(Fc2, field_dependencies=:c2)

            model = NonhydrostaticModel(grid;
                                        tracers = (:c1, :c2),
                                        forcing = (c1=c1_forcing, c2=c2_forcing))

            set!(model, c1=1, c2=1)

            Δt = 0.01 # Floating point number chosen conservatively to flag rounding errors
            simulation = Simulation(model, Δt=Δt, stop_time=50Δt)

            ∫c1_dxdy = Field(Average(model.tracers.c1, dims=(1, 2)))
            ∫c2_dxdy = Field(Average(model.tracers.c2, dims=(1, 2)))

            jld2_outputs = Dict("c1" => ∫c1_dxdy, "c2" => ∫c2_dxdy)
            horizontal_average_jld2_filepath = "decay_averaged_field_test.jld2"

            simulation.output_writers[:horizontal_average] = JLD2Writer(model,
                                                                        jld2_outputs,
                                                                        schedule = TimeInterval(10Δt),
                                                                        dir = ".",
                                                                        with_halos = false,
                                                                        filename = horizontal_average_jld2_filepath,
                                                                        overwrite_existing = true)

            multiple_time_average_jld2_filepath = "decay_windowed_time_average_test.jld2"
            single_time_average_jld2_filepath = "single_decay_windowed_time_average_test.jld2"
            window = 6Δt

            single_jld2_output = Dict("c1" => ∫c1_dxdy)

            simulation.output_writers[:single_output_time_average] = JLD2Writer(model,
                                                                                single_jld2_output,
                                                                                schedule = AveragedTimeInterval(10Δt, window = window, stride = stride),
                                                                                dir = ".",
                                                                                with_halos = false,
                                                                                filename = single_time_average_jld2_filepath,
                                                                                overwrite_existing = true)

            simulation.output_writers[:multiple_output_time_average] = JLD2Writer(model,
                                                                                  jld2_outputs,
                                                                                  schedule = AveragedTimeInterval(10Δt, window = window, stride = stride),
                                                                                  dir = ".",
                                                                                  with_halos = false,
                                                                                  filename = multiple_time_average_jld2_filepath,
                                                                                  overwrite_existing = true)

            run!(simulation)

            ##### For each λ, horizontal average should evaluate to
            #####
            #####     c̄(z, t) = ∫₀¹ ∫₀¹ exp{- λ(x, y, z) * t} dx dy
            #####             = 1 / (Nx*Ny) * Σᵢ₌₁ᴺˣ Σⱼ₌₁ᴺʸ exp{- λ(i, j, k) * t}
            #####
            ##### which we can compute analytically.

            c1 = FieldTimeSeries(horizontal_average_jld2_filepath, "c1")
            c2 = FieldTimeSeries(horizontal_average_jld2_filepath, "c2")

            Nx, Ny, Nz = size(c1.grid)
            xs, ys, zs = c1.grid.xᶜᵃᵃ[1:Nx], c1.grid.yᵃᶜᵃ[1:Ny], c1.grid.z.cᵃᵃᶜ[1:Nz]

            c̄1(z, t) = 1 / (Nx * Ny) * sum(exp(-λ1(x, y, z) * t) for x in xs for y in ys)
            c̄2(z, t) = 1 / (Nx * Ny) * sum(exp(-λ2(x, y, z) * t) for x in xs for y in ys)

            for (n, t) in enumerate(c1.times)
                @test all(isapprox.(c1[1, 1, :, n], c̄1.(zs, t), rtol=rtol))
                @test all(isapprox.(c2[1, 1, :, n], c̄2.(zs, t), rtol=rtol))
            end

            # Compute time averages...
            c̄1(ts) = 1/length(ts) * sum(c̄1.(zs, t) for t in ts)
            c̄2(ts) = 1/length(ts) * sum(c̄2.(zs, t) for t in ts)

            #####
            ##### Test strided windowed time average against analytic solution
            ##### for *single* JLD2 output
            #####
            c1_single = FieldTimeSeries(single_time_average_jld2_filepath, "c1")

            window_size = Int(window/Δt)

            @info "    Testing time-averaging of a single JLD2 output [$(typeof(arch))]..."

            for (n, t) in enumerate(c1_single.times[2:end])
                averaging_times = [t - n*Δt for n in 0:stride:window_size-1 if t - n*Δt >= 0]
                @test all(isapprox.(c1_single[1, 1, :, n+1], c̄1(averaging_times), rtol=rtol, atol=rtol))
            end

            #####
            ##### Test strided windowed time average against analytic solution
            ##### for *multiple* JLD2 outputs
            #####

            c2_multiple = FieldTimeSeries(multiple_time_average_jld2_filepath, "c2")

            @info "    Testing time-averaging of multiple JLD2 outputs [$(typeof(arch))]..."

            for (n, t) in enumerate(c2_multiple.times[2:end])
                averaging_times = [t - n*Δt for n in 0:stride:window_size-1 if t - n*Δt >= 0]
                @test all(isapprox.(c2_multiple[1, 1, :, n+1], c̄2(averaging_times), rtol=rtol))
            end

            rm(horizontal_average_jld2_filepath)
            rm(single_time_average_jld2_filepath)
            rm(multiple_time_average_jld2_filepath)
        end
    end
    return nothing
end

for arch in archs
    # Some tests can reuse this same grid and model.
    topo =(Periodic, Periodic, Bounded)
    grid = RectilinearGrid(arch, topology=topo, size=(4, 4, 4), extent=(1, 1, 1))
    background_u = BackgroundField((x, y, z, t) -> 0)
    model = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S), background_fields=(u=background_u,))

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

        simulation.output_writers[:velocities] = JLD2Writer(model, vanilla_outputs,
                                                            schedule = IterationInterval(1),
                                                            dir = ".",
                                                            filename = "vanilla_jld2_test",
                                                            indices = (:, :, :),
                                                            with_halos = false,
                                                            overwrite_existing = true)

        simulation.output_writers[:sliced] = JLD2Writer(model, model.velocities,
                                                        schedule = TimeInterval(1),
                                                        indices = (1:2, 1:4, :),
                                                        with_halos = false,
                                                        dir = ".",
                                                        filename = "sliced_jld2_test",
                                                        overwrite_existing = true)

        func_outputs = (u = model -> u, v = model -> v, w = model -> w)

        simulation.output_writers[:sliced_funcs] = JLD2Writer(model, func_outputs,
                                                              schedule = TimeInterval(1),
                                                              indices = (1:2, 1:4, :),
                                                              with_halos = false,
                                                              dir = ".",
                                                              filename = "sliced_funcs_jld2_test",
                                                              overwrite_existing = true)


        simulation.output_writers[:sliced_func_fields] = JLD2Writer(model, function_and_background_fields,
                                                                    schedule = TimeInterval(1),
                                                                    indices = (1:2, 1:4, :),
                                                                    with_halos = false,
                                                                    dir = ".",
                                                                    filename = "sliced_func_fields_jld2_test",
                                                                    overwrite_existing = true)



        u₀ = @allowscalar model.velocities.u[3, 3, 3]
        v₀ = @allowscalar model.velocities.v[3, 3, 3]
        w₀ = @allowscalar model.velocities.w[3, 3, 3]

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

        @test FT(αt₀) == FT(α * t₀)
        @test FT(αt₁) == FT(α * t₁)

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
        ##### Async I/O
        #####

        test_jld2_async_io_matches_sync(arch)

        #####
        ##### Time-averaging
        #####

        test_jld2_time_averaging_of_horizontal_averages(model)

        #####
        ##### Time-averaging (same test as in NetCDFWriter)
        #####
        test_jld2_time_averaging(arch)

        # Test that free surface can be output
        grid = RectilinearGrid(arch, size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1))
        free_surface = SplitExplicitFreeSurface(substeps=10)
        model = HydrostaticFreeSurfaceModel(grid; free_surface)
        simulation = Simulation(model, Δt=1, stop_iteration=2)
        filename = "test_free_surface_output.jld2"
        ow = JLD2Writer(model, (; η=model.free_surface.displacement); filename,
                        schedule = IterationInterval(1),
                        with_halos = false,
                        overwrite_existing = true)
        simulation.output_writers[:free_surface] = ow
        run!(simulation)
        ηt = FieldTimeSeries(filename, "η")
        @test size(parent(ηt[1])) == (4, 4, 1)
    end
end
