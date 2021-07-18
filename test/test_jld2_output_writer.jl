using Oceananigans.Fields: FieldSlicer

#####
##### JLD2OutputWriter tests
#####

function jld2_field_output(model)

    model.clock.iteration = 0
    model.clock.time = 0.0

    set!(model, u = (x, y, z) -> rand(),
                v = (x, y, z) -> rand(),
                w = (x, y, z) -> rand(),
                T = 0,
                S = 0)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)

    simulation.output_writers[:velocities] = JLD2OutputWriter(model, model.velocities,
                                                                    schedule = TimeInterval(1),
                                                                        dir = ".",
                                                                     prefix = "test",
                                                                      force = true)

    u₀ = CUDA.@allowscalar data(model.velocities.u)[3, 3, 3]
    v₀ = CUDA.@allowscalar data(model.velocities.v)[3, 3, 3]
    w₀ = CUDA.@allowscalar data(model.velocities.w)[3, 3, 3]

    run!(simulation)

    file = jldopen("test.jld2")

    # Data is saved without halos by default
    u₁ = file["timeseries/u/0"][3, 3, 3]
    v₁ = file["timeseries/v/0"][3, 3, 3]
    w₁ = file["timeseries/w/0"][3, 3, 3]

    close(file)

    rm("test.jld2")

    FT = typeof(u₁)

    return FT(u₀) == u₁ && FT(v₀) == v₁ && FT(w₀) == w₁
end

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
                         field_slicer = FieldSlicer(i=1:2, j=1:2:4, k=:),
                                  dir = ".",
                               prefix = "test",
                                force = true)

    run!(simulation)

    file = jldopen("test.jld2")

    u₁ = file["timeseries/u/0"]
    v₁ = file["timeseries/v/0"]
    w₁ = file["timeseries/w/0"]

    close(file)

    rm("test.jld2")

    return size(u₁) == (2, 2, 4) && size(v₁) == (2, 2, 4) && size(w₁) == (2, 2, 5)
end

function run_jld2_file_splitting_tests(arch)
    model = NonhydrostaticModel(architecture=arch, grid=RegularRectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1)))
    simulation = Simulation(model, Δt=1, stop_iteration=10)

    function fake_bc_init(file, model)
        file["boundary_conditions/fake"] = π
    end

    ow = JLD2OutputWriter(model, (u=model.velocities.u,); dir=".", prefix="test", schedule=IterationInterval(1),
                          init=fake_bc_init, including=[:grid],
                          field_slicer=nothing, array_type=Array{Float64},
                          max_filesize=200KiB, force=true)

    push!(simulation.output_writers, ow)

    # 531 KiB of output will be written which should get split into 3 files.
    run!(simulation)

    # Test that files has been split according to size as expected.
    @test filesize("test_part1.jld2") > 200KiB
    @test filesize("test_part2.jld2") > 200KiB
    @test filesize("test_part3.jld2") < 200KiB

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
end

function jld2_time_averaging_of_horizontal_averages(model)

    model.clock.iteration = 0
    model.clock.time = 0.0

    set!(model, u = (x, y, z) -> 1,
                v = (x, y, z) -> 2,
                w = (x, y, z) -> 0,
                T = (x, y, z) -> 4)

    u, v, w = model.velocities
    T, S = model.tracers

    simulation = Simulation(model, Δt=1.0, stop_iteration=5)

    u, v, w = model.velocities
    T, S = model.tracers

    average_fluxes = (wu = AveragedField(w * u, dims=(1, 2)),
                      uv = AveragedField(u * v, dims=(1, 2)),
                      wT = AveragedField(w * T, dims=(1, 2)))

    simulation.output_writers[:fluxes] = JLD2OutputWriter(model, average_fluxes,
                                                          schedule = AveragedTimeInterval(4, window=2),
                                                               dir = ".",
                                                            prefix = "test",
                                                             force = true)

    run!(simulation)

    file = jldopen("test.jld2")

    # Data is saved without halos by default
    wu = file["timeseries/wu/4"][1, 1, 3]
    uv = file["timeseries/uv/4"][1, 1, 3]
    wT = file["timeseries/wT/4"][1, 1, 3]

    close(file)

    rm("test.jld2")

    FT = eltype(model.grid)

    return wu == zero(FT) && wT == zero(FT) && uv == FT(2)
end

for arch in archs
    # Some tests can reuse this same grid and model.
    topo =(Periodic, Periodic, Bounded)
    grid = RegularRectilinearGrid(topology=topo, size=(4, 4, 4), extent=(1, 1, 1))
    model = NonhydrostaticModel(architecture=arch, grid=grid)

    @testset "JLD2 output writer [$(typeof(arch))]" begin
        @info "  Testing JLD2 output writer [$(typeof(arch))]..."

        @test jld2_field_output(model)
        @test jld2_sliced_field_output(model)
        @test jld2_sliced_field_output(model, (u = model -> model.velocities.u,
                                               v = model -> model.velocities.v,
                                               w = model -> model.velocities.w))

        run_jld2_file_splitting_tests(arch)

        @test jld2_time_averaging_of_horizontal_averages(model)
    end
end
