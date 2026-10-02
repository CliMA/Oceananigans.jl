include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

@testset "JLD2 included grid metadata [$(typeof(arch))]" for arch in archs
    grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid)
    output_grid = RectilinearGrid(arch; size=(6, 4, 4), extent=(2, 1, 1))
    output_field = CenterField(output_grid)
    set!(output_field, 7)

    mktempdir() do dir
        writer = JLD2Writer(model, (; u=model.velocities.u);
                            dir, filename="included_grid.jld2",
                            schedule=IterationInterval(1), including=[:grid])

        @test !isfile(writer.filepath)
        @test_logs min_level=Logging.Warn Oceananigans.initialize!(writer, model)

        jldopen(writer.filepath, "r") do file
            @test file["grid/Nx"] == size(grid, 1)
            @test size(file["serialized/grid"]) == size(grid)
        end
    end

    mktempdir() do dir
        writer = JLD2Writer(model, (; c=output_field);
                            dir, filename="different_grid.jld2",
                            schedule=IterationInterval(1), including=[:grid])

        @test_logs min_level=Logging.Warn Oceananigans.initialize!(writer, model)
        model.clock.iteration = 1
        model.clock.time = 1
        Oceananigans.write_output!(writer, model)

        jldopen(writer.filepath, "r") do file
            @test file["grid/Nx"] == size(grid, 1)
            @test size(file["serialized/grid"]) == size(output_grid)
        end

        c = FieldTimeSeries(writer.filepath, "c")
        @test size(c.grid) == size(output_grid)
        @test c[1, 1, 1, 1] == 7
    end

    mktempdir() do dir
        writer = JLD2Writer(model, (; c=output_field);
                            dir, filename="default.jld2",
                            schedule=IterationInterval(1))

        @test_logs min_level=Logging.Warn Oceananigans.initialize!(writer, model)
        jldopen(writer.filepath, "r") do file
            @test !haskey(file, "grid/Nx")
            @test size(file["serialized/grid"]) == size(output_grid)
        end
    end

    mktempdir() do dir
        writer = JLD2Writer(model, (; scalar=(m -> 1));
                            dir, filename="no_output_grid.jld2",
                            schedule=IterationInterval(1), including=[:grid])

        @test_logs min_level=Logging.Warn Oceananigans.initialize!(writer, model)
        jldopen(writer.filepath, "r") do file
            @test file["grid/Nx"] == size(grid, 1)
            @test size(file["serialized/grid"]) == size(grid)
        end
    end

    mktempdir() do dir
        second_grid = RectilinearGrid(arch; size=(3, 4, 4), extent=(3, 1, 1))
        writer = JLD2Writer(model, (; a=output_field, b=CenterField(second_grid));
                            dir, filename="multiple_output_grids.jld2",
                            schedule=IterationInterval(1), including=[:grid])

        @test_logs min_level=Logging.Warn Oceananigans.initialize!(writer, model)
        jldopen(writer.filepath, "r") do file
            a_index = file["timeseries/a/serialized/grid_index"]
            b_index = file["timeseries/b/serialized/grid_index"]
            @test file["grid/Nx"] == size(grid, 1)
            @test size(file["serialized/grid"]) == size(grid)
            @test size(file["serialized/grid_$a_index"]) == size(output_grid)
            @test size(file["serialized/grid_$b_index"]) == size(second_grid)
        end
    end

    mktempdir() do dir
        writer = JLD2Writer(model, (; c=output_field);
                            dir, filename="split.jld2",
                            schedule=IterationInterval(1), including=[:grid],
                            file_splitting=TimeInterval(1))

        @test_logs min_level=Logging.Warn begin
            Oceananigans.initialize!(writer, model)
            Oceananigans.OutputWriters.start_next_file(model, writer)
        end

        @test writer.part == 2
        for part in 1:2
            jldopen(joinpath(dir, "split_part$part.jld2"), "r") do file
                @test file["grid/Nx"] == size(grid, 1)
                @test size(file["serialized/grid"]) == size(output_grid)
            end
        end
    end
end
