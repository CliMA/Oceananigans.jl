include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using NCDatasets
using Oceananigans.Fields: PointwiseInterpolation, RemappedField, remapping
using Oceananigans.OutputWriters: output_lookup_key

# A linear function is interpolated exactly between grids that share their coordinates
linear(x, y, z) = x + 2y + 3z

function remapping_test_setup(arch)
    grid = RectilinearGrid(arch; size=(8, 8, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    coarse_grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
    model = NonhydrostaticModel(grid; tracers = :c)
    set!(model, c = linear)
    return grid, coarse_grid, model
end

@testset "Remapped fields [$(typeof(arch))]" for arch in archs
    grid, coarse_grid, model = remapping_test_setup(arch)
    c = model.tracers.c
    u, v, w = model.velocities

    cᶜ = remapped_field(c, coarse_grid)
    @test cᶜ isa RemappedField
    @test cᶜ.grid === coarse_grid
    @test location(cᶜ) == (Center, Center, Center)
    @test remapping(cᶜ, c) isa PointwiseInterpolation
    @test occursin("Remap of", summary(cᶜ.operand))

    expected = CenterField(coarse_grid)
    set!(expected, linear)
    @test interior(cᶜ) ≈ interior(expected)
    @test interior(cᶜ) == interior(interpolate!(CenterField(coarse_grid), c))

    # The field is recomputed from its source
    set!(c, (x, y, z) -> 2 * linear(x, y, z))
    compute!(cᶜ)
    @test interior(cᶜ) ≈ 2 .* interior(expected)

    # Operations are materialized, and the vertical window of the source is kept
    @test interior(remapped_field(c / 2, coarse_grid)) ≈ interior(expected)
    cᵏ = remapped_field(Field(c; indices=(:, :, 2)), coarse_grid)
    @test size(cᵏ) == (4, 4, 1)
    @test interior(cᵏ) ≈ 2 .* interior(expected, :, :, 2:2)
    @test interior(remapped_field(c, coarse_grid; indices=(:, :, 3))) ≈ 2 .* interior(expected, :, :, 3:3)
    @test interior(remapped_field(Field(c; indices=(:, :, 2)), coarse_grid; indices=(:, :, 1:3))) ≈ 2 .* interior(expected, :, :, 2:2)

    @test_throws ArgumentError remapped_field(Field(c; indices=(1:2, :, :)), coarse_grid)
    @test_throws ArgumentError remapped_field((u, v), coarse_grid)
end

@testset "Remapped output [$(typeof(arch))]" for arch in archs
    grid, coarse_grid, model = remapping_test_setup(arch)
    c = model.tracers.c
    simulation = Simulation(model; Δt=1, stop_iteration=2, verbose=false)

    expected = CenterField(coarse_grid)
    set!(expected, linear)
    expected² = remapped_field(c^2, coarse_grid) # the interpolation of a quadratic is not the square of the interpolation
    already_coarse = CenterField(coarse_grid)
    set!(already_coarse, 1)

    dir = mktempdir()
    outputs = (; c, c² = c^2, cᵏ = Field(c; indices=(:, :, 2)), already_coarse, clock_time = model -> model.clock.time)
    message = r"will not be remapped"
    simulation.output_writers[:jld2] = @test_logs (:warn, message) JLD2Writer(model, outputs; dir, grid = coarse_grid,
                                                                             filename = "remapped", with_halos = false,
                                                                             schedule = IterationInterval(1))

    simulation.output_writers[:slice] = JLD2Writer(model, (; c); dir, grid = coarse_grid, indices = (:, :, 3),
                                                   filename = "slice", with_halos = false, schedule = IterationInterval(1))

    simulation.output_writers[:average] = JLD2Writer(model, (; c, c² = c^2); dir, grid = coarse_grid, filename = "average",
                                                     with_halos = false, schedule = AveragedTimeInterval(2))

    # A NetCDF file holds one set of dimensions per grid, so a level is written with the writer's `indices`
    netcdf_outputs = (; c, c² = c^2, already_coarse)
    simulation.output_writers[:netcdf] = NetCDFWriter(model, netcdf_outputs; dir, grid = coarse_grid,
                                                      filename = "remapped", schedule = IterationInterval(1))
    simulation.output_writers[:netcdf_level] = NetCDFWriter(model, (; c); dir, grid = coarse_grid, indices = (:, :, 2),
                                                            filename = "level", schedule = IterationInterval(1))

    # Every output is a field on the coarse grid, except the function, and fields already on it are not remapped
    for writer in (simulation.output_writers[:jld2], simulation.output_writers[:netcdf])
        for name in (:c, :c²)
            output = writer.outputs[output_lookup_key(writer, name)]
            @test output isa RemappedField
            @test output.grid === coarse_grid
        end
        output = writer.outputs[output_lookup_key(writer, :already_coarse)]
        @test !(output isa RemappedField)
        @test output.grid === coarse_grid
    end
    @test simulation.output_writers[:jld2].outputs.cᵏ isa RemappedField

    run!(simulation)

    jldopen(joinpath(dir, "remapped.jld2")) do file
        @test file["serialized/grid"] == on_architecture(CPU(), coarse_grid)
        @test file["timeseries/c/2"] ≈ Array{Float32}(interior(expected))
        @test file["timeseries/c²/2"] ≈ Array{Float32}(interior(expected²))
        @test file["timeseries/cᵏ/2"] ≈ Array{Float32}(interior(expected, :, :, 2:2))
        @test file["timeseries/cᵏ/serialized/indices"] == (1:4, 1:4, 2:2)
        @test all(file["timeseries/already_coarse/2"] .== 1)
        @test file["timeseries/clock_time/2"] == 2
    end

    time_series = FieldTimeSeries(joinpath(dir, "remapped.jld2"), "c")
    @test time_series.grid == on_architecture(CPU(), coarse_grid)
    @test size(time_series.grid) == (4, 4, 4)
    @test interior(time_series[end]) ≈ interior(expected)

    jldopen(joinpath(dir, "slice.jld2")) do file
        @test size(file["timeseries/c/2"]) == (4, 4, 1)
        @test file["timeseries/c/2"] ≈ Array{Float32}(interior(expected, :, :, 3:3))
    end

    # The tracer is steady, so its time average is its remapping
    jldopen(joinpath(dir, "average.jld2")) do file
        @test file["timeseries/c/2"] ≈ Array{Float32}(interior(expected))
        @test file["timeseries/c²/2"] ≈ Array{Float32}(interior(expected²))
    end

    NCDataset(joinpath(dir, "remapped.nc")) do ds
        @test length(ds["x_caa"]) == 4
        @test ds["x_caa"][:] ≈ Array(xnodes(coarse_grid, Center()))
        @test ds["c"][:, :, :, end] ≈ Array{Float32}(interior(expected))
    end

    NCDataset(joinpath(dir, "level.nc")) do ds
        @test length(ds["z_aac"]) == 1
        @test ds["c"][:, :, 1, end] ≈ Array{Float32}(interior(expected, :, :, 2))
    end

    rm(dir; recursive = true)
end
