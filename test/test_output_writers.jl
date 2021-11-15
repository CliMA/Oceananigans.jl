using Statistics
using NCDatasets
using Test

using Oceananigans
using Oceananigans.Diagnostics
using Oceananigans.Fields
using Oceananigans.OutputWriters

using Dates: Millisecond
using Oceananigans: write_output!
using Oceananigans.BoundaryConditions: PBC, FBC, ZFBC, ContinuousBoundaryFunction
using Oceananigans.TimeSteppers: update_state!

include("utils_for_runtests.jl")

archs = test_architectures()

#####
##### WindowedTimeAverage tests
#####

function instantiate_windowed_time_average(model)

    set!(model, u = (x, y, z) -> rand())

    u, v, w = model.velocities

    u₀ = similar(interior(u))
    u₀ .= interior(u)

    wta = WindowedTimeAverage(model.velocities.u, schedule=AveragedTimeInterval(10, window=1))

    return all(wta(model) .== u₀)
end

function time_step_with_windowed_time_average(model)

    model.clock.iteration = 0
    model.clock.time = 0.0

    set!(model, u=0, v=0, w=0, T=0, S=0)

    wta = WindowedTimeAverage(model.velocities.u, schedule=AveragedTimeInterval(4, window=2))

    simulation = Simulation(model, Δt=1.0, stop_time=4.0)
    simulation.diagnostics[:u_avg] = wta
    run!(simulation)

    return all(wta(model) .== interior(model.velocities.u))
end

#####
##### Dependency adding tests
#####

function dependencies_added_correctly!(model, windowed_time_average, output_writer)

    model.clock.iteration = 0
    model.clock.time = 0.0

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    push!(simulation.output_writers, output_writer)
    run!(simulation)

    return windowed_time_average ∈ values(simulation.diagnostics)
end

function test_dependency_adding(model)

    windowed_time_average = WindowedTimeAverage(model.velocities.u, schedule=AveragedTimeInterval(4, window=2))

    output = Dict("time_average" => windowed_time_average)
    attributes = Dict("time_average" => Dict("longname" => "A time average",  "units" => "arbitrary"))
    dimensions = Dict("time_average" => ("xF", "yC", "zC"))

    # JLD2 dependencies test
    jld2_output_writer = JLD2OutputWriter(model, output, schedule=TimeInterval(4), dir=".", prefix="test", force=true)

    @test dependencies_added_correctly!(model, windowed_time_average, jld2_output_writer)

    # NetCDF dependency test
    netcdf_output_writer = NetCDFOutputWriter(model, output,
                                              schedule = TimeInterval(4),
                                              filepath = "test.nc",
                                              output_attributes = attributes,
                                              dimensions = dimensions)

    @test dependencies_added_correctly!(model, windowed_time_average, netcdf_output_writer)

    rm("test.nc")

    return nothing
end

#####
##### Test time averaging of output
#####

function test_windowed_time_averaging_simulation(model)

    jld_filename1 = "test_windowed_time_averaging1"
    jld_filename2 = "test_windowed_time_averaging2"

    model.clock.iteration = model.clock.time = 0
    simulation = Simulation(model, Δt=1.0, stop_iteration=0)

    jld2_output_writer = JLD2OutputWriter(model, model.velocities,
                                          schedule = AveragedTimeInterval(π, window=1),
                                          prefix = jld_filename1,
                                          force = true)

    # https://github.com/Alexander-Barth/NCDatasets.jl/issues/105
    nc_filepath1 = "windowed_time_average_test1.nc"
    nc_outputs = Dict(string(name) => field for (name, field) in pairs(model.velocities))
    nc_output_writer = NetCDFOutputWriter(model, nc_outputs, filepath=nc_filepath1,
                                          schedule = AveragedTimeInterval(π, window=1))

    jld2_outputs_are_time_averaged = Tuple(typeof(out) <: WindowedTimeAverage for out in jld2_output_writer.outputs)
      nc_outputs_are_time_averaged = Tuple(typeof(out) <: WindowedTimeAverage for out in values(nc_output_writer.outputs))

    @test all(jld2_outputs_are_time_averaged)
    @test all(nc_outputs_are_time_averaged)

    # Test that the collection does *not* start when a simulation is initialized
    # when time_interval ≠ time_averaging_window
    simulation.output_writers[:jld2] = jld2_output_writer
    simulation.output_writers[:nc] = nc_output_writer

    run!(simulation)

    jld2_u_windowed_time_average = simulation.output_writers[:jld2].outputs.u
    nc_w_windowed_time_average = simulation.output_writers[:nc].outputs["w"]

    @test !(jld2_u_windowed_time_average.schedule.collecting)
    @test !(nc_w_windowed_time_average.schedule.collecting)

    # Test that time-averaging is finalized prior to output even when averaging over
    # time_window is not fully realized. For this, step forward to a time at which
    # collection should start. Note that time_interval = π and time_window = 1.0.
    simulation.Δt = 1.5
    simulation.stop_iteration = 2
    run!(simulation) # model.clock.time = 3.0, just before output but after average-collection.

    @test jld2_u_windowed_time_average.schedule.collecting
    @test nc_w_windowed_time_average.schedule.collecting

    # Step forward such that time_window is not reached, but output will occur.
    simulation.Δt = π - 3 + 0.01 # ≈ 0.15 < 1.0
    simulation.stop_iteration = 3
    run!(simulation) # model.clock.time ≈ 3.15, after output

    @test jld2_u_windowed_time_average.schedule.previous_interval_stop_time ==
        model.clock.time - rem(model.clock.time, jld2_u_windowed_time_average.schedule.interval)

    @test nc_w_windowed_time_average.schedule.previous_interval_stop_time ==
        model.clock.time - rem(model.clock.time, nc_w_windowed_time_average.schedule.interval)

    # Test that collection does start when a simulation is initialized and
    # time_interval == time_averaging_window
    model.clock.iteration = model.clock.time = 0

    simulation.output_writers[:jld2] = JLD2OutputWriter(model, model.velocities,
                                                        schedule = AveragedTimeInterval(π, window=π),
                                                          prefix = jld_filename2,
                                                           force = true)

    nc_filepath2 = "windowed_time_average_test2.nc"
    nc_outputs = Dict(string(name) => field for (name, field) in pairs(model.velocities))
    simulation.output_writers[:nc] = NetCDFOutputWriter(model, nc_outputs, filepath=nc_filepath2,
                                                        schedule=AveragedTimeInterval(π, window=π))

    run!(simulation)

    @test simulation.output_writers[:jld2].outputs.u.schedule.collecting
    @test simulation.output_writers[:nc].outputs["w"].schedule.collecting

    rm(nc_filepath1)
    rm(nc_filepath2)
    rm(jld_filename1 * ".jld2")
    rm(jld_filename2 * ".jld2")

    return nothing
end

#####
##### Run output writer tests!
#####

@testset "Output writers" begin
    @info "Testing output writers..."

    for arch in archs
        # Some tests can reuse this same grid and model.
        topo = (Periodic, Periodic, Bounded)
        grid = RectilinearGrid(topology=topo, size=(4, 4, 4), extent=(1, 1, 1))
        model = NonhydrostaticModel(architecture=arch, grid=grid)

        @testset "WindowedTimeAverage [$(typeof(arch))]" begin
            @info "  Testing WindowedTimeAverage [$(typeof(arch))]..."
            @test instantiate_windowed_time_average(model)
            @test time_step_with_windowed_time_average(model)
            @test_throws ArgumentError AveragedTimeInterval(1.0, window=1.1)
        end
    end

    include("test_netcdf_output_writer.jl")
    include("test_jld2_output_writer.jl")
    include("test_checkpointer.jl")

    for arch in archs
        topo =(Periodic, Periodic, Bounded)
        grid = RectilinearGrid(topology=topo, size=(4, 4, 4), extent=(1, 1, 1))
        model = NonhydrostaticModel(architecture=arch, grid=grid)

        @testset "Dependency adding [$(typeof(arch))]" begin
            @info "    Testing dependency adding [$(typeof(arch))]..."
            test_dependency_adding(model)
        end

        @testset "Time averaging of output [$(typeof(arch))]" begin
            @info "    Testing time averaging of output [$(typeof(arch))]..."
            test_windowed_time_averaging_simulation(model)
        end
    end
end
