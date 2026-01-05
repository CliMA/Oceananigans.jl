include("dependencies_for_runtests.jl")

using Oceananigans.OutputWriters: AveragedSpecifiedTimes, WindowedTimeAverage, outside_window, end_of_window, advance_time_average!
using Oceananigans.Utils: SpecifiedTimes
using Oceananigans: write_output!
using Dates
using Statistics
using NCDatasets

#####
##### Test time averaging of output with AveragedSpecifiedTimes
#####

function test_averaged_specified_times_simulation(model)
    jld_filename1 = "test_averaged_specified_times1.jld2"
    jld_filename2 = "test_averaged_specified_times2.jld2"

    model.clock.iteration = model.clock.time = 0
    simulation = Simulation(model, Δt=1.0, stop_iteration=0)

    times = [π, 2π, 3π]
    window = 1.0
    
    jld2_output_writer = JLD2Writer(model, model.velocities,
                                    schedule = AveragedSpecifiedTimes(times; window),
                                    filename = jld_filename1,
                                    overwrite_existing = true)

    nc_filepath1 = "averaged_specified_times_test1.nc"
    nc_output_writer = NetCDFWriter(model, model.velocities,
                                    filename = nc_filepath1,
                                    schedule = AveragedSpecifiedTimes(times; window),
                                    overwrite_existing = true)

    jld2_outputs_are_time_averaged = Tuple(typeof(out) <: WindowedTimeAverage for out in jld2_output_writer.outputs)
    nc_outputs_are_time_averaged = Tuple(typeof(out) <: WindowedTimeAverage for out in values(nc_output_writer.outputs))

    @test all(jld2_outputs_are_time_averaged)
    @test all(nc_outputs_are_time_averaged)

    # Test that the collection does *not* start when a simulation is initialized
    # when specified time is far in the future
    simulation.output_writers[:jld2] = jld2_output_writer
    simulation.output_writers[:nc] = nc_output_writer

    run!(simulation)

    jld2_u_windowed_time_average = simulation.output_writers[:jld2].outputs.u
    nc_w_windowed_time_average = simulation.output_writers[:nc].outputs["w"]

    @test !(jld2_u_windowed_time_average.schedule.collecting)
    @test !(nc_w_windowed_time_average.schedule.collecting)

    # Test that time-averaging starts when we enter the window
    simulation.Δt = 1.5
    simulation.stop_iteration = 2
    run!(simulation) # model.clock.time = 3.0, just after average-collection starts (π - 1.0 ≈ 2.14)

    @test jld2_u_windowed_time_average.schedule.collecting
    @test nc_w_windowed_time_average.schedule.collecting

    # Step forward such that we reach the first output time
    simulation.Δt = π - 3 + 0.01
    simulation.stop_iteration = 3
    run!(simulation) # model.clock.time ≈ π, after first output

    model.clock.iteration = model.clock.time = 0

    immediate_times = [π, 2π]
    # Use window size that extends to simulation start but doesn't overlap
    window_size = π  # Non-overlapping: window 1: [0,π], window 2: [π,2π]

    simulation.output_writers[:jld2] = JLD2Writer(model, model.velocities,
                                                  schedule = AveragedSpecifiedTimes(immediate_times; window=window_size),
                                                  filename = jld_filename2,
                                                  overwrite_existing = true)

    nc_filepath2 = "averaged_specified_times_test2.nc"

    simulation.output_writers[:nc] = NetCDFWriter(model, model.velocities,
                                                  filename = nc_filepath2,
                                                  schedule = AveragedSpecifiedTimes(immediate_times; window=window_size))

    run!(simulation)

    @test simulation.output_writers[:jld2].outputs.u.schedule.collecting
    @test simulation.output_writers[:nc].outputs["w"].schedule.collecting

    rm(nc_filepath1)
    rm(nc_filepath2)
    rm(jld_filename1)
    rm(jld_filename2)

    return nothing
end

#####
##### Test error handling
#####

function test_averaged_specified_times_overlapping_windows()
    # Test that mismatched lengths throw an error
    times = [1.0, 2.0, 3.0]
    window = [0.5, 0.3]  # Wrong length

    @test_throws ArgumentError AveragedSpecifiedTimes(times; window)

    # Test overlapping windows with scalar window
    times_overlap = [1.0, 1.5, 3.0]
    window_scalar = 1.0  # Windows overlap: gap between 1.0 and 1.5 is only 0.5

    @test_throws ArgumentError AveragedSpecifiedTimes(times_overlap; window=window_scalar)

    # Test overlapping windows with vector of numeric windows
    times_numeric = [1.0, 2.0, 4.0]
    windows_numeric = [0.5, 1.5, 1.0]  # Window 2 starts at 0.5, overlaps with time 1.0

    @test_throws ArgumentError AveragedSpecifiedTimes(times_numeric; window=windows_numeric)

    # Test overlapping windows with DateTime and Period windows
    times_datetime = [DateTime(2000, 1, 10), DateTime(2000, 1, 15), DateTime(2000, 2, 1)]
    windows_periods = [Day(5), Day(10), Day(5)]  # Window 2 starts at Jan 5, overlaps with Jan 10

    @test_throws ArgumentError AveragedSpecifiedTimes(times_datetime; window=windows_periods)

    # Test valid non-overlapping windows (should not throw)
    times_valid = [1.0, 3.0, 6.0]
    windows_valid = [0.5, 1.0, 1.5]

    @test AveragedSpecifiedTimes(times_valid; window=windows_valid) isa AveragedSpecifiedTimes

    # Test valid DateTime non-overlapping windows (should not throw)
    times_datetime_valid = [DateTime(2000, 1, 10), DateTime(2000, 1, 20), DateTime(2000, 2, 15)]
    windows_periods_valid = [Day(3), Day(5), Day(10)]

    @test AveragedSpecifiedTimes(times_datetime_valid; window=windows_periods_valid) isa AveragedSpecifiedTimes

    return nothing
end

#####
##### Quantitative validation tests
#####

function test_averaging_scalar_window(model)
    @info "  Testing quantitative correctness with scalar window..."

    # Use generic Float type to support both Float32 and Float64
    FT = eltype(model.grid)

    # Reset model state
    model.clock.iteration = 0
    model.clock.time = 0
    set!(model, u=0, v=0, w=0)

    # Test with discrete ramp: u will be set to iteration number
    # With dt=0.1, at t=1.0 we have 10 iterations (0 through 9)
    # Left Riemann sum: ⟨u⟩ = (1/1.0) × 0.1 × (0+1+2+...+9) = 4.5

    dt = 0.1
    output_time = 1
    window = 1

    jld_filename = "test_quantitative_scalar.jld2"

    jld2_output_writer = JLD2Writer(model, (; u=model.velocities.u),
                                    schedule = AveragedSpecifiedTimes([output_time]; window),
                                    filename = jld_filename,
                                    overwrite_existing = true)

    simulation = Simulation(model, Δt=dt, stop_time=output_time)
    simulation.output_writers[:jld2] = jld2_output_writer

    # Set u to iteration number at each timestep
    function set_u_to_iteration!(sim)
        iter = sim.model.clock.iteration
        set!(sim.model, u=iter)
        return nothing
    end

    simulation.callbacks[:set_u] = Callback(set_u_to_iteration!, IterationInterval(1))

    run!(simulation)

    # Read back the results
    u_ts = FieldTimeSeries(jld_filename, "u")
    recorded_u = interior(u_ts, 1, 1, 1, :)

    # Expected: left Riemann sum
    # Execution order: diagnostics run, then callbacks run
    # So at each iteration i, diagnostic samples the value set by callback at iteration i-1
    # We sample u=0,1,2,...,9 (values from iterations 0-9)
    # ⟨u⟩ = 0.1 × (0+1+2+...+9) = 0.1 × 45 = 4.5
    expected = dt * sum(FT(i) for i in 0:9)

    # Tolerance accounting for floating point errors
    tolerance = FT(100) * eps(FT)

    # Skip first value (initial condition at t=0)
    @test abs(recorded_u[2] - expected) < tolerance

    rm(jld_filename)

    return nothing
end

function test_averaging_varying_windows(model)
    @info "  Testing quantitative correctness with varying windows..."

    # Use generic Float type to support both Float32 and Float64
    FT = eltype(model.grid)

    # Reset model state
    model.clock.iteration = 0
    model.clock.time = 0
    set!(model, u=0, v=0, w=0)

    # Test with discrete ramp at multiple output times with varying windows
    dt = FT(0.1)
    # Use non-overlapping windows
    # Window 1: [0.0, 0.5], Window 2: [0.6, 1.0]
    output_times = [FT(0.5), FT(1.0)]
    windows = [FT(0.5), FT(0.4)]  # Non-overlapping after sorting

    # Calculate expected values:
    # Window 1 at t=0.5 with window=0.5: samples from t=0.0 to t=0.5
    #   Iterations 0-4 (times 0.0, 0.1, 0.2, 0.3, 0.4)
    #   u values: 0, 1, 2, 3, 4
    #   Left Riemann sum: dt × (0+1+2+3+4) = 0.1 × 10 = 1.0
    #   BUT: averaging normalizes by window duration, so we need to recalculate:
    #   At each diagnostic call, we accumulate (result * T_prev + integrand * dt) / T_current
    #   This gives us the average over the window, which is: (0.1 × 10) / 0.5 = 2.0
    #
    # Window 2 at t=1.0 with window=0.4: samples from t=0.6 to t=1.0
    #   Iterations 6-9 (times 0.6, 0.7, 0.8, 0.9)
    #   u values: 6, 7, 8, 9
    #   Left Riemann sum: dt × (6+7+8+9) = 0.1 × 30 = 3.0
    #   Average: (0.1 × 30) / 0.4 = 7.5

    expected_values = [FT(2.0), FT(7.5)]

    jld_filename = "test_quantitative_varying.jld2"

    jld2_output_writer = JLD2Writer(model, (; u=model.velocities.u),
                                    schedule = AveragedSpecifiedTimes(output_times; window=windows),
                                    filename = jld_filename,
                                    overwrite_existing = true)

    simulation = Simulation(model, Δt=dt, stop_time=FT(1.0))
    simulation.output_writers[:jld2] = jld2_output_writer

    # Set u to iteration number at each timestep
    function set_u_to_iteration!(sim)
        iter = sim.model.clock.iteration
        set!(sim.model, u=FT(iter))
        return nothing
    end

    simulation.callbacks[:set_u] = Callback(set_u_to_iteration!, IterationInterval(1))

    run!(simulation)

    u_ts = FieldTimeSeries(jld_filename, "u")
    recorded_u = interior(u_ts, 1, 1, 1, :)

    # Skip first value (initial condition at t=0)
    @test length(u_ts.times) == length(output_times) + 1
    @test all(isapprox.(u_ts.times[2:end], output_times, rtol=1e-5))

    @test all(isapprox.(recorded_u[2:end], expected_values))

    rm(jld_filename)

    return nothing
end

function test_averaging_datetime_windows(model)
    @info "  Testing DateTime with Month windows (date-dependent)..."

    # This tests the critical feature that Month(1) produces different window sizes
    # for different months (28-31 days), which is the intended behavior
    # Using 3 months: Jan (31 days), Feb (29 days in leap year 2000), Mar (31 days)

    start_time = DateTime(2000, 1, 1)
    stop_time = start_time + Month(3)  # Just 3 months instead of full year

    # Complete times includes the start
    complete_times = start_time:Month(1):stop_time
    # Output times start at end of first month
    times = start_time + Month(1):Month(1):stop_time
    window = Month(1)

    # Create DateTime model using the passed grid's architecture
    grid = model.grid
    clock = Clock(time=start_time)
    datetime_model = HydrostaticFreeSurfaceModel(; grid, clock)

    simulation = Simulation(datetime_model, Δt=Day(1), stop_time=stop_time)

    jld_filename = "test_datetime_month_windows.jld2"

    jld2_output_writer = JLD2Writer(datetime_model, datetime_model.velocities,
                                    schedule = AveragedSpecifiedTimes(times; window),
                                    filename = jld_filename,
                                    overwrite_existing = true)

    # Set u to the day number from start
    function set_u_to_day!(sim)
        day_value = Float64(Day(sim.model.clock.time - start_time).value)
        set!(sim.model, u=day_value, v=day_value, w=day_value)
        return nothing
    end

    set!(datetime_model, u=0.0, v=0.0, w=0.0)

    simulation.callbacks[:set_u] = Callback(set_u_to_day!, IterationInterval(1))
    simulation.output_writers[:jld2] = jld2_output_writer

    run!(simulation)

    expected_averages = Float64[]

    for i in 1:(length(complete_times)-1)
        month_start = complete_times[i]
        month_end = complete_times[i+1]

        # Days from start to month boundaries
        day_start = Day(month_start - start_time).value
        day_end = Day(month_end - start_time).value

        # Average of day numbers in this month: (first + last) / 2
        avg = (day_start + day_end - 1) / 2
        push!(expected_averages, avg)
    end

    # Read back and verify
    u_ts = FieldTimeSeries(jld_filename, "u")
    recorded_u = interior(u_ts, 1, 1, 1, :)

    # Check that we got the right number of outputs
    @test length(u_ts.times) == length(times) + 1  # +1 for initial condition

    # Verify the averaged values match expected (skip first value which is initial condition)
    @test all(isapprox.(recorded_u[2:end], expected_averages, rtol=1e-5))

    # Verify that Month(1) produces different window sizes
    # Jan has 31 days, Feb has 29 days (2000 is leap year), Mar has 31 days
    # Check that Feb average is correct for 29-day window (not 31 or 30)
    # Feb window: days 31-59, avg should be (31+59)/2 = 45.0
    # If it were 30 days: avg would be (30+59)/2 = 44.5
    # If it were 31 days: avg would be (31+61)/2 = 46.0
    @test isapprox(recorded_u[3], 45.0, rtol=1e-5)  # Feb output at index 3

    rm(jld_filename)

    return nothing
end

function test_averaging_datetime_varying_period_windows(model)
    @info "  Testing DateTime with varying Period windows..."

    # Test with mixed Period types (Days and Month) to ensure variable-length windows work
    # Window 1: 10 days, Window 2: 1 Month (31 days in Jan), Window 3: 15 days

    start_time = DateTime(2000, 1, 1)

    # Output times: Jan 11, Feb 11, Feb 26
    times = [DateTime(2000, 1, 11), DateTime(2000, 2, 11), DateTime(2000, 2, 26)]
    windows = [Day(10), Month(1), Day(15)]  # Mixed Period types

    stop_time = times[end]

    # Create DateTime model using the passed grid's architecture
    grid = model.grid
    clock = Clock(time=start_time)
    datetime_model = HydrostaticFreeSurfaceModel(; grid, clock)

    simulation = Simulation(datetime_model, Δt=Day(1), stop_time=stop_time)

    jld_filename = "test_datetime_varying_periods.jld2"

    jld2_output_writer = JLD2Writer(datetime_model, datetime_model.velocities,
                                    schedule = AveragedSpecifiedTimes(times; window=windows),
                                    filename = jld_filename,
                                    overwrite_existing = true)

    # Set u to the day number from start
    function set_u_to_day!(sim)
        day_value = Float64(Day(sim.model.clock.time - start_time).value)
        set!(sim.model, u=day_value, v=day_value, w=day_value)
        return nothing
    end

    set!(datetime_model, u=0.0, v=0.0, w=0.0)

    simulation.callbacks[:set_u] = Callback(set_u_to_day!, IterationInterval(1))
    simulation.output_writers[:jld2] = jld2_output_writer

    run!(simulation)

    # Calculate expected averages manually:
    # Window 1: Jan 1 - Jan 11 (days 0-10), avg = 5.0
    # Window 2: Jan 11 - Feb 11 (days 10-41, since Jan has 31 days), avg = 25.5
    # Window 3: Feb 11 - Feb 26 (days 41-56), avg = 48.5

    window_starts = [start_time, DateTime(2000, 1, 11), DateTime(2000, 2, 11)]
    expected_averages = Float64[]

    for i in 1:length(times)
        window_start = times[i] - windows[i]
        window_end = times[i]

        day_start = Day(window_start - start_time).value
        day_end = Day(window_end - start_time).value

        # Average of day numbers in this window: (first + last) / 2
        avg = (day_start + day_end - 1) / 2
        push!(expected_averages, avg)
    end

    # Read back and verify
    u_ts = FieldTimeSeries(jld_filename, "u")
    recorded_u = interior(u_ts, 1, 1, 1, :)

    # Check that we got the right number of outputs
    @test length(u_ts.times) == length(times) + 1  # +1 for initial condition

    # Verify the averaged values match expected (skip first value which is initial condition)
    @test all(isapprox.(recorded_u[2:end], expected_averages, rtol=1e-5))

    # Verify that different Period types work correctly
    # The three windows have lengths: 10 days, 31 days, 15 days
    # So the averages should all be different
    @test length(unique(diff(recorded_u[2:end]))) == 2  # Two unique differences

    rm(jld_filename)

    return nothing
end

#####
##### Test runtime validation of averaging windows
#####

function test_averaged_specified_times_runtime_validation()
    @info "  Testing runtime validation for AveragedSpecifiedTimes..."

    # Test 1: Valid case - window doesn't extend before start
    @testset "Valid window (doesn't extend before start)" begin
        grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid)

        # First specified time at 5.0, window is 2.0 (goes back to 3.0)
        # Simulation starts at 0.0, so this should be fine
        schedule = AveragedSpecifiedTimes([5.0, 10.0], window=2.0)
        simulation = Simulation(model, Δt=0.1, stop_time=10.0)
        simulation.output_writers[:test] = JLD2Writer(model, model.velocities,
                                                       filename="test_valid_runtime.jld2",
                                                       schedule=schedule,
                                                       overwrite_existing=true)

        # Should not throw an error
        @test_nowarn run!(simulation)

        rm("test_valid_runtime.jld2", force=true)
    end

    # Test 2: Invalid case - window extends before simulation start
    @testset "Invalid window (extends before start)" begin
        grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid)

        # First specified time at 2.0, window is 5.0 (goes back to -3.0)
        # Simulation starts at 0.0, so this should fail at runtime
        schedule = AveragedSpecifiedTimes([2.0, 10.0], window=5.0)
        simulation = Simulation(model, Δt=0.1, stop_time=10.0)
        simulation.output_writers[:test] = JLD2Writer(model, model.velocities,
                                                       filename="test_invalid_runtime.jld2",
                                                       schedule=schedule,
                                                       overwrite_existing=true)

        # Should throw ArgumentError at runtime during initialize!
        @test_throws ArgumentError run!(simulation)

        rm("test_invalid_runtime.jld2", force=true)
    end

    # Test 3: Invalid case with vector windows
    @testset "Invalid vector windows (first window extends before start)" begin
        grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid)

        # First window: time=2.0, window=5.0 -> starts at -3.0 (invalid)
        # Second window: time=10.0, window=2.0 -> starts at 8.0 (valid)
        schedule = AveragedSpecifiedTimes([2.0, 10.0], window=[5.0, 2.0])
        simulation = Simulation(model, Δt=0.1, stop_time=10.0)
        simulation.output_writers[:test] = JLD2Writer(model, model.velocities,
                                                       filename="test_invalid_vector_runtime.jld2",
                                                       schedule=schedule,
                                                       overwrite_existing=true)

        # Should throw ArgumentError at runtime during initialize!
        @test_throws ArgumentError run!(simulation)

        rm("test_invalid_vector_runtime.jld2", force=true)
    end

    # Test 4: Valid case with non-zero start time
    @testset "Valid window with non-zero start time" begin
        grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid)

        # Start simulation at t=5.0
        model.clock.time = 5.0

        # First specified time at 10.0, window is 3.0 (goes back to 7.0)
        # This is after start time (5.0), so should be valid
        schedule = AveragedSpecifiedTimes([10.0, 15.0], window=3.0)
        simulation = Simulation(model, Δt=0.1, stop_time=15.0)
        simulation.output_writers[:test] = JLD2Writer(model, model.velocities,
                                                       filename="test_valid_nonzero_runtime.jld2",
                                                       schedule=schedule,
                                                       overwrite_existing=true)

        # Should not throw an error
        @test_nowarn run!(simulation)

        rm("test_valid_nonzero_runtime.jld2", force=true)
    end

    # Test 5: Invalid case with non-zero start time
    @testset "Invalid window with non-zero start time" begin
        grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid)

        # Start simulation at t=5.0
        model.clock.time = 5.0

        # First specified time at 7.0, window is 5.0 (goes back to 2.0)
        # This is before start time (5.0), so should fail
        schedule = AveragedSpecifiedTimes([7.0, 15.0], window=5.0)
        simulation = Simulation(model, Δt=0.1, stop_time=15.0)
        simulation.output_writers[:test] = JLD2Writer(model, model.velocities,
                                                       filename="test_invalid_nonzero_runtime.jld2",
                                                       schedule=schedule,
                                                       overwrite_existing=true)

        # Should throw ArgumentError
        @test_throws ArgumentError run!(simulation)

        rm("test_invalid_nonzero_runtime.jld2", force=true)
    end

    # Test 6: DateTime - Invalid case (runtime validation with non-zero start)
    @testset "Invalid DateTime window (runtime validation)" begin
        grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))

        # Start simulation at Jan 10, 2000
        start_time = DateTime(2000, 1, 10)
        clock = Clock(time=start_time)
        model = HydrostaticFreeSurfaceModel(; grid, clock)

        # First specified time at Jan 15, window is 10 days (goes back to Jan 5)
        # Simulation starts at Jan 10, 2000, so window extends before start
        times = [DateTime(2000, 1, 15), DateTime(2000, 1, 25)]
        schedule = AveragedSpecifiedTimes(times, window=Day(10))
        simulation = Simulation(model, Δt=Day(1), stop_time=DateTime(2000, 1, 25))
        simulation.output_writers[:test] = JLD2Writer(model, model.velocities,
                                                       filename="test_invalid_datetime_runtime.jld2",
                                                       schedule=schedule,
                                                       overwrite_existing=true)

        # Should throw ArgumentError at runtime during initialize!
        @test_throws ArgumentError run!(simulation)

        rm("test_invalid_datetime_runtime.jld2", force=true)
    end

    # Test 7: DateTime - Valid case
    @testset "Valid DateTime window" begin
        grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))

        start_time = DateTime(2000, 1, 1)
        clock = Clock(time=start_time)
        model = HydrostaticFreeSurfaceModel(; grid, clock)

        # First specified time at Jan 10, window is 5 days (goes back to Jan 5)
        # Simulation starts at Jan 1, 2000, so this should be valid
        times = [DateTime(2000, 1, 10), DateTime(2000, 1, 20)]
        schedule = AveragedSpecifiedTimes(times, window=Day(5))
        simulation = Simulation(model, Δt=Day(1), stop_time=DateTime(2000, 1, 20))
        simulation.output_writers[:test] = JLD2Writer(model, model.velocities,
                                                       filename="test_valid_datetime_runtime.jld2",
                                                       schedule=schedule,
                                                       overwrite_existing=true)

        # Should not throw an error
        @test_nowarn run!(simulation)

        rm("test_valid_datetime_runtime.jld2", force=true)
    end

    return nothing
end

#####
##### Run AveragedSpecifiedTimes tests
#####

@testset "AveragedSpecifiedTimes" begin
    @info "Testing AveragedSpecifiedTimes..."

    # Error handling tests
    @testset "AveragedSpecifiedTimes error handling" begin
        test_averaged_specified_times_overlapping_windows()
    end

    # Runtime validation tests
    @testset "Runtime validation of averaging windows" begin
        test_averaged_specified_times_runtime_validation()
    end

    topo = (Periodic, Periodic, Bounded)

    for arch in archs
        grid = RectilinearGrid(arch, topology=topo, size=(4, 4, 4), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))

        @testset "Time averaging simulation with AveragedSpecifiedTimes [$(typeof(arch))]" begin
            test_averaged_specified_times_simulation(model)
        end

        @testset "Averaging validation tests [$(typeof(arch))]" begin
            test_averaging_scalar_window(model)
            test_averaging_varying_windows(model)
            test_averaging_datetime_windows(model)
            test_averaging_datetime_varying_period_windows(model)
        end
    end
end

