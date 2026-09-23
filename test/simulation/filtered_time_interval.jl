include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Units

# A slow signal carrying a semidiurnal and a diurnal tide.
slow_signal(t) = 2 + cos(2π * t / 10days)
tidal_signal(t) = slow_signal(t) + cos(2π * t / 12.4206hours) + cos(2π * t / 25.8193hours)
@inline tidal_signal(i, j, k, grid, clock) = tidal_signal(clock.time)
@inline linear_signal(i, j, k, grid, clock) = 1 + clock.time / 1day

function filtered_tidal_signal(arch, directory; stop_time, pickup = false)
    grid = RectilinearGrid(arch; size = (1, 1, 1), extent = (1, 1, 1))
    model = NonhydrostaticModel(grid)
    c = Field(KernelFunctionOperation{Center, Center, Center}(tidal_signal, grid, model.clock))

    simulation = Simulation(model; Δt = 10minutes, stop_time)
    simulation.output_writers[:daily] = JLD2Writer(model, (; c); dir = directory, filename = "daily",
                                                   schedule = FilteredTimeInterval(Lanczos(40hours); interval = 1days, window = 5days), overwrite_files = pickup === false)
    simulation.output_writers[:weekly] = JLD2Writer(model, (; c); dir = directory, filename = "weekly",
                                                    schedule = FilteredTimeInterval(Lanczos(40hours); interval = 7days, window = 5days), overwrite_files = pickup === false)
    simulation.output_writers[:checkpointer] = Checkpointer(model; dir = directory, prefix = "checkpoint",
                                                            schedule = TimeInterval(5days))
    run!(simulation; pickup)

    daily = FieldTimeSeries(simulation.output_writers[:daily].filepath, "c")
    weekly = FieldTimeSeries(simulation.output_writers[:weekly].filepath, "c")
    return daily, weekly
end

frames(series) = Array(interior(series))[1, 1, 1, :]

for arch in archs
    A = typeof(arch)

    @testset "FilteredTimeInterval removes the tides [$A]" begin
        daily, weekly = filtered_tidal_signal(arch, mktempdir(); stop_time = 20days)

        @test daily.times ≈ (3:17) .* days
        @test weekly.times ≈ [7days, 14days]
        @test maximum(abs, frames(daily) .- slow_signal.(daily.times)) < 0.02
        @test maximum(abs, frames(weekly) .- slow_signal.(weekly.times)) < 0.02
    end

    @testset "FilteredTimeInterval continues across a checkpoint [$A]" begin
        continuous, _ = filtered_tidal_signal(arch, mktempdir(); stop_time = 20days)

        directory = mktempdir()
        filtered_tidal_signal(arch, directory; stop_time = 10days)
        restarted, _ = filtered_tidal_signal(arch, directory; stop_time = 20days,
                                             pickup = Int(5days / 10minutes))

        @test restarted.times ≈ continuous.times
        @test frames(restarted) == frames(continuous)
    end

    @testset "FilteredTimeInterval kernels [$A]" begin
        @test Lanczos(2days)(0, 5days) == 1
        @test Boxcar()(1day, 5days) == 1
        @test Boxcar()(3days, 5days) == 0

        @test Hanning()(0, 4days) == 1
        @test Hanning()(2days, 4days) ≈ 0 atol = 1e-15
        @test Hanning()(3days, 4days) == 0

        filter = FilteredTimeInterval(Lanczos(2days); interval = 1day, window = 6days)
        @test filter.kernel isa Lanczos
        @test filter.kernel.cutoff == 2days
        @test FilteredTimeInterval(Boxcar(); interval = 1day, window = 6days).kernel isa Boxcar

        # A running mean over a window centered on a frame returns a linear signal's value at its center.
        grid = RectilinearGrid(arch; size = (1, 1, 1), extent = (1, 1, 1))
        model = NonhydrostaticModel(grid)
        c = Field(KernelFunctionOperation{Center, Center, Center}(linear_signal, grid, model.clock))
        directory = mktempdir()
        simulation = Simulation(model; Δt = 10minutes, stop_time = 10days)
        simulation.output_writers[:boxcar] = JLD2Writer(model, (; c); dir = directory, filename = "boxcar",
                                                        schedule = FilteredTimeInterval(Boxcar(); interval = 1day, window = 4days))
        run!(simulation)

        boxcar = FieldTimeSeries(simulation.output_writers[:boxcar].filepath, "c")
        @test boxcar.times ≈ (2:8) .* days
        # The first frame's window starts at t = 0, where the initial sample has zero weight.
        @test frames(boxcar)[2:end] ≈ 1 .+ boxcar.times[2:end] ./ 1day
        @test frames(boxcar)[1] ≈ 1 + boxcar.times[1] / 1day atol = 0.01
    end
end
