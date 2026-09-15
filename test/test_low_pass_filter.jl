include("dependencies_for_runtests.jl")

using Oceananigans.Units

# A slow signal carrying a semidiurnal and a diurnal tide.
slow_signal(t) = 2 + cos(2π * t / 10days)
tidal_signal(t) = slow_signal(t) + cos(2π * t / 12.4206hours) + cos(2π * t / 25.8193hours)
@inline tidal_signal(i, j, k, grid, clock) = tidal_signal(clock.time)

function filtered_tidal_signal(arch, directory; stop_time, pickup = false)
    grid = RectilinearGrid(arch; size = (1, 1, 1), extent = (1, 1, 1))
    model = NonhydrostaticModel(grid)
    c = Field(KernelFunctionOperation{Center, Center, Center}(tidal_signal, grid, model.clock))

    simulation = Simulation(model; Δt = 10minutes, stop_time)
    simulation.output_writers[:daily] = JLD2Writer(model, (; c); dir = directory, filename = "daily",
                                                   schedule = LowPassFilter(1days), overwrite_existing = pickup === false)
    simulation.output_writers[:weekly] = JLD2Writer(model, (; c); dir = directory, filename = "weekly",
                                                    schedule = LowPassFilter(7days), overwrite_existing = pickup === false)
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

    @testset "LowPassFilter removes the tides [$A]" begin
        daily, weekly = filtered_tidal_signal(arch, mktempdir(); stop_time = 20days)

        @test daily.times ≈ (3:17) .* days
        @test weekly.times ≈ [7days, 14days]
        @test maximum(abs, frames(daily) .- slow_signal.(daily.times)) < 0.02
        @test maximum(abs, frames(weekly) .- slow_signal.(weekly.times)) < 0.02
    end

    @testset "LowPassFilter continues across a checkpoint [$A]" begin
        continuous, _ = filtered_tidal_signal(arch, mktempdir(); stop_time = 20days)

        directory = mktempdir()
        filtered_tidal_signal(arch, directory; stop_time = 10days)
        restarted, _ = filtered_tidal_signal(arch, directory; stop_time = 20days,
                                             pickup = Int(5days / 10minutes))

        @test restarted.times ≈ continuous.times
        @test frames(restarted) == frames(continuous)
    end
end
