include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Units
using NCDatasets
using Zarr

# A slow signal carrying a semidiurnal and a diurnal tide.
slow_signal(t) = 2 + cos(2π * t / 10days)
tidal_signal(t) = slow_signal(t) + cos(2π * t / 12.4206hours) + cos(2π * t / 25.8193hours)
@inline tidal_signal(i, j, k, grid, clock) = tidal_signal(clock.time)
@inline linear_signal(i, j, k, grid, clock) = 1 + clock.time / 1day

function filtered_tidal_signal(arch, Writer, directory; stop_time, pickup = false)
    grid = RectilinearGrid(arch; size = (1, 1, 1), extent = (1, 1, 1))
    model = NonhydrostaticModel(grid)
    c = Field(KernelFunctionOperation{Center, Center, Center}(tidal_signal, grid, model.clock))

    simulation = Simulation(model; Δt = 10minutes, stop_time)
    simulation.output_writers[:daily] = Writer(model, (; c); dir = directory, filename = "daily",
                                                   schedule = FilteredTimeInterval(LanczosKernel(5days; cutoff = 40hours); interval = 1days), overwrite_files = pickup === false)
    simulation.output_writers[:weekly] = Writer(model, (; c); dir = directory, filename = "weekly",
                                                    schedule = FilteredTimeInterval(LanczosKernel(5days; cutoff = 40hours); interval = 7days), overwrite_files = pickup === false)
    simulation.output_writers[:checkpointer] = Checkpointer(model; dir = directory, prefix = "checkpoint",
                                                            schedule = TimeInterval(5days))
    run!(simulation; pickup)

    return simulation.output_writers[:daily].filepath, simulation.output_writers[:weekly].filepath
end

read_filtered(filepath) = FieldTimeSeries(filepath, "c")

frames(series) = Array(interior(series))[1, 1, 1, :]

for arch in archs
    A = typeof(arch)

    @testset "FilteredTimeInterval kernels [$A]" begin
        @test LanczosKernel(5days; cutoff = 2days)(0) == 1
        @test BoxcarKernel(5days)(1day) == 1
        @test BoxcarKernel(5days)(3days) == 0

        @test HanningKernel(4days)(0) == 1
        @test HanningKernel(4days)(2days) ≈ 0 atol = 1e-15
        @test HanningKernel(4days)(3days) == 0

        filter = FilteredTimeInterval(LanczosKernel(6days; cutoff = 2days); interval = 1day)
        @test filter.kernel isa LanczosKernel
        @test filter.kernel.cutoff == 2days
        @test FilteredTimeInterval(BoxcarKernel(6days); interval = 1day).kernel isa BoxcarKernel
    end

    for Writer in (JLD2Writer, NetCDFWriter, ZarrWriter)
        @testset "FilteredTimeInterval removes the tides [$A, $Writer]" begin
            daily, weekly = read_filtered.(filtered_tidal_signal(arch, Writer, mktempdir(); stop_time = 20days))

            @test daily.times ≈ (3:17) .* days
            @test weekly.times ≈ [7days, 14days]
            @test maximum(abs, frames(daily) .- slow_signal.(daily.times)) < 0.02
            @test maximum(abs, frames(weekly) .- slow_signal.(weekly.times)) < 0.02
        end

        @testset "FilteredTimeInterval continues across a checkpoint [$A, $Writer]" begin
            continuous = read_filtered(first(filtered_tidal_signal(arch, Writer, mktempdir(); stop_time = 20days)))

            directory = mktempdir()
            filtered_tidal_signal(arch, Writer, directory; stop_time = 10days)
            restarted = read_filtered(first(filtered_tidal_signal(arch, Writer, directory; stop_time = 20days,
                                                                  pickup = Int(5days / 10minutes))))

            @test restarted.times ≈ continuous.times
            @test frames(restarted) == frames(continuous)
        end

        # A running mean over a window centered on the output time returns a linear signal's value there.
        @testset "FilteredTimeInterval with BoxcarKernel [$A, $Writer]" begin
            grid = RectilinearGrid(arch; size = (1, 1, 1), extent = (1, 1, 1))
            model = NonhydrostaticModel(grid)
            c = Field(KernelFunctionOperation{Center, Center, Center}(linear_signal, grid, model.clock))
            simulation = Simulation(model; Δt = 10minutes, stop_time = 10days)
            simulation.output_writers[:boxcar] = Writer(model, (; c); dir = mktempdir(), filename = "boxcar",
                                                        schedule = FilteredTimeInterval(BoxcarKernel(4days); interval = 1day))
            run!(simulation)

            boxcar = FieldTimeSeries(simulation.output_writers[:boxcar].filepath, "c")
            @test boxcar.times ≈ (2:8) .* days
            # The first window starts at t = 0, where the initial sample has zero weight.
            @test frames(boxcar)[2:end] ≈ 1 .+ boxcar.times[2:end] ./ 1day
            @test frames(boxcar)[1] ≈ 1 + boxcar.times[1] / 1day atol = 0.01
        end
    end
end
