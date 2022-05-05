push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks
using Plots
pyplot()
# Benchmark function

function benchmark_nonhydrostatic_model(Arch, FT, N)
    grid = RectilinearGrid(Arch(), FT, size=(N, N, N), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

function benchmark_hydrostatic_model(Arch, FT, N)
    grid = RectilinearGrid(Arch(), FT, size=(N, N, 10), extent=(1, 1, 1))
    model = HydrostaticFreeSurfaceModel(grid=grid, 
                                        tracers = (),
                                        buoyancy = nothing,
                                        free_surface=ImplicitFreeSurface())

    time_step!(model, 0.001) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 0.001)
    end samples=10

    return trial
end

function benchmark_shallowwater_model(Arch, FT, N)
    grid = RectilinearGrid(Arch(), FT, size=(N, N), extent=(1, 1), topology = (Periodic, Periodic, Flat))
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1.0)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Float_types = [Float64]
Ns = [32, 64, 128, 256]

# Run and summarize benchmarks

print_system_info()

for (model, name) in zip((:nonhydrostatic, :hydrostatic, :shallowwater), ("NonhydrostaticModel", "HydrostaticFreeSurfaceModel", "ShallowWaterModel"))

    benchmark_func = Symbol(:benchmark_, model, :_model)
    @eval begin
        suite = run_benchmarks($benchmark_func; Architectures, Float_types, Ns)
    end

    df = benchmarks_dataframe(suite)
    benchmarks_pretty_table(df, title=name * " benchmarks")
end
