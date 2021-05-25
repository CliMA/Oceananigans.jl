using BenchmarkTools
using CUDA
using Oceananigans
using Oceananigans.Models: ShallowWaterModel
using Benchmarks

# Benchmark function

function benchmark_shallow_water_model(Arch, FT, N)
    grid = RegularRectilinearGrid(FT, size=(N, N), extent=(1, 1), topology=(Periodic, Periodic, Flat), halo=(3, 3, 0))
    model = ShallowWaterModel(architecture=Arch(), grid=grid, gravitational_acceleration=1.0)
    set!(model, h=1)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        CUDA.@sync blocking=true time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Float_types = [Float64]
Ns = [32, 64, 128, 256, 512, 1024, 2048, 4096]

# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_shallow_water_model; Architectures, Float_types, Ns)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Float_types, :Ns], by=(string, string, identity))
benchmarks_pretty_table(df, title="Shallow water model benchmarks")

if GPU in Architectures
    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df_Δ, [:Float_types, :Ns], by=(string, identity))
    benchmarks_pretty_table(df_Δ, title="Shallow water model CPU to GPU speedup")
end
