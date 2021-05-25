using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

# Benchmark function

function benchmark_incompressible_model(Arch, FT, N)
    grid = RegularRectilinearGrid(FT, size=(N, N, N), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=Arch(), float_type=FT, grid=grid)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Float_types = [Float32, Float64]
Ns = [32, 64, 128, 256]

# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_incompressible_model; Architectures, Float_types, Ns)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Float_types, :Ns], by=(string, string, identity))
benchmarks_pretty_table(df, title="Incompressible model benchmarks")

if GPU in Architectures
    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df_Δ, [:Float_types, :Ns], by=(string, identity))
    benchmarks_pretty_table(df_Δ, title="Incompressible model CPU to GPU speedup")
end
