using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

# Benchmark function

function benchmark_vertically_stretched_incompressible_model(Arch, FT, N)
    grid = VerticallyStretchedRectilinearGrid(architecture=Arch(), size=(N, N, N), x=(0, 1), y=(0, 1), z_faces=collect(0:N))
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
suite = run_benchmarks(benchmark_vertically_stretched_incompressible_model; Architectures, Float_types, Ns)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Float_types, :Ns], by=(string, string, identity))
benchmarks_pretty_table(df, title="Vertically-stretched incompressible model benchmarks")

if GPU in Architectures
    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df_Δ, [:Float_types, :Ns], by=(string, identity))
    benchmarks_pretty_table(df_Δ, title="Vertically-stretched incompressible model CPU to GPU speedup")
end
