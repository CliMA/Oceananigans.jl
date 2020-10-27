using BenchmarkTools
using CUDA
using Oceananigans

include("Benchmarks.jl")
using .Benchmarks

function benchmark_incompressible_model(Arch, FT, N)
    grid = RegularCartesianGrid(FT, size=(N, N, N), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=Arch(), float_type=FT, grid=grid)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10
    
    return trial
end

Archs = has_cuda() ? [CPU, GPU] : [CPU]
FT = [Float32, Float64]
Ns = [32, 64, 128, 256]

suite = run_benchmark_suite(benchmark_incompressible_model; Archs, FT, Ns)

df = benchmark_suite_to_dataframe(suite)
sort!(df, [:Archs, :FT, :Ns], by=(string, string, identity))
summarize_benchmark_suite(df, title="Incompressible model benchmarks")

if length(Archs) > 1
    df = gpu_speedup_suite(suite) |> speedup_suite_to_dataframe
    sort!(df, [:FT, :Ns], by=(string, identity))
    summarize_benchmark_suite(df, title="Incompressible model CPU -> GPU speedup")
end
