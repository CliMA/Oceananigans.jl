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

Architecture = has_cuda() ? [CPU, GPU] : [CPU]
Float_type = [Float32, Float64]
N = [32, 64, 128, 256]

suite = run_benchmarks(benchmark_incompressible_model; Architecture, Float_type, N)

df = benchmarks_dataframe(suite)
sort!(df, [:Architecture, :Float_type, :N], by=(string, string, identity))
benchmarks_pretty_table(df, title="Incompressible model benchmarks")

if length(Archs) > 1
    df = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df, [:Float_type, :N], by=(string, identity))
    benchmarks_pretty_table(df, title="Incompressible model CPU -> GPU speedup")
end
