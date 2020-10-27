using BenchmarkTools
using CUDA

using Oceananigans
using Oceananigans.Advection

include("Benchmarks.jl")
using .Benchmarks

# Benchmark function

function benchmark_advection_scheme(Arch, Scheme)
    grid = RegularCartesianGrid(size=(192, 192, 192), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=Arch(), grid=grid, advection=Scheme())

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10
    
    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Schemes = (CenteredSecondOrder, CenteredFourthOrder, UpwindBiasedThirdOrder, UpwindBiasedFifthOrder, WENO5)

# Run and summarize benchmarks

suite = run_benchmarks(benchmark_advection_scheme; Architectures, Schemes)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Schemes], by=string)
benchmarks_pretty_table(df, title="Advection scheme benchmarks")

for Arch in Architectures
    suite_arch = speedups_suite(suite[@tagged Arch], base_case=(Arch, CenteredSecondOrder))
    df = speedups_dataframe(suite_arch, slowdown=true)
    sort!(df, :Schemes, by=string)
    benchmarks_pretty_table(df, title="Advection schemes relative performance ($Arch)")
end
