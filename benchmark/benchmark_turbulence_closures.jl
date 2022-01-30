push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Oceananigans.TurbulenceClosures
using Benchmarks

# Benchmark function

function benchmark_closure(Arch, Closure)
    grid = RectilinearGrid(size=(128, 128, 128), extent=(1, 1, 1))
    model = NonhydrostaticModel(architecture=Arch(), grid=grid, closure=Closure())

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = CUDA.functional() ? [CPU, GPU] : [CPU]

Closures = [Nothing,
            IsotropicDiffusivity,
            AnisotropicDiffusivity,
            AnisotropicBiharmonicDiffusivity,
            TwoDimensionalLeith,
            SmagorinskyLilly,
            AnisotropicMinimumDissipation]

# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_closure; Architectures, Closures)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Closures], by=(string, string))
benchmarks_pretty_table(df, title="Turbulence closure benchmarks")

if GPU in Architectures
    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df_Δ, :Closures, by=string)
    benchmarks_pretty_table(df_Δ, title="Turbulence closure CPU to GPU speedup")
end

for Arch in Architectures
    suite_arch = speedups_suite(suite[@tagged Arch], base_case=(Arch, Nothing))
    df_arch = speedups_dataframe(suite_arch, slowdown=true)
    sort!(df_arch, :Closures, by=string)
    benchmarks_pretty_table(df_arch, title="Turbulence closures relative performance ($Arch)")
end
