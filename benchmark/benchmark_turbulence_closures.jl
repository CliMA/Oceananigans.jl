using Oceananigans
using Benchmarks

# Benchmark function

function benchmark_closure(Arch, Closure)
    grid = RegularCartesianGrid(size=(192, 192, 192), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=Arch(), grid=grid, advection=Closure())

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10
    
    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]

Closures = [Nothing,
            IsotropicDiffusivity,
            AnisotropicDiffusivity,
            AnisotropicBiharmonicDiffusivity,
            TwoDimensionalLeith,
            SmagorinskyLilly,
            AnisotropicMinimumDissipation]

# Run and summarize benchmarks

print_machine_info()
suite = run_benchmarks(benchmark_closure; Architectures, Closures)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Closures], by=(string, string))
benchmarks_pretty_table(df, title="Turbulence closure benchmarks")

if GPU in Architectures
    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df_Δ, :Closures, by=string)
    benchmarks_pretty_table(df, title="Turbulence closure CPU -> GPU speedup")
end
