using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

# Benchmark function

function benchmark_time_stepper(Arch, N, TimeStepper)
    grid = RegularRectilinearGrid(size=(N, N, N), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=Arch(), grid=grid, timestepper=TimeStepper)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Ns = [192]
TimeSteppers = [:QuasiAdamsBashforth2, :RungeKutta3]

# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_time_stepper; Architectures, Ns, TimeSteppers)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :TimeSteppers, :Ns], by=(string, string, identity))
benchmarks_pretty_table(df, title="Time stepping benchmarks")

if GPU in Architectures
    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df_Δ, [:TimeSteppers, :Ns], by=(string, identity))
    benchmarks_pretty_table(df_Δ, title="Time stepping CPU to GPU speedup")
end

for Arch in Architectures
    suite_arch = speedups_suite(suite[@tagged Arch], base_case=(Arch, Ns[1], :QuasiAdamsBashforth2))
    df_arch = speedups_dataframe(suite_arch, slowdown=true)
    sort!(df_arch, :TimeSteppers, by=string)
    benchmarks_pretty_table(df_arch, title="Time stepping relative performance ($Arch)")
end
