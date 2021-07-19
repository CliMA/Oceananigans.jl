push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Oceananigans.BuoyancyModels
using SeawaterPolynomials
using Benchmarks

# Benchmark function

function benchmark_equation_of_state(Arch, EOS)
    grid = RegularRectilinearGrid(size=(192, 192, 192), extent=(1, 1, 1))
    buoyancy = SeawaterBuoyancy(equation_of_state=EOS())
    model = NonhydrostaticModel(architecture=Arch(), grid=grid, buoyancy=buoyancy)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
EquationsOfState = [LinearEquationOfState, SeawaterPolynomials.RoquetEquationOfState, SeawaterPolynomials.TEOS10EquationOfState]

# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_equation_of_state; Architectures, EquationsOfState)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :EquationsOfState], by=string)
benchmarks_pretty_table(df, title="Equation of state benchmarks")

if GPU in Architectures
    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df_Δ, :EquationsOfState, by=string)
    benchmarks_pretty_table(df_Δ, title="Equation of state CPU to GPU speedup")
end

for Arch in Architectures
    suite_arch = speedups_suite(suite[@tagged Arch], base_case=(Arch, LinearEquationOfState))
    df_arch = speedups_dataframe(suite_arch, slowdown=true)
    sort!(df_arch, :EquationsOfState, by=string)
    benchmarks_pretty_table(df_arch, title="Equation of state relative performance ($Arch)")
end
