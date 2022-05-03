push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

# Benchmark function

function benchmark_topology(Arch, size, topology, extent=(1, 1, 1))
    grid = RectilinearGrid(Arch(); topology, size, extent)
    closure = ScalarDiffusivity(ν=1, κ=1)
    advection = WENO5()
    model = NonhydrostaticModel(; advection, grid, closure, tracers=:c)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
#Ns = [(128, 128, 128)]
Ns = [(256, 256, 1)]
PB = (Periodic, Bounded)
Topologies = collect(Iterators.product(PB, PB, PB))[:]

# Run and summarize benchmarks

suite = run_benchmarks(benchmark_topology; Architectures, Ns, Topologies)

df = benchmarks_dataframe(suite)
benchmarks_pretty_table(df, title="Topologies benchmarks")

if GPU in Architectures
    df = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df, [:Topologies, :Ns], by=(string, identity))
    benchmarks_pretty_table(df, title="Topologies CPU to GPU speedup")
end

for Arch in Architectures
    suite_arch = speedups_suite(suite[@tagged Arch], base_case=(Arch, Ns[1], (Periodic, Periodic, Periodic)))
    df_arch = speedups_dataframe(suite_arch, slowdown=true)
    sort!(df_arch, [:Topologies, :Ns], by=string)
    benchmarks_pretty_table(df_arch, title="Topologies relative performance ($Arch)")
end
