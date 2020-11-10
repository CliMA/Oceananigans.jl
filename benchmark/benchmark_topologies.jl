using Oceananigans
using Benchmarks

# Benchmark function

function benchmark_topology(Arch, N, topo)
    grid = RegularCartesianGrid(topology=topo, size=(N, N, N), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=Arch(), grid=grid)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10
    
    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Ns = [192]
Topologies = [(Periodic, Periodic, Periodic),
              (Periodic, Periodic,  Bounded),
              (Periodic, Bounded,   Bounded)]

# Run and summarize benchmarks

suite = run_benchmarks(benchmark_topology; Architectures, Ns, Topologies)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Topologies, :Ns], by=(string, string, identity))
benchmarks_pretty_table(df, title="Topologies benchmarks")

if GPU in Architectures
    df = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df, [:Topologies, :Ns], by=(string, identity))
    benchmarks_pretty_table(df, title="Topologies CPU -> GPU speedup")
end
