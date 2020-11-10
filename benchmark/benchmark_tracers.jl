using Oceananigans
using Benchmarks

# Utility functions for generating tracer lists

function active_tracers(n)
    n == 0 && return []
    n == 1 && return [:b]
    n == 2 && return [:T, :S]
    throw(ArgumentError("Can't have more than 2 active tracers!"))
end

passive_tracers(n) = [Symbol("C" * string(m)) for m in 1:n]

tracer_list(n_active, n_passive) =
    Tuple(vcat(active_tracers(n_active), passive_tracers(n_passive)))

function buoyancy(n_active)
    n_active == 0 && return nothing
    n_active == 1 && return BuoyancyTracer()
    n_active == 2 && return SeawaterBuoyancy()
    throw(ArgumentError("Can't have more than 2 active tracers!"))
end

# Benchmark function

function benchmark_tracers(Arch, N, n_active, n_passive)
    grid = RegularCartesianGrid(topology=topo, size=(N, N, N), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=Arch(), grid=grid, buoyancy=buoyancy(n_active),
                                tracers=tracer_list(n_active, n_passive))

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10
    
    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Ns = [192]

# Each test case specifies (number of active tracers, number of passive tracers)
test_cases = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 3), (2, 5), (2, 10)]

N_active  = [test_case[1] for test_case in test_cases]
N_passive = [test_case[2] for test_case in test_cases]

# Run benchmarks

print_machine_info()
suite = run_benchmarks(benchmark_time_stepper; Architectures, Ns, TimeSteppers)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :N_active, :N_passive, :Ns], by=(string, identity, identity, identity))
benchmarks_pretty_table(df, title="Tracers benchmarks")

if GPU in Architectures
    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df_Δ, [:N_active, :N_passive, :Ns])
    benchmarks_pretty_table(df, title="Tracers CPU -> GPU speedup")
end
