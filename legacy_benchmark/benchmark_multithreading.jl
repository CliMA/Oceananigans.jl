push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using BSON
using Benchmarks

# Benchmark parameters

N = 512
n_threads = min.(2 .^ (0:10), Sys.CPU_THREADS) |> unique

# Run and collect benchmarks

print_system_info()

for t in n_threads
    @info "Benchmarking multithreading (N=$N, threads=$t)..."
    julia = Base.julia_cmd()
    run(`$julia -t $t --project benchmark_multithreading_single.jl $N`)
end

suite = BenchmarkGroup(["size", "threads"])
for t in n_threads
    suite[(N, t)] = BSON.load("multithreading_benchmark_$t.bson")[:trial]
end

# Summarize benchmarks

df = benchmarks_dataframe(suite)
sort!(df, :threads)
benchmarks_pretty_table(df, title="Multithreading benchmarks")

suite_Δ = speedups_suite(suite, base_case=(N, 1))
df_Δ = speedups_dataframe(suite_Δ)
sort!(df_Δ, :threads)
benchmarks_pretty_table(df_Δ, title="Multithreading speedup")
