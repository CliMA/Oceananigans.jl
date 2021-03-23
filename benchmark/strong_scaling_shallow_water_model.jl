using JLD2
using BenchmarkTools
using Benchmarks

# Benchmark parameters

Nx = 4096
Ny = 4096

ranks = (1, 2, 4, 8, 16)

# Run benchmarks

print_system_info()

for r in ranks
    @info "Benchmarking distributed shallow water model strong scaling [N=($Nx, $Ny), ranks=$r]..."
    julia = Base.julia_cmd()
    run(`mpiexec -np $r $julia --project strong_scaling_shallow_water_model_single.jl $Nx $Ny`)
end

# Collect benchmarks

suite = BenchmarkGroup(["size", "ranks"])
for R in ranks
    for local_rank in 0:R-1
        filename = string("strong_scaling_shallow_water_model_$(R)_$local_rank.jld2")
        jldopen(filename, "r") do file
            if local_rank == 0
                suite[((Nx, Ny), R)] = file["trial"]
            else
                merged_trial = suite[((Nx, Ny), R)]
                local_trial = file["trial"]
                append!(merged_trial.times, local_trial.times)
                append!(merged_trial.gctimes, local_trial.gctimes)
            end
        end
    end
end

# Summarize benchmarks

df = benchmarks_dataframe(suite)
sort!(df, :ranks)
benchmarks_pretty_table(df, title="Shallow water model strong scaling benchmark")

base_case = ((Nx, Ny), 1)
suite_Δ = speedups_suite(suite, base_case=base_case)
df_Δ = speedups_dataframe(suite_Δ, efficiency=:strong, base_case=base_case, key2rank=k->k[2])
sort!(df_Δ, :ranks)
benchmarks_pretty_table(df_Δ, title="Shallow water model strong scaling speedup")
