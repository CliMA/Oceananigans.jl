using JLD2
using BenchmarkTools 
using Benchmarks

# Benchmark parameters

Nx = 4096
Ny = 4096

ranks = (1, 2, 4, 8, 16)

# Run and collect benchmarks

print_system_info()

for r in ranks
    @info "Benchmarking distributed shallow water model strong scaling [N=($Nx, $Ny), ranks=$r]..."
    julia = Base.julia_cmd()
    run(`mpiexec -np $r $julia --project strong_scaling_shallow_water_model_single.jl $Nx $Ny`)
end

suite = BenchmarkGroup(["size", "ranks"])
for r in ranks
    for local_rank in collect(0:(r-1))
        file_name = string("strong_scaling_shallow_water_model_",r,"_",local_rank,".jld2")
        jldopen(file_name, "r") do file suite[((Nx, Ny), r)] = file["trial"]
        end
    end
end

# Summarize benchmarks

df = benchmarks_dataframe(suite)
sort!(df, :ranks)
benchmarks_pretty_table(df, title="Shallow water model strong scaling benchmark")

suite_Δ = speedups_suite(suite, base_case=((Nx, Ny), 1))
df_Δ = speedups_dataframe(suite_Δ)
sort!(df_Δ, :ranks)
benchmarks_pretty_table(df_Δ, title="Shallow water model strong scaling speedup")

