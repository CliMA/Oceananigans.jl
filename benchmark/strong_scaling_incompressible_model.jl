using JLD2
using BenchmarkTools
using Benchmarks

# Benchmark parameters

Nx = 256
Ny = 256
Nz = Nx

ranks = (1, 2, 4, 8, 16)

# Run and collect benchmarks

print_system_info()

for r in ranks
    @info "Benchmarking distributed incompressible model strong scaling [N=($Nx, $Ny, $Nz), ranks=$r]..."
    julia = Base.julia_cmd()
    run(`mpiexec -np $r $julia --project strong_scaling_incompressible_model_single.jl $Nx $Ny $Nz`)
end

suite = BenchmarkGroup(["size", "ranks"])
for r in ranks
    jldopen("strong_scaling_incompressible_model_$r.jld2", "r") do file
        suite[((Nx, Ny, Nz), r)] = file["trial"]
    end
end

# Summarize benchmarks

df = benchmarks_dataframe(suite)
sort!(df, :ranks)
benchmarks_pretty_table(df, title="Incompressible model strong scaling benchmark")

suite_Δ = speedups_suite(suite, base_case=((Nx, Ny, Nz), 1))
df_Δ = speedups_dataframe(suite_Δ)
sort!(df_Δ, :ranks)
benchmarks_pretty_table(df_Δ, title="Incompressible model strong scaling speedup")
