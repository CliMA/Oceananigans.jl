using JLD2
using BenchmarkTools
using Benchmarks

# Benchmark parameters

decomposition = Slab()

ranks = (1, 2, 4, 8, 16)

grid_size(R, decomposition) = (256, 256, 256)

rank_size(R, ::Slab) = (1, R, 1)
rank_size(R, ::Pencil) = Int.((1, √R, √R))

# Run benchmarks

print_system_info()

# for R in ranks
#     Nx, Ny, Nz = grid_size(R, decomposition)
#     Rx, Ry, Rz = rank_size(R, decomposition)
#     @info "Benchmarking distributed incompressible model strong scaling with $(typeof(decomposition)) decomposition [N=($Nx, $Ny, $Nz), ranks=($Rx, $Ry, $Rz)]..."
#     julia = Base.julia_cmd()
#     run(`mpiexec -np $R $julia --project strong_scaling_incompressible_model_single.jl $(typeof(decomposition)) $Nx $Ny $Nz $Rx $Ry $Rz`)
# end

# Collect and merge benchmarks from all ranks

suite = BenchmarkGroup(["size", "ranks"])

for R in ranks
    Nx, Ny, Nz = grid_size(R, decomposition)
    Rx, Ry, Rz = rank_size(R, decomposition)
    case = ((Nx, Ny, Nz), (Rx, Ry, Rz))

    for local_rank in 0:R-1
        filename = string("strong_scaling_incompressible_model_$(R)ranks_$(typeof(decomposition))_$local_rank.jld2")
        jldopen(filename, "r") do file
            if local_rank == 0
                suite[case] = file["trial"]
            else
                merged_trial = suite[case]
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
benchmarks_pretty_table(df, title="Incompressible model strong scaling benchmark")

base_case = (grid_size(1, decomposition), rank_size(1, decomposition))
suite_Δ = speedups_suite(suite, base_case=base_case)
df_Δ = speedups_dataframe(suite_Δ, efficiency=:strong, base_case=base_case, key2rank=k->prod(k[2]))
sort!(df_Δ, :ranks)
benchmarks_pretty_table(df_Δ, title="Incompressible model strong scaling speedup")
