push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using JLD2
using BenchmarkTools
using Benchmarks
using Plots
pyplot()
# Benchmark parameters

decomposition = Slab()

ranks = (1, 2, 4, 8, 16)

grid_size(R, decomposition) = (256, 256, 256)

rank_size(R, ::Slab) = (1, R, 1)
rank_size(R, ::Pencil) = Int.((1, √R, √R))

# Run benchmarks

print_system_info()

for R in ranks
    Nx, Ny, Nz = grid_size(R, decomposition)
    Rx, Ry, Rz = rank_size(R, decomposition)
    @info "Benchmarking distributed nonhydrostatic model strong scaling with $(typeof(decomposition)) decomposition [N=($Nx, $Ny, $Nz), ranks=($Rx, $Ry, $Rz)]..."
    julia = Base.julia_cmd()
    run(`mpiexec -np $R $julia --project strong_scaling_nonhydrostatic_model_single.jl $(typeof(decomposition)) $Nx $Ny $Nz $Rx $Ry $Rz`)
end

# Collect and merge benchmarks from all ranks

suite = BenchmarkGroup(["size", "ranks"])

for R in ranks
    Nx, Ny, Nz = grid_size(R, decomposition)
    Rx, Ry, Rz = rank_size(R, decomposition)
    case = ((Nx, Ny, Nz), (Rx, Ry, Rz))

    for local_rank in 0:R-1
        filename = string("strong_scaling_nonhydrostatic_model_$(R)ranks_$(typeof(decomposition))_$local_rank.jld2")
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

plot_keys = collect(keys(suite))
sort!(plot_keys, by = v -> v[2][2])
plot_num = length(plot_keys)
rank_num = zeros(Int64, plot_num)
run_times = zeros(Float64, plot_num)
eff_ratio = zeros(Float64, plot_num)
for i in 1:plot_num
    rank_num[i] = plot_keys[i][2][2]
    run_times[i] = mean(suite[plot_keys[i]].times) / 1.0e6
    eff_ratio[i] = median(suite[plot_keys[1]].times) /(rank_num[i] * median(suite[plot_keys[i]].times))
end

plt = plot(rank_num, run_times, lw=4, xaxis=:log2, legend=:none,
          xlabel="Cores", ylabel="Times (ms)", title="Strong Scaling Nonhydrostatic Times")
display(plt)
savefig(plt, "ss_nonhydrostatic_times.png")


plt2 = plot(rank_num, eff_ratio, lw=4, xaxis=:log2, legend=:none, ylims=(0,1.1),
            xlabel="Cores", ylabel="Efficiency", title="Strong Scaling Nonhydrostatic Efficiency")
display(plt2)
savefig(plt2, "ss_nonhydrostatic_efficiency.png")

df = benchmarks_dataframe(suite)
sort!(df, :ranks)
benchmarks_pretty_table(df, title="Nonhydrostatic model strong scaling benchmark")

base_case = (grid_size(1, decomposition), rank_size(1, decomposition))
suite_Δ = speedups_suite(suite, base_case=base_case)
df_Δ = speedups_dataframe(suite_Δ, efficiency=:strong, base_case=base_case, key2rank=k->prod(k[2]))
sort!(df_Δ, :ranks)
benchmarks_pretty_table(df_Δ, title="Nonhydrostatic model strong scaling speedup")
