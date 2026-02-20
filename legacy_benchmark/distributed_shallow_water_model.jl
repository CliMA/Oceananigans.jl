push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using JLD2
using BenchmarkTools
using Benchmarks
using Plots
pyplot()
# Benchmark parameters

print_system_info()

#set to true to use strong scaling, false to use weak scaling
strong = true
#set to true to use threads, false to use mpi
threaded = false

decomposition = Slab()
# decomposition = Pencil()

#automatically generates thread tuple
#Slab() => min.(2 .^ (0:10), Sys.CPU_THREADS) |> unique
ranks = Dict(
    Slab() => (1, 2, 4, 8, 16, 32, 64, 128),
    Pencil() => (1, 4, 16)
)[decomposition]

if strong
    grid_size(R, decomposition) = (4096, 4096)
    title = "strong"
else
    grid_size(R, ::Slab) = (4096, 256R)
    grid_size(R, ::Pencil) = (1024 * Int(√R), 1024 * Int(√R))
    title = "weak"
end

if threaded
    command(julia, R, Nx, Ny, Rx, Ry) = `$julia --threads $R --project distributed_shallow_water_model_threaded.jl $Nx $Ny`
    label = "threads"
    dis_type = "threaded"
    keyop = v -> v[2]
else
    command(julia, R, Nx, Ny, Rx, Ry) = `mpiexec -np $R $julia --project distributed_shallow_water_model_mpi.jl $Nx $Ny $Rx $Ry`
    label = "ranks"
    dis_type = "mpi"
    keyop = v -> v[2][2]
end

rank_size(R, ::Slab) = (1, R)
rank_size(R, ::Pencil) = Int.((√R, √R))

# Run benchmarks
for R in ranks
    Nx, Ny = grid_size(R, decomposition)
    Rx, Ry = rank_size(R, decomposition)
    @info string("Benchmarking ", title, " scaling shallow water model with $(typeof(decomposition)) decomposition [N=($Nx, $Ny), ", label, "=($Rx, $Ry)]...")
    julia = Base.julia_cmd()
    run(command(julia, R, Nx, Ny, Rx, Ry))
end

# Collect and merge benchmarks from all ranks
suite = BenchmarkGroup(["size", label])

for R in ranks
    Nx, Ny = grid_size(R, decomposition)
    Rx, Ry = rank_size(R, decomposition)

    if threaded
        case = ((Nx, Ny), R)
        filename = string("distributed_shallow_water_model_threads$(R).jld2")
        file = jldopen(filename, "r")
        suite[case] = file["trial"]
    else
        case = ((Nx, Ny), (Rx, Ry))
        for local_rank in 0:R-1
            filename = string("distributed_shallow_water_model_$(R)ranks_$local_rank.jld2")
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
end

plot_keys = collect(keys(suite))
sort!(plot_keys, by = keyop)
plot_num = length(plot_keys)
rank_num = zeros(Int64, plot_num)
run_times = zeros(Float64, plot_num)
eff_ratio = zeros(Float64, plot_num)
for i in 1:plot_num
    run_times[i] = mean(suite[plot_keys[i]].times) / 1.0e6
    threaded ? rank_num[i] = plot_keys[i][2] : rank_num[i] = plot_keys[i][2][2]
    eff_ratio[i] = median(suite[plot_keys[1]].times) / ((rank_num[i]^strong) * median(suite[plot_keys[i]].times))
end

plt = plot(rank_num, run_times, lw=4, xaxis=:log2, legend=:none,
          xlabel=label, ylabel="Times (ms)", title=string(dis_type, " ", title, " scaling shallow water times"))
display(plt)
savefig(plt, string(dis_type , "_", title, "_shallow_water_times.png"))


plt2 = plot(rank_num, eff_ratio, lw=4, xaxis=:log2, legend=:none, ylims=(0,1.1),
            xlabel=label, ylabel="Efficiency", title=string(dis_type, " ", title, " scaling shallow water efficiency"))
display(plt2)
savefig(plt2, string(dis_type, "_", title, "_shallow_water_efficiency.png"))

# Summarize benchmarks

df = benchmarks_dataframe(suite)
sort!(df, Symbol(label))
benchmarks_pretty_table(df, title=string(dis_type, " ", title," scaling shallow water times"))

base_case = (grid_size(ranks[1], decomposition), threaded ? ranks[1] : rank_size(ranks[1], decomposition))
suite_Δ = speedups_suite(suite, base_case=base_case)
df_Δ = speedups_dataframe(suite_Δ, slowdown=true, efficiency=Symbol(title), base_case=base_case, key2rank=k->prod(k[2]))
sort!(df_Δ, Symbol(label))
benchmarks_pretty_table(df_Δ, title=string(dis_type, " ", title," scaling shallow water efficiency"))
