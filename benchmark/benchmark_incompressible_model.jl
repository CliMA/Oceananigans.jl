using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks
using Plots
pyplot()
# Benchmark function

function benchmark_incompressible_model(Arch, FT, N)
    grid = RegularRectilinearGrid(FT, size=(N, N, N), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=Arch(), float_type=FT, grid=grid)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Float_types = [Float32, Float64]
Ns = [32, 64, 128, 256]
#
# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_incompressible_model; Architectures, Float_types, Ns)

plot_num = length(Ns)
CPU_Float32 = zeros(Float64, plot_num)
CPU_Float64 = zeros(Float64, plot_num)
GPU_Float32 = zeros(Float64, plot_num)
GPU_Float64 = zeros(Float64, plot_num)
plot_keys = collect(keys(suite))
sort!(plot_keys, by = v -> [Symbol(v[1]), Symbol(v[2]), v[3]])

for i in 1:plot_num
    CPU_Float32[i] = mean(suite[plot_keys[i]].times) / 1.0e6
    CPU_Float64[i] = mean(suite[plot_keys[i+plot_num]].times) / 1.0e6
    GPU_Float32[i] = mean(suite[plot_keys[i+2plot_num]].times) / 1.0e6
    GPU_Float64[i] = mean(suite[plot_keys[i+3plot_num]].times) / 1.0e6
end

plt = plot(Ns, CPU_Float32, lw=4, label="CPU Float32", xaxis=:log2, yaxis=:log, legend=:topleft,
          xlabel="Nx", ylabel="Times (ms)", title="Incompressible Model Benchmarks: CPU vs GPU")
plot!(plt, Ns, CPU_Float64, lw=4, label="CPU Float64")
plot!(plt, Ns, GPU_Float32, lw=4, label="GPU Float32")
plot!(plt, Ns, GPU_Float64, lw=4, label="GPU Float64")
display(plt)
savefig(plt, "incompressible_times.png")


plt2 = plot(Ns, CPU_Float32./GPU_Float32, lw=4, xaxis=:log2, legend=:topleft, label="Float32",
            xlabel="Nx", ylabel="Speedup Ratio", title="Incompressible Model Benchmarks: CPU/GPU")
plot!(plt2, Ns, CPU_Float64./GPU_Float64, lw=4, label="Float64")
display(plt2)
savefig(plt2, "incompressible_speedup.png")

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Float_types, :Ns], by=(string, string, identity))
benchmarks_pretty_table(df, title="Incompressible model benchmarks")

if GPU in Architectures
    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df_Δ, [:Float_types, :Ns], by=(string, identity))
    benchmarks_pretty_table(df_Δ, title="Incompressible model CPU to GPU speedup")
end
