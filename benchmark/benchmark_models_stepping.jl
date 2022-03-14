push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks
using Plots
pyplot()
# Benchmark function

function benchmark_nonhydrostatic_model(Arch, FT, N)
    grid = RectilinearGrid(Arch(), FT, size=(N, N, N), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid=grid)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

function benchmark_hydrostatic_model(Arch, FT, N)
    grid = RectilinearGrid(Arch(), FT, size=(N, N, 10), extent=(1, 1, 1))
    model = HydrostaticFreeSurfaceModel(grid=grid, 
                                        tracers = (),
                                        buoyancy = nothing,
                                        free_surface=ImplicitFreeSurface())

    time_step!(model, 0.001) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 0.001)
    end samples=10

    return trial
end

function benchmark_shallowwater_model(Arch, FT, N)
    grid = RectilinearGrid(Arch(), FT, size=(N, N), extent=(1, 1), topology = (Periodic, Periodic, Flat))
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1.0)

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

# Run and summarize benchmarks

print_system_info()

for (model, name) in zip((:nonhydrostatic, :hydrostatic, :shallowwater), ("NonhydrostaticModel", "HydrostaticFreeSurfaceModel", "ShallowWaterModel"))

    benchmark_func = Symbol(:benchmark_, model, :_model)
    @eval begin
        suite = run_benchmarks($benchmark_func; Architectures, Float_types, Ns)
    end

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
            xlabel="Nx", ylabel="Times (ms)", title= name * " Benchmarks: CPU vs GPU")
    plot!(plt, Ns, CPU_Float64, lw=4, label="CPU Float64")
    plot!(plt, Ns, GPU_Float32, lw=4, label="GPU Float32")
    plot!(plt, Ns, GPU_Float64, lw=4, label="GPU Float64")
    display(plt)
    savefig(plt, string(model) * "_times.png")

    plt2 = plot(Ns, CPU_Float32./GPU_Float32, lw=4, xaxis=:log2, legend=:topleft, label="Float32",
                xlabel="Nx", ylabel="Speedup Ratio", title= name * "Model Benchmarks: CPU/GPU")
    plot!(plt2, Ns, CPU_Float64./GPU_Float64, lw=4, label="Float64")
    display(plt2)
    savefig(plt2, string(model) * "_speedup.png")

    df = benchmarks_dataframe(suite)
    benchmarks_pretty_table(df, title=name * " benchmarks")
end
