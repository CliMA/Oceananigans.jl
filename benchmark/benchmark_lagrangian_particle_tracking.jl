push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

# Benchmark function

function benchmark_particle_tracking(Arch, N_particles)
    grid = RectilinearGrid(size=(128, 128, 128), extent=(1, 1, 1))

    if N_particles == 0
        particles = nothing
    else
        ArrayType = Arch == CPU ? Array : CuArray
        x₀ = zeros(N_particles) |> ArrayType
        y₀ = zeros(N_particles) |> ArrayType
        z₀ = zeros(N_particles) |> ArrayType
        particles = LagrangianParticles(x=x₀, y=y₀, z=z₀)
    end

    model = NonhydrostaticModel(architecture=Arch(), grid=grid, particles=particles)

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = CUDA.functional() ? [CPU, GPU] : [CPU]
N_particles = [0, 1, 10, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8]

# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_particle_tracking; Architectures, N_particles)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :N_particles], by=(string, identity))
benchmarks_pretty_table(df, title="Lagrangian particle tracking benchmarks")

if GPU in Architectures
    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df_Δ, :N_particles)
    benchmarks_pretty_table(df_Δ, title="Lagrangian particle tracking CPU to GPU speedup")
end

for Arch in Architectures
    suite_arch = speedups_suite(suite[@tagged Arch], base_case=(Arch, N_particles[1]))
    df_arch = speedups_dataframe(suite_arch, slowdown=true)
    sort!(df_arch, :N_particles)
    benchmarks_pretty_table(df_arch, title="Lagrangian particle tracking relative performance ($Arch)")
end
