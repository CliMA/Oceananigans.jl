using TimerOutputs, Printf
using Oceananigans

include("benchmark_utils.jl")

const timer = TimerOutput()

Ni = 2   # Number of iterations before benchmarking starts.
Nt = 10  # Number of iterations to use for benchmarking time stepping.

# Model resolutions to benchmarks. Focusing on 3D models for GPU.
Ns = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]

float_types = [Float32, Float64]  # Float types to benchmark.
archs = [CPU()]  # Architectures to benchmark on.

# Benchmark GPU on systems with CUDA-enabled GPUs.
@hascuda archs = [CPU(), GPU()]

for arch in archs, float_type in float_types, N in Ns
    Nx, Ny, Nz = N
    Lx, Ly, Lz = 100, 100, 100

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=1e-4, κ=1e-4,
                  arch=arch, float_type=float_type)

    # Free convection model setup.
    cₚ = 4181.3  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]
    dTdz = 0.01  # [K/m]
    top_flux = -25 / (model.eos.ρ₀ * cₚ)  # 25 W/m² cooling flux.

    model.boundary_conditions.T.z.left = BoundaryCondition(Flux, top_flux)
    model.boundary_conditions.T.z.right = BoundaryCondition(Gradient, dTdz)

    # Initial temperature profile.
    T_prof = 20 .+ dTdz .* reshape(model.grid.zC, 1, 1, Nz)
    data(model.tracers.T) .= T_prof

    time_step!(model, Ni, 1)  # First 1~2 iterations are usually slower.

    bname =  benchmark_name(N, arch, float_type)
    @printf("Running free convection benchmark: %s...\n", bname)
    for i in 1:Nt
        @timeit timer bname time_step!(model, 1, 1)
    end
end

print_timer(timer, title="Oceananigans.jl free convection benchmarks")

println("\n\nCPU Float64 -> Float32 speedup:")
for N in Ns
    bn32 = benchmark_name(N, CPU(), Float32)
    bn64 = benchmark_name(N, CPU(), Float64)
    t32  = TimerOutputs.time(timer[bn32])
    t64  = TimerOutputs.time(timer[bn64])
    @printf("%s: %.3f\n", benchmark_name(N), t64/t32)
end

@hascuda begin
    println("\nGPU Float64 -> Float32 speedup:")
    for N in Ns
        bn32 = benchmark_name(N, GPU(), Float32)
        bn64 = benchmark_name(N, GPU(), Float64)
        t32  = TimerOutputs.time(timer[bn32])
        t64  = TimerOutputs.time(timer[bn64])
        @printf("%s: %.3f\n", benchmark_name(N), t64/t32)
    end

    println("\nCPU -> GPU speedup:")
    for N in Ns, ft in float_types
        bn_cpu = benchmark_name(N, CPU(), ft)
        bn_gpu = benchmark_name(N, GPU(), ft)
        t_cpu  = TimerOutputs.time(timer[bn_cpu])
        t_gpu  = TimerOutputs.time(timer[bn_gpu])
        @printf("%s: %.3f\n", benchmark_name(N, ft), t_cpu/t_gpu)
    end
end
