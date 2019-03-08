using TimerOutputs, Printf
using Oceananigans

const timer = TimerOutput()

Nt = 100  # Number of time steps to use for benchmarking time stepping.

# Model resolutions to benchmarks.
Ns = [(32, 32, 32), (64, 64, 64)]

float_types = [Float32, Float64]  # Float types to benchmark.
archs = [:CPU]  # Architectures to benchmark on.

Oceananigans.@hascuda archs = [:CPU, :GPU]  # Benchmark GPU on CUDA-enabled computers.

benchmark_name(N)           = "$(N[1])x$(N[2])x$(N[3]) static ocean"
benchmark_name(N, ft)       = "$(N[1])x$(N[2])x$(N[3]) static ocean ($ft)"
benchmark_name(N, arch)     = "$(N[1])x$(N[2])x$(N[3]) static ocean ($arch)"
benchmark_name(N, arch, ft) = "$(N[1])x$(N[2])x$(N[3]) static ocean ($arch, $ft)"

for arch in archs, float_type in float_types, N in Ns
    Nx, Ny, Nz = N
    Lx, Ly, Lz = 100, 100, 100

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=float_type)
    time_step!(model, 1, 1)  # First time step is usually slower.

    for i in 1:Nt
        @timeit timer benchmark_name(N, arch, float_type) time_step!(model, 1, 1)
    end
end

print_timer(timer, title="Oceananigans.jl benchmarks")

println("\n\nCPU Float64 -> Float32 speedups:")
for N in Ns
    bn32 = benchmark_name(N, :CPU, Float32)
    bn64 = benchmark_name(N, :CPU, Float64)
    t32  = TimerOutputs.time(timer[bn32])
    t64  = TimerOutputs.time(timer[bn64])
    @printf("%s: %.3f\n", benchmark_name(N), t64/t32)
end

Oceananigans.@hascuda begin
    println("\nGPU Float64 -> Float32 speedups:")
    for N in Ns
        bn32 = benchmark_name(N, :GPU, Float32)
        bn64 = benchmark_name(N, :GPU, Float64)
        t32  = TimerOutputs.time(timer[bn32])
        t64  = TimerOutputs.time(timer[bn64])
        @printf("%s: %.3f\n", benchmark_name(N), t64/t32)
    end

    println("\nCPU -> GPU speedsup:")
    for N in Ns, ft in float_types
        bn_cpu = benchmark_name(N, :CPU, ft)
        bn_gpu = benchmark_name(N, :GPU, ft)
        t_cpu  = TimerOutputs.time(timer[bn_cpu])
        t_gpu  = TimerOutputs.time(timer[bn_gpu])
        @printf("%s: %.3f\n", benchmark_name(N, ft), t_gpu/t_cpu)
    end
end
