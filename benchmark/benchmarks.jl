using TimerOutputs, Printf
using Oceananigans

const timer = TimerOutput()

Nt = 50  # Number of time steps to use for benchmarking time stepping.

# Model resolutions to benchmarks.
Ns = [(1, 1, 512),
      (1, 128, 128), (128, 1, 128), (128, 128, 1),
      (32, 32, 32), (64, 64, 64)]

float_types = [Float32, Float64]  # Float types to benchmark.
archs = [:CPU]  # Architectures to benchmark on.

Oceananigans.@hascuda archs = [:CPU, :GPU]  # Benchmark GPU on CUDA-enabled computers.

benchmark_name(N, id)               = benchmark_name(N, id, nothing, nothing)
benchmark_name(N, id, arch::Symbol) = benchmark_name(N, id, arch, nothing)
benchmark_name(N, id, ft::DataType) = benchmark_name(N, id, nothing, ft)

function benchmark_name(N, id, arch, ft; npad=3)
    Nx, Ny, Nz = N
    print_arch = typeof(arch) == Symbol ? true : false
    print_ft   = typeof(ft) == DataType && ft <: AbstractFloat ? true : false

    bn = ""
    bn *= lpad(Nx, npad, " ") * "x" * lpad(Ny, npad, " ") * "x" * lpad(Nz, npad, " ")
    bn *= " $id"

    if print_arch && print_ft
        bn *= " ($arch, $ft)"
    elseif print_arch && !print_ft
        bn *= " ($arch)"
    elseif !print_arch && print_ft
        bn *= " ($ft)"
    end

    return bn
end

for arch in archs, float_type in float_types, N in Ns
    Nx, Ny, Nz = N
    Lx, Ly, Lz = 100, 100, 100

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=float_type)
    time_step!(model, 1, 1)  # First time step is usually slower.

    bn =  benchmark_name(N, "static ocean", arch, float_type)
    for i in 1:Nt
        @timeit timer bn time_step!(model, 1, 1)
    end
end

print_timer(timer, title="Oceananigans.jl benchmarks")

bid = "static ocean"  # Benchmark ID. We only have one right now.

println("\n\nCPU Float64 -> Float32 speedups:")
for N in Ns
    bn32 = benchmark_name(N, bid, :CPU, Float32)
    bn64 = benchmark_name(N, bid, :CPU, Float64)
    t32  = TimerOutputs.time(timer[bn32])
    t64  = TimerOutputs.time(timer[bn64])
    @printf("%s: %.3f\n", benchmark_name(N, bid), t64/t32)
end

Oceananigans.@hascuda begin
    println("\nGPU Float64 -> Float32 speedups:")
    for N in Ns
        bn32 = benchmark_name(N, bid, :GPU, Float32)
        bn64 = benchmark_name(N, bid, :GPU, Float64)
        t32  = TimerOutputs.time(timer[bn32])
        t64  = TimerOutputs.time(timer[bn64])
        @printf("%s: %.3f\n", benchmark_name(N, bid), t64/t32)
    end

    println("\nCPU -> GPU speedsup:")
    for N in Ns, ft in float_types
        bn_cpu = benchmark_name(N, bid, :CPU, ft)
        bn_gpu = benchmark_name(N, bid, :GPU, ft)
        t_cpu  = TimerOutputs.time(timer[bn_cpu])
        t_gpu  = TimerOutputs.time(timer[bn_gpu])
        @printf("%s: %.3f\n", benchmark_name(N, bid, ft), t_gpu/t_cpu)
    end
end
