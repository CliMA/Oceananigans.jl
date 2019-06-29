using TimerOutputs, Printf
using Oceananigans

const timer = TimerOutput()

Ni = 2   # Number of iterations before benchmarking starts.
Nt = 10  # Number of iterations to use for benchmarking time stepping.

# Model resolutions to benchmarks. Focusing on 3D models for GPU.
Ns = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]

float_types = [Float32, Float64]  # Float types to benchmark.
archs = [CPU()]  # Architectures to benchmark on.

# Benchmark GPU on systems with CUDA-enabled GPUs.
@hascuda archs = [CPU(), GPU()]

arch_name(::CPU) = "CPU"
arch_name(::GPU) = "GPU"

benchmark_name(N)               = benchmark_name(N, nothing, nothing)
benchmark_name(N, arch::Symbol) = benchmark_name(N, arch, nothing)
benchmark_name(N, ft::DataType) = benchmark_name(N, nothing, ft)

function benchmark_name(N, arch, ft; npad=3)
    Nx, Ny, Nz = N
    print_arch = typeof(arch) <: Architecture ? true : false
    print_ft   = typeof(ft) == DataType && ft <: AbstractFloat ? true : false

    bn = ""
    bn *= lpad(Nx, npad, " ") * "×" * lpad(Ny, npad, " ") * "×" * lpad(Nz, npad, " ")

    if print_arch && print_ft
        arch = arch_name(arch)
        bn *= " ($arch, $ft)"
    elseif print_arch && !print_ft
        arch = arch_name(arch)
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
    time_step!(model, Ni, 1)  # First 1~2 iterations are usually slower.

    bname =  benchmark_name(N, arch, float_type)
    @printf("Running benchmark: %s...\n", bname)
    for i in 1:Nt
        @timeit timer bname time_step!(model, 1, 1)
    end
end

print_timer(timer, title="Oceananigans.jl static ocean benchmarks")

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
