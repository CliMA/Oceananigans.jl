using TimerOutputs, Printf

using Oceananigans
using Oceananigans.TurbulenceClosures

include("benchmark_utils.jl")

const timer = TimerOutput()

Ni = 2   # Number of iterations before benchmarking starts.
Nt = 10  # Number of iterations to use for benchmarking time stepping.

# Run benchmark across these parameters.
            Ns = [(32, 32, 32), (64, 64, 64)]
      closures = [ConstantIsotropicDiffusivity, ConstantAnisotropicDiffusivity, ConstantSmagorinsky]
   float_types = [Float32, Float64]     # Float types to benchmark.
         archs = [CPU()]                # Architectures to benchmark on.
@hascuda archs = [CPU(), GPU()]         # Benchmark GPU on systems with CUDA-enabled GPUs.


for arch in archs, float_type in float_types, N in Ns, Closure in closures
    Nx, Ny, Nz = N
    Lx, Ly, Lz = 1, 1, 1

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=float_type, closure=Closure(float_type))
    time_step!(model, Ni, 1)

    bn =  benchmark_name(N, string(Closure), arch, float_type; npad=2)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(model, 1, 1)
    end
end

print_timer(timer, title="Oceananigans.jl turbulence closure benchmarks")
println("")

