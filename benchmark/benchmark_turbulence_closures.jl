using TimerOutputs, Printf

using Oceananigans
using Oceananigans.TurbulenceClosures

include("benchmark_utils.jl")

#####
##### Benchmark setup and parameters
#####

const timer = TimerOutput()

Ni = 2   # Number of iterations before benchmarking starts.
Nt = 10  # Number of iterations to use for benchmarking time stepping.

# Run benchmark across these parameters.
            Ns = [(32, 32, 32), (64, 64, 64)]
      closures = [ConstantIsotropicDiffusivity, ConstantAnisotropicDiffusivity, ConstantSmagorinsky]
   float_types = [Float32, Float64]     # Float types to benchmark.
         archs = [CPU()]                # Architectures to benchmark on.
@hascuda archs = [CPU(), GPU()]         # Benchmark GPU on systems with CUDA-enabled GPUs.

#####
##### Run benchmarks
#####

for arch in archs, FT in float_types, N in Ns, Closure in closures
    Nx, Ny, Nz = N
    Lx, Ly, Lz = 1, 1, 1

    model = Model(architecture = arch, float_type = FT,
                  grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz)),
		  closure = Closure(FT))
    
    time_step!(model, Ni, 1)

    bn =  benchmark_name(N, string(Closure), arch, FT; npad=2)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(model, 1, 1)
    end
end

#####
##### Print benchmark results
#####

print_benchmark_info()
print_timer(timer, title="Turbulence closure benchmarks")
println()
