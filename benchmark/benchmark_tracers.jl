using Printf, TimerOutputs
using Oceananigans

include("benchmark_utils.jl")

const timer = TimerOutput()

####
#### Benchmark parameters
####

Ni = 2   # Number of iterations before benchmarking starts.
Nt = 10  # Number of iterations to use for benchmarking time stepping.

         archs = [CPU()]             # Architectures to benchmark on.
@hascuda archs = [GPU()]      # Benchmark GPU on systems with CUDA-enabled GPUs.

FT = Float64
Nxyz(::CPU) = (32, 32, 32)
Nxyz(::GPU) = (256, 256, 256)

####
#### Utility functions for generating tracer lists
####

function active_tracers(n)
    n == 0 && return []
    n == 1 && return [:b]
    n == 2 && return [:T, :S]
    throw(ArgumentError("Can't have more than 2 active tracers!"))
end

passive_tracers(n) = [Symbol("C" * string(n)) for n in 1:n]

tracer_list(na, np) = Tuple(vcat(active_tracers(na), passive_tracers(np)))

function na2buoyancy(n)
    n == 0 && return nothing
    n == 1 && return BuoyancyTracer()
    n == 2 && return SeawaterBuoyancy()
    throw(ArgumentError("Can't have more than 2 active tracers!"))
end

####
#### Run benchmarks.
####

test_cases = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 3), (2, 5), (2, 10)]

for arch in archs, test_case in test_cases
    N = Nxyz(arch)
    Nx, Ny, Nz = N
    Lx, Ly, Lz = 1, 1, 1

    na, np = test_case
    tracers = tracer_list(na, np)

    bname =  benchmark_name(N, "$na active + $(lpad(np, 2)) passive", arch, FT)
    @printf("Running benchmark: %s...\n", bname)
    
    model = Model(architecture = arch,
                    float_type = FT,
                          grid = RegularCartesianGrid(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz)),
                      buoyancy = na2buoyancy(na),
                       tracers = tracers)
    time_step!(model, Ni, 1)

    for i in 1:Nt
        @timeit timer bname time_step!(model, 1, 1)
    end
end

print_timer(timer, title="Tracer benchmarks", sortby=:name)
println("")

