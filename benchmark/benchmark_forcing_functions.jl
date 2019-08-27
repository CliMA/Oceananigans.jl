using TimerOutputs, Printf

using Oceananigans

const timer = TimerOutput()

Ni = 2   # Number of iterations before benchmarking starts.
Nt = 10  # Number of iterations to use for benchmarking time stepping.

# Run benchmark across these parameters.
            Ns = [(64, 64, 64)]
   float_types = [Float64]  # Float types to benchmark.
         archs = [CPU()]             # Architectures to benchmark on.
@hascuda archs = [CPU(), GPU()]      # Benchmark GPU on systems with CUDA-enabled GPUs.

arch_name(::String) = ""
arch_name(::CPU) = "CPU"
arch_name(::GPU) = "GPU"

function benchmark_name(N, id, arch, FT; npad=2)
    Nx, Ny, Nz = N

    bn = ""
    bn *= lpad(Nx, npad, " ") * "×" * lpad(Ny, npad, " ") * "×" * lpad(Nz, npad, " ")
    bn *= " $id"

    arch = arch_name(arch)
    bn *= " ($arch, $FT)"

    return bn
end

benchmark_name(N, id) = benchmark_name(N, id, "", "", "")

for arch in archs, float_type in float_types, N in Ns
    Nx, Ny, Nz = N
    Lx, Ly, Lz = 1, 1, 1

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=float_type)
    time_step!(model, Ni, 1)  # First 1-2 iterations usually slower.

    bn =  benchmark_name(N, "  no forcing", arch, float_type)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(model, 1, 1)
    end

    @inline FT(grid, U, Φ, i, j, k) = ifelse(k == 1, -1e-4 * (Φ.T[i, j, 1] - 0), 0)
    forcing = Forcing(FT=FT)

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=float_type, forcing=forcing)
    time_step!(model, Ni, 1)  # First 1-2 iterations usually slower.

    bn =  benchmark_name(N, "with forcing", arch, float_type)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(model, 1, 1)
    end
end

print_timer(timer, title="Forcing function benchmarks")

println("")
