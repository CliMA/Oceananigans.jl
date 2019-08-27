using TimerOutputs, Printf

using Oceananigans

const timer = TimerOutput()

Ni = 2   # Number of iterations before benchmarking starts.
Nt = 10  # Number of iterations to use for benchmarking time stepping.

# Run benchmark across these parameters.
            Ns = [(128, 128, 128)]
   float_types = [Float64]  # Float types to benchmark.
         archs = [CPU()]    # Architectures to benchmark on.
@hascuda archs = [GPU()]    # Benchmark GPU on systems with CUDA-enabled GPUs.

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

@inline function Fu(grid, U, Φ, i, j, k)
    if k == 1
        return @inbounds -2*0.1/grid.Δz^2 * (U.u[i, j, 1] - 0)
    elseif k == grid.Nz
        return @inbounds -2*0.1/grid.Δz^2 * (U.u[i, j, grid.Nz] - 0)
    else
        return 0
    end
end

@inline FT(grid, U, Φ, i, j, k) = @inbounds ifelse(k == 1, -1e-4 * (Φ.T[i, j, 1] - 0), 0)
forcing = Forcing(Fu=Fu, FT=FT)

for arch in archs, float_type in float_types, N in Ns
    Nx, Ny, Nz = N
    Lx, Ly, Lz = 1, 1, 1
    
    forced_model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=float_type, forcing=forcing)
    time_step!(forced_model, Ni, 1)  # First 1-2 iterations usually slower.

    bn =  benchmark_name(N, "with forcing", arch, float_type)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(forced_model, 1, 1)
    end

    unforced_model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=float_type)
    time_step!(unforced_model, Ni, 1)  # First 1-2 iterations usually slower.

    bn =  benchmark_name(N, "  no forcing", arch, float_type)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(unforced_model, 1, 1)
    end
end

print_timer(timer, title="Forcing function benchmarks")
println("")

