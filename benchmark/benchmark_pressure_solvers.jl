using Printf
using TimerOutputs
using Oceananigans
using Oceananigans.Solvers

include("benchmark_utils.jl")

#####
##### Benchmark setup and parameters
#####

const timer = TimerOutput()

Ni = 2   # Number of iterations before benchmarking starts.
Nt = 10  # Number of iterations to use for benchmarking time stepping.

# Model resolutions to benchmarks. Focusing on 3D models for GPU benchmarking.
            Ns = [(256, 256, 256)]
   float_types = [Float32, Float64]  # Float types to benchmark.
         archs = [CPU()]             # Architectures to benchmark on.
@hascuda archs = [CPU(), GPU()]      # Benchmark GPU on systems with CUDA-enabled GPUs.

#####
##### Run benchmarks
#####

for arch in archs, float_type in float_types, N in Ns

    #####
    ##### Horizontally periodic pressure solve on CPU
    #####

    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), extent=(1.0, 2.5, 3.6))
    solver = PoissonSolver(arch, PPN(), grid)
    fbcs = HorizontallyPeriodicBCs()

    RHS = CellField(FT, arch, grid)
    interior(RHS) .= rand(Nx, Ny, Nz)
    interior(RHS) .= interior(RHS) .- mean(interior(RHS))

    RHS_orig = deepcopy(RHS)
    solver.storage .= interior(RHS)
    solve_poisson_3d!(solver, grid)

    Nx, Ny, Nz = N
    Lx, Ly, Lz = 1, 1, 1

    model = Model(architecture=arch, float_type=float_type, grid=RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)))
    time_step!(model, Ni, 1)

    bname =  benchmark_name(N, "", arch, float_type)
    @printf("Running static ocean benchmark: %s...\n", bname)
    for i in 1:Nt
        @timeit timer bname time_step!(model, 1, 1)
    end
end
