push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using DataDeps
using Oceananigans
using Benchmarks
using Statistics
using Oceananigans.Solvers: sparse_approximate_inverse

function benchmark_spai_preconditioner(N, ε, nzrel, inverse)

    grid = RectilinearGrid(CPU(), size=(N, N, 1), extent=(1, 1, 1))

    model = HydrostaticFreeSurfaceModel(
                      grid = grid,
        momentum_advection = VectorInvariant(), 
              free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver, preconditioner_method = nothing)
    )

    # to correctly create the matrix
    time_step!(model, 1)

    matrix = model.free_surface.implicit_step_solver.matrix_iterative_solver.matrix

    if !inverse
        trial = @benchmark begin
            CUDA.@sync blocking = true sparse_approximate_inverse($matrix, ε = $ε, nzrel = $nzrel)
        end samples = 5
    else
        trial = @benchmark begin
            CUDA.@sync blocking = true inv(Array($matrix))
        end samples = 5
    end

    return trial
end

N       = [64, 128, 256]
ε       = [0.1, 0.3]
nzrel   = [0.5, 1.0, 2.0]
inverse = (false, )

# Run and summarize benchmarks
print_system_info()
suite = run_benchmarks(benchmark_spai_preconditioner; N, ε, nzrel, inverse)

df = benchmarks_dataframe(suite)
benchmarks_pretty_table(df, title="SPAI preconditioner benchmarks")

