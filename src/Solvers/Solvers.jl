module Solvers

export
    BatchedTridiagonalSolver, solve!,
    FFTBasedPoissonSolver,
    FourierTridiagonalPoissonSolver,
    ConjugateGradientSolver,
    HeptadiagonalIterativeSolver

using Statistics
using FFTW
using CUDA
using SparseArrays
using KernelAbstractions

using Oceananigans.Architectures: device, CPU, GPU, array_type, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.BoundaryConditions
using Oceananigans.Fields

using Oceananigans.Grids: unpack_grid, inactive_cell
using Oceananigans.Grids: XYRegularRG, XZRegularRG, YZRegularRG, XYZRegularRG

"""
    ω(M, k)

Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)

reshaped_size(N, dim) = dim == 1 ? (N, 1, 1) :
                        dim == 2 ? (1, N, 1) :
                        dim == 3 ? (1, 1, N) : nothing

include("batched_tridiagonal_solver.jl")
include("conjugate_gradient_solver.jl")
include("poisson_eigenvalues.jl")
include("index_permutations.jl")
include("discrete_transforms.jl")
include("plan_transforms.jl")
include("fft_based_poisson_solver.jl")
include("fourier_tridiagonal_poisson_solver.jl")
include("conjugate_gradient_poisson_solver.jl")
include("sparse_approximate_inverse.jl")
include("matrix_solver_utils.jl")
include("sparse_preconditioners.jl")
include("heptadiagonal_iterative_solver.jl")

const GridWithFFTSolver = Union{XYZRegularRG, XYRegularRG, XZRegularRG, YZRegularRG}
const GridWithFourierTridiagonalSolver = Union{XYRegularRG, XZRegularRG, YZRegularRG}

fft_poisson_solver(grid::XYZRegularRG) = FFTBasedPoissonSolver(grid)
fft_poisson_solver(grid::GridWithFourierTridiagonalSolver) =
    FourierTridiagonalPoissonSolver(grid.underlying_grid)

end # module
