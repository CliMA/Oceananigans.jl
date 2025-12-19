module Solvers

export
    BatchedTridiagonalSolver, solve!,
    FFTBasedPoissonSolver,
    FourierTridiagonalPoissonSolver,
    ConjugateGradientSolver,
    KrylovSolver

using Statistics
using FFTW
using GPUArraysCore
using SparseArrays
using KernelAbstractions

using Oceananigans.Architectures: CPU, GPU, on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.BoundaryConditions
using Oceananigans.Fields

using Oceananigans.Grids: inactive_cell
using Oceananigans.Grids: XYRegularRG, XZRegularRG, YZRegularRG, XYZRegularRG, RectilinearGrid, RegularVerticalCoordinate

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
include("krylov_solver.jl")

const GridWithFFTSolver = Union{XYZRegularRG, XYRegularRG, XZRegularRG, YZRegularRG}
const GridWithFourierTridiagonalSolver = Union{XYRegularRG, XZRegularRG, YZRegularRG}

# Type alias for non-distributed architectures to avoid ambiguity with DistributedComputations methods
const SingleArchitecture = Union{CPU, GPU}

# Constrain to non-distributed grids by requiring the architecture parameter to be CPU or GPU
fft_poisson_solver(grid::RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate, <:Number, <:Number, <:Any, <:Any, <:SingleArchitecture}) =
    FFTBasedPoissonSolver(grid)

fft_poisson_solver(grid::RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate, <:Number, <:Any, <:Any, <:Any, <:SingleArchitecture}) =
    FourierTridiagonalPoissonSolver(grid)

fft_poisson_solver(grid::RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate, <:Any, <:Number, <:Any, <:Any, <:SingleArchitecture}) =
    FourierTridiagonalPoissonSolver(grid)

fft_poisson_solver(grid::RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Number, <:Number, <:Any, <:Any, <:SingleArchitecture}) =
    FourierTridiagonalPoissonSolver(grid)

const FFTW_NUM_THREADS = Ref{Int}(1)

function __init__()

    # See: https://github.com/CliMA/Oceananigans.jl/issues/1113
    # but don't affect global FFTW configuration for other packages using FFTW
    # FFTW.set_num_threads(4threads)
    FFTW_NUM_THREADS[] = 4 * Threads.nthreads()
end

end # module
