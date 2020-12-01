module Solvers

export
    BatchedTridiagonalSolver, solve_batched_tridiagonal_system!,
    PressureSolver, solve_for_pressure!

using FFTW
using CUDA
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Architectures: device, @hascuda, CPU, GPU, array_type, arch_array
using Oceananigans.Utils
using Oceananigans.Grids

using Oceananigans.Grids: unpack_grid

"""
    ω(M, k)

Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)

reshaped_size(N, dim) = dim == 1 ? (N, 1, 1) :
                        dim == 2 ? (1, N, 1) :
                        dim == 3 ? (1, 1, N) : nothing

include("batched_tridiagonal_solver.jl")
include("poisson_eigenvalues.jl")
include("transforms.jl")
include("plan_transforms.jl")
include("pressure_solver.jl")
include("solve_poisson_equation.jl")
include("index_permutations.jl")
include("solve_for_pressure.jl")

end
