module Solvers

export
    PoissonSolver, PoissonBCs, solve_poisson_3d!,
    BatchedTridiagonalSolver, solve_batched_tridiagonal_system!,
    solve_for_pressure!, calculate_poisson_right_hand_side!, idct_permute!


using Oceananigans.Grids

using Oceananigans: @hascuda
@hascuda using CUDAnative, CuArrays

abstract type PoissonBCs end

include("solver_utils.jl")
include("poisson_solver_cpu.jl")
include("poisson_solver_gpu.jl")
include("batched_tridiagonal_solver.jl")

abstract type AbstractPressureSolver{A} end

include("discrete_eigenvalues.jl")
include("plan_transforms.jl")
include("horizontally_periodic_pressure_solver.jl")
include("index_permutations.jl")
include("calculate_right_hand_side.jl")
include("solve_for_pressure.jl")

end
