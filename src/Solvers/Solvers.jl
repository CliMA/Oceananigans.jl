module Solvers

export
    BatchedTridiagonalSolver, solve_batched_tridiagonal_system!,
    PressureSolver, solve_for_pressure!


using Oceananigans.Grids

using Oceananigans: @hascuda
@hascuda using CUDAnative, CuArrays

abstract type AbstractPressureSolver{A} end

include("solver_utils.jl")
include("batched_tridiagonal_solver.jl")

include("discrete_eigenvalues.jl")
include("plan_transforms.jl")
include("horizontally_periodic_pressure_solver.jl")
include("channel_pressure_solver.jl")
include("index_permutations.jl")
include("solve_for_pressure.jl")

end
