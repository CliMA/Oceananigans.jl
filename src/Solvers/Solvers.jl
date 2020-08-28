module Solvers

export
    BatchedTridiagonalSolver, solve_batched_tridiagonal_system!,
    PressureSolver, solve_for_pressure!

using CUDA
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Architectures: device, @hascuda, CPU, GPU, array_type
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Grids: unpack_grid

include("solver_utils.jl")
include("batched_tridiagonal_solver.jl")

include("discrete_eigenvalues.jl")
include("plan_transforms.jl")
include("pressure_solver.jl")
include("triply_periodic_pressure_solver.jl")
include("horizontally_periodic_pressure_solver.jl")
include("channel_pressure_solver.jl")
include("box_pressure_solver.jl")
include("index_permutations.jl")
include("solve_for_pressure.jl")

end
