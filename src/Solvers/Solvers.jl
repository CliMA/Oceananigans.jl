module Solvers

export PoissonSolver, PoissonBCs, solve_poisson_3d!

using Oceananigans.Grids

abstract type PoissonBCs end

include("solver_utils.jl")
include("poisson_solver_cpu.jl")
include("poisson_solver_gpu.jl")

end
