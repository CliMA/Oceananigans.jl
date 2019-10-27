module Solvers

export
    PoissonSolver, PoissonBCs, solve_poisson_3d!

abstract type PoissonBCs end

include("utils.jl")
include("poisson_solver_cpu.jl")
include("poisson_solver_gpu.jl")

end
