using Oceananigans.Solvers: calculate_pressure_source_term_fft_based_solver!

import Oceananigans.Solvers: solve_for_pressure!, source_term_storage, source_term_kernel, solution_storage

source_term_storage(solver::DistributedFFTBasedPoissonSolver) = first(solver.storage)

source_term_kernel(::DistributedFFTBasedPoissonSolver) = calculate_pressure_source_term_fft_based_solver!

solution_storage(solver::DistributedFFTBasedPoissonSolver) = first(solver.storage)
