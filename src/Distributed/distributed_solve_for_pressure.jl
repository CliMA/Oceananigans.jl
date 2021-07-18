using Oceananigans.Solvers: calculate_pressure_source_term_fft_based_solver!

import Oceananigans.Solvers: calculate_pressure_poisson_rhs!

source_term_storage(solver::DistributedFFTBasedPoissonSolver) = first(solver.storage)

source_term_kernel(::DistributedFFTBasedPoissonSolver) = calculate_pressure_source_term_fft_based_solver!

solution_storage(solver::DistributedFFTBasedPoissonSolver) = first(solver.storage)

function calculate_pressure_poisson_rhs!(solver::DistributedFFTBasedPoissonSolver, Δt, U★)
    rhs = first(solver.storage)
    arch = solver.architecture
    grid = solver.grid

    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fft_based_solver!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    return solver.storage[2] # ??
end
