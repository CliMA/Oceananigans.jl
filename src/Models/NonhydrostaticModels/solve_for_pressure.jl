using Oceananigans.Operators
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.Grids: XDirection, YDirection, ZDirection

#####
##### Calculate the right-hand-side of the non-hydrostatic pressure Poisson equation.
#####

#####
##### Solve for pressure
#####

function solve_for_pressure!(pressure, solver::FFTBasedPoissonSolver, Δt, U★)

    # Calculate right hand side:
    rhs = solver.storage
    arch = architecture(solver)
    grid = solver.grid

    launch!(arch, grid, :xyz, calculate_pressure_source_term_fft_based_solver!,
            rhs, grid, Δt, U★)

    # Solve pressure Poisson given for pressure, given rhs
    solve!(pressure, solver, rhs)

    return nothing
end

function solve_for_pressure!(pressure, solver::FourierTridiagonalPoissonSolver, Δt, U★)

    # Calculate right hand side:
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid

    launch!(arch, grid, :xyz, calculate_pressure_source_term_fourier_tridiagonal_solver!,
            rhs, grid, Δt, U★, solver.batched_tridiagonal_solver.tridiagonal_direction)

    # Pressure Poisson rhs, scaled by the spacing in the stretched direction at ᶜᶜᶜ, is stored in solver.source_term:
    solve!(pressure, solver)

    return nothing
end
