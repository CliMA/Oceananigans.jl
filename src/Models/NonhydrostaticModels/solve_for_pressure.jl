using Oceananigans.Operators
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.Distributed: DistributedFFTBasedPoissonSolver
using Oceananigans.Grids: XDirection, YDirection, ZDirection

#####
##### Calculate the right-hand-side of the non-hydrostatic pressure Poisson equation.
#####

@kernel function calculate_pressure_source_term_fft_based_solver!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@kernel function calculate_pressure_source_term_fourier_tridiagonal_solver!(rhs, grid, Δt, U★, ::XDirection)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = Δxᶜᶜᶜ(i, j, k, grid) * divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@kernel function calculate_pressure_source_term_fourier_tridiagonal_solver!(rhs, grid, Δt, U★, ::YDirection)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = Δyᶜᶜᶜ(i, j, k, grid) * divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@kernel function calculate_pressure_source_term_fourier_tridiagonal_solver!(rhs, grid, Δt, U★, ::ZDirection)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = Δzᶜᶜᶜ(i, j, k, grid) * divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

#####
##### Solve for pressure
#####

function solve_for_pressure!(pressure, solver::DistributedFFTBasedPoissonSolver, Δt, U★)
    rhs = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid

    launch!(arch, grid, :xyz, calculate_pressure_source_term_fft_based_solver!,
            rhs, grid, Δt, U★)

    # Solve pressure Poisson equation for pressure, given rhs
    solve!(pressure, solver)

    return pressure
end

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
