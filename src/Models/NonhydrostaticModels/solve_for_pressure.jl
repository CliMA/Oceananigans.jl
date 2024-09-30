using Oceananigans.Operators
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.DistributedComputations: DistributedFFTBasedPoissonSolver
using Oceananigans.Grids: XDirection, YDirection, ZDirection

#####
##### Calculate the right-hand-side of the non-hydrostatic pressure Poisson equation.
#####

@kernel function _compute_source_term!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w)
    @inbounds rhs[i, j, k] = active * δ / Δt
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::XDirection, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w)
    @inbounds rhs[i, j, k] = active * Δxᶜᶜᶜ(i, j, k, grid) * δ / Δt
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::YDirection, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w)
    @inbounds rhs[i, j, k] = active * Δyᶜᶜᶜ(i, j, k, grid) * δ / Δt
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::ZDirection, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w)
    @inbounds rhs[i, j, k] = active * Δzᶜᶜᶜ(i, j, k, grid) * δ / Δt
end

function compute_source_term!(pressure, solver::DistributedFFTBasedPoissonSolver, Δt, U★)
    rhs  = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Δt, U★)
    return nothing        
end

function compute_source_term!(pressure, solver::DistributedFourierTridiagonalPoissonSolver, Δt, U★)
    rhs = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
    tridiagonal_dir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!,
            rhs, grid, Δt, U★, tridiagonal_dir)
    return nothing
end

function compute_source_term!(pressure, solver::FourierTridiagonalPoissonSolver, Δt, U★)
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid
    tridiagonal_dir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!,
            rhs, grid, Δt, U★, tridiagonal_dir)
    return nothing
end

function compute_source_term!(pressure, solver::FFTBasedPoissonSolver, Δt, U★)
    rhs = solver.storage
    arch = architecture(solver)
    grid = solver.grid
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Δt, U★)
    return nothing
end

#####
##### Solve for pressure
#####

function solve_for_pressure!(pressure, solver, Δt, U★)
    compute_source_term!(pressure, solver, Δt, U★)
    solve!(pressure, solver)
    return pressure
end

