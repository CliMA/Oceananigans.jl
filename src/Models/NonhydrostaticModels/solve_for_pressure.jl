using Oceananigans.Operators
using Oceananigans.DistributedComputations: DistributedFFTBasedPoissonSolver
using Oceananigans.Grids: XDirection, YDirection, ZDirection, inactive_cell
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Solvers: solve!

#####
##### Calculate the right-hand-side of the non-hydrostatic pressure Poisson equation.
#####

@kernel function _compute_source_term!(rhs, grid, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    u, v, w = Ũ
    δ = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    @inbounds rhs[i, j, k] = active * δ
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::XDirection, grid, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    u, v, w = Ũ
    δ = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    @inbounds rhs[i, j, k] = active * Δxᶜᶜᶜ(i, j, k, grid) * δ
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::YDirection, grid, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    u, v, w = Ũ
    δ = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    @inbounds rhs[i, j, k] = active * Δyᶜᶜᶜ(i, j, k, grid) * δ
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::ZDirection, grid, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    u, v, w = Ũ
    δ = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    @inbounds rhs[i, j, k] = active * Δzᶜᶜᶜ(i, j, k, grid) * δ
end

function compute_source_term!(solver::DistributedFFTBasedPoissonSolver, Ũ)
    rhs  = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Ũ)
    return nothing
end

function compute_source_term!(solver::DistributedFourierTridiagonalPoissonSolver, Ũ)
    rhs = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, Ũ)
    return nothing
end

function compute_source_term!(solver::FourierTridiagonalPoissonSolver, Ũ)
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid
    @show tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, Ũ)
    return nothing
end

function compute_source_term!(solver::FFTBasedPoissonSolver, Ũ)
    rhs = solver.storage
    arch = architecture(solver)
    grid = solver.grid
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Ũ)
    return nothing
end

#####
##### Solve for pressure
#####

# Note that Δt is unused here.
function solve_for_pressure!(pressure, solver, Δt, args...)
    compute_source_term!(solver, args...)
    solve!(pressure, solver)
    return pressure
end

function solve_for_pressure!(pressure, solver::ConjugateGradientPoissonSolver, Δt, args...)
    ϵ = eps(eltype(pressure))
    Δt⁺ = max(ϵ, Δt)
    Δt★ = Δt⁺ * isfinite(Δt)
    pressure .*= Δt★

    rhs = solver.right_hand_side
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, args...)
    return solve!(pressure, solver.conjugate_gradient_solver, rhs)
end

