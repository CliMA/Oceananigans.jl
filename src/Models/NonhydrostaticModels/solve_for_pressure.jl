using Oceananigans.Operators
using Oceananigans.DistributedComputations: DistributedFFTBasedPoissonSolver
using Oceananigans.Grids: XDirection, YDirection, ZDirection, inactive_cell
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Solvers: solve!
using Oceananigans.Solvers: multiply_with_sqrt_spacing_operator, subtract_and_mask!
using Statistics

#####
##### Calculate the right-hand-side of the non-hydrostatic pressure Poisson equation.
#####

@kernel function _compute_source_term!(rhs, grid, Δt, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, Ũ.u, Ũ.v, Ũ.w)
    @inbounds rhs[i, j, k] = active * δ
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::XDirection, grid, Δt, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, Ũ.u, Ũ.v, Ũ.w)
    @inbounds rhs[i, j, k] = active * Δxᶜᶜᶜ(i, j, k, grid) * δ
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::YDirection, grid, Δt, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, Ũ.u, Ũ.v, Ũ.w)
    @inbounds rhs[i, j, k] = active * Δyᶜᶜᶜ(i, j, k, grid) * δ
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::ZDirection, grid, Δt, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, Ũ.u, Ũ.v, Ũ.w)
    @inbounds rhs[i, j, k] = active * Δzᶜᶜᶜ(i, j, k, grid) * δ
end

function compute_source_term!(pressure, solver::DistributedFFTBasedPoissonSolver, Δt, Ũ)
    rhs  = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Δt, Ũ)
    return nothing
end

function compute_source_term!(pressure, solver::DistributedFourierTridiagonalPoissonSolver, Δt, Ũ)
    rhs = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, Δt, Ũ)
    return nothing
end

function compute_source_term!(pressure, solver::FourierTridiagonalPoissonSolver, Δt, Ũ)
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, Δt, Ũ)
    return nothing
end

function compute_source_term!(pressure, solver::FFTBasedPoissonSolver, Δt, Ũ)
    rhs = solver.storage
    arch = architecture(solver)
    grid = solver.grid
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Δt, Ũ)
    return nothing
end

#####
##### Solve for pressure
#####

function solve_for_pressure!(pressure, solver, Δt, Ũ)
    ϵ = eps(eltype(pressure))
    Δt⁺ = max(ϵ, Δt)
    Δt★ = Δt⁺ * isfinite(Δt)
    pressure .*= Δt★

    compute_source_term!(pressure, solver, Δt, Ũ)
    solve!(pressure, solver)
    return pressure
end

function solve_for_pressure!(pressure, solver::ConjugateGradientPoissonSolver, Δt, Ũ)
    ϵ = eps(eltype(pressure))
    Δt⁺ = max(ϵ, Δt)
    Δt★ = Δt⁺ * isfinite(Δt)
    pressure .*= Δt★

    rhs = solver.right_hand_side
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Δt, Ũ)

    if solver.symmetrized
        launch!(arch, grid, :xyz, multiply_with_sqrt_spacing_operator, rhs, grid, Vᶜᶜᶜ)
    end

    solve!(pressure, solver.conjugate_gradient_solver, rhs)

    if solver.symmetrized
        launch!(arch, grid, :xyz, multiply_with_sqrt_spacing_operator, pressure, grid, V⁻¹ᶜᶜᶜ)
    end

    mean_p = mean(pressure)
    mean_rhs = mean(rhs)

    launch!(arch, grid, :xyz, subtract_and_mask!, pressure, grid, mean_p)
    launch!(arch, grid, :xyz, subtract_and_mask!, rhs, grid, mean_rhs)

    return pressure
end

