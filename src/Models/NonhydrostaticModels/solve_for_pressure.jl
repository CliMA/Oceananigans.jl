using Oceananigans.Operators
using Oceananigans.DistributedComputations: DistributedFFTBasedPoissonSolver
using Oceananigans.Grids: XDirection, YDirection, ZDirection, inactive_cell
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Solvers: solve!

#####
##### Calculate the right-hand-side of the non-hydrostatic pressure Poisson equation.
#####

@kernel function _compute_source_term!(rhs, grid, Δt, Ũ, η)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, Ũ.u, Ũ.v, Ũ.w)
    @inbounds rhs[i, j, k] = active * δ / Δt
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::XDirection, grid, Δt, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, Ũ.u, Ũ.v, Ũ.w)
    @inbounds rhs[i, j, k] = active * Δxᶜᶜᶜ(i, j, k, grid) * δ / Δt
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::YDirection, grid, Δt, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, Ũ.u, Ũ.v, Ũ.w)
    @inbounds rhs[i, j, k] = active * Δyᶜᶜᶜ(i, j, k, grid) * δ / Δt
end

@kernel function _fourier_tridiagonal_source_term!(rhs, ::ZDirection, grid, Δt, Ũ, η)
    g = 10
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, Ũ.u, Ũ.v, Ũ.w)

    source_term = active * Δzᶜᶜᶜ(i, j, k, grid) * δ / Δt

    # modifies rhs of pressure solve surface boundary condition
    if k == grid.Nz && active
        source_term -= ((η[i,j] + Δt * Ũ.w[i, j, k+1])/(Δt^2 + Δzᶜᶜᶜ(i, j, k, grid) / (2*g)))
    end
    
    @inbounds rhs[i, j, k] = source_term
end

function compute_source_term!(pressure, solver::DistributedFFTBasedPoissonSolver, Δt, Ũ, η)
    rhs  = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Δt, Ũ, η)
    return nothing        
end

function compute_source_term!(pressure, solver::DistributedFourierTridiagonalPoissonSolver, Δt, Ũ, η)
    rhs = solver.storage.zfield
    arch = architecture(solver)
    grid = solver.local_grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, Δt, Ũ, η)
    return nothing
end

# now passing \eta into function
function compute_source_term!(pressure, solver::FourierTridiagonalPoissonSolver, Δt, Ũ, η)
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, Δt, Ũ, η)
    return nothing
end

function compute_source_term!(pressure, solver::FFTBasedPoissonSolver, Δt, Ũ, η)
    rhs = solver.storage
    arch = architecture(solver)
    grid = solver.grid
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Δt, Ũ, η)
    return nothing
end

#####
##### Solve for pressure
#####

function solve_for_pressure!(pressure, solver, Δt, Ũ, η)
    compute_source_term!(pressure, solver, Δt, Ũ, η)
    solve!(pressure, solver)
    return pressure
end

function solve_for_pressure!(pressure, solver::ConjugateGradientPoissonSolver, Δt, Ũ, η)
    rhs = solver.right_hand_side
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Δt, Ũ, η)
    return solve!(pressure, solver.conjugate_gradient_solver, rhs)
end

