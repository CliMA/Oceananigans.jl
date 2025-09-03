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

add_inhomogeneous_boundary_terms!(rhs, grid, Ũ, Δt, ::Nothing, ::Nothing) = nothing

@kernel function _add_inhomogeneous_boundary_terms!(rhs, grid, w̃, Δt, g, η)
    i, j = @index(Global, NTuple)
    k = grid.Nz

    @inbounds begin
        num = η[i, j, k+1] + Δt * w̃[i, j, k]
        den = Δzᶜᶜᶜ(i, j, k, grid) * Δt^2 + Δzᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶠ(i, j, k, grid) / 2g
        rhs[i, j, k] -= num / den
    end
end

function add_inhomogeneous_boundary_terms!(rhs, grid, Ũ, Δt, g, η)
    arch = grid.architecture
    launch!(arch, grid, :xy, _add_inhomogeneous_boundary_terms!, rhs, grid, Ũ.w, Δt, g, η)
    return nothing
end

function compute_source_term!(solver::FourierTridiagonalPoissonSolver, Ũ, Δt, g, η)
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, Ũ)

    # When g and η are given, we assume that we are using an implicit free surface
    # formulation, and add the associated inhomgeneous terms on the top boundary which
    # represent a Robin boundary condition on pressure.
    add_inhomogeneous_boundary_terms!(rhs, grid, Ũ, Δt, g, η)

    return nothing
end

function compute_source_term!(solver::FFTBasedPoissonSolver, Ũ, Δt, g, η)
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
function solve_for_pressure!(pressure, solver, Ũ, Δt, g, η)
    compute_source_term!(solver, Ũ, Δt, g, η)
    # update_fourier_tridiagonal_solver!(solver, Ũ, Δt, g, η)
        #=
        D[i, j, Nz] = -(-1 / Δzᵃᵃᶠ(i, j, Nz, grid) *((-3 / (2*g*Δt^2) - 1 / Δzᵃᵃᶠ(i, j, Nz, grid))/(1 / Δzᵃᵃᶠ(i, j, Nz, grid) + 1 / (2*g*Δt^2)))) - Δzᵃᵃᶜ(i, j, Nz, grid) * (λx[i] + λy[j])
        =#
    solve!(pressure, solver)
    return pressure
end

function solve_for_pressure!(pressure, solver::ConjugateGradientPoissonSolver, Ũ, Δt, g, η)
    ϵ = eps(eltype(pressure))
    Δt⁺ = max(ϵ, Δt)
    Δt★ = Δt⁺ * isfinite(Δt)
    pressure .*= Δt★

    rhs = solver.right_hand_side
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Ũ)

    return solve!(pressure, solver.conjugate_gradient_solver, rhs)
end

# TODO: write a function to add the inhomogeneous boundary contributions to `rhs`
# for a non-hydrostatic implicit free surface
#=
function add_inhomogeneous_boundary_terms!(rhs, solver, grid, Ũ, Δt, g, η)
    launch!(arch, grid, :xy, _add_implicit_free_surface_source_term!, rhs, grid, Ũ, Δt, g, η)
end

function _add_implicit_free_surface_source_term(rhs, solver, grid, Ũ, Δt, g, η)
    # modifies rhs of pressure solve surface boundary condition to allow for free surface
    if k == grid.Nz && active
        source_term -= ((η[i,j] + Δt * w[i, j, k+1])/(Δt^2 + Δzᶜᶜᶜ(i, j, k, grid) / (2*g))) * Δt
    end
=#