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

add_inhomogeneous_boundary_terms!(rhs, ::Nothing, grid, Ũ, Δt) = nothing

@kernel function _add_inhomogeneous_boundary_terms!(rhs, grid, w̃, Δt, g, η)
    i, j = @index(Global, NTuple)
    k = grid.Nz

    Δzᶜ = Δzᵃᵃᶜ(i, j, k, grid)
    Δzᶠ = Δzᵃᵃᶠ(i, j, k, grid)

    @inbounds begin
        num = η[i, j, k+1] + Δt * w̃[i, j, k]
        den = Δzᶜ * Δt^2 + Δzᶜ * Δzᶠ / 2g
        rhs[i, j, k] -= num / den
    end
end

# function add_inhomogeneous_boundary_terms!(rhs, free_surface::ImplicitFreeSurface, grid, Ũ, Δt)
function add_inhomogeneous_boundary_terms!(rhs, free_surface, grid, Ũ, Δt)
    g = free_surface.gravitational_acceleration
    η = free_surface.η
    arch = grid.architecture
    launch!(arch, grid, :xy, _add_inhomogeneous_boundary_terms!, rhs, grid, Ũ.w, Δt, g, η)
    return nothing
end

function compute_source_term!(solver::FourierTridiagonalPoissonSolver, free_surface, Ũ, Δt)
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, _fourier_tridiagonal_source_term!, rhs, tdir, grid, Ũ)

    # Add the inhomgeneous terms on the top boundary associated with an implicit
    # free surface formulation represneting a Robin boundary condition on pressure.
    add_inhomogeneous_boundary_terms!(rhs, free_surface, grid, Ũ, Δt)

    return nothing
end

function compute_source_term!(solver::FFTBasedPoissonSolver, ::Nothing, Ũ, Δt)
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
function solve_for_pressure!(pressure, solver, free_surface, Ũ, Δt)
    compute_source_term!(solver, free_surface, Ũ, Δt)
    update_fourier_tridiagonal_solver!(solver, free_surface, Ũ, Δt)
    solve!(pressure, solver)
    return pressure
end

update_fourier_tridiagonal_solver!(solver, ::Nothing, Ũ, Δt) = nothing

function update_fourier_tridiagonal_solver!(solver, free_surface, Ũ, Δt)
    g = free_surface.gravitational_acceleration
    η = free_surface.η
    λx, λy = solver.poisson_eigenvalues
    grid = solver.grid
    arch = grid.architecture
    diagonal = solver.batched_tridiagonal_solver.b
    launch!(arch, grid, :xy, _update_fourier_tridiagonal_solver!, diagonal, grid, Ũ, Δt, g, η, λx, λy)
end

@kernel function _update_fourier_tridiagonal_solver!(diagonal, grid, Ũ, Δt, g, η, λx, λy)
    i, j, = @index(Global, NTuple)
    Nz = grid.Nz
    Δzᶠ = Δzᵃᵃᶠ(i, j, Nz + 1, grid)
    Δzᶜ = Δzᵃᵃᶠ(i, j, Nz, grid)
    num = 1 / Δzᶠ + 3 / (2g * Δt^2)
    den = 1 / Δzᶠ + 1 / (2g * Δt^2)
    @inbounds diagonal[i, j, Nz] = 1 / Δzᶠ * num / den - Δzᶜ * (λx[i] + λy[j])
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