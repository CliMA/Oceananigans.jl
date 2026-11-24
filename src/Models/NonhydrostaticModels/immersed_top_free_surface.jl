using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using Oceananigans.Fields: Field
using Oceananigans.ImmersedBoundaries: ImmersedTopIBG, immersed_cell
using Oceananigans.Operators: Az_∂zᶜᶜᶠ
using Oceananigans.Solvers: Solvers, V∇²ᶜᶜᶜ

#####
##### Implicit free surface beneath an immersed top
#####
##### The free-surface condition is applied at the top wet cell of each column, so the free surface
##### is the pressure at the immersed top in top-covered columns.
#####

"""
$(TYPEDEF)

`ConjugateGradientPoissonSolver` for a `NonhydrostaticModel` with an implicit free surface on a grid with an
immersed top. The top face of a column covered by the immersed top is masked, so its vertical velocity is
kept here between time steps.

$(TYPEDFIELDS)
"""
struct ImmersedTopFreeSurfacePoissonSolver{S, W}
    "conjugate-gradient Poisson solver with the free-surface condition"
    conjugate_gradient_poisson_solver :: S
    "vertical velocity at the top of each column covered by the immersed top"
    surface_vertical_velocity :: W
end

Architectures.architecture(solver::ImmersedTopFreeSurfacePoissonSolver) = architecture(solver.conjugate_gradient_poisson_solver)
Solvers.iteration(solver::ImmersedTopFreeSurfacePoissonSolver) = Solvers.iteration(solver.conjugate_gradient_poisson_solver)
Base.summary(solver::ImmersedTopFreeSurfacePoissonSolver) =
    "ImmersedTopFreeSurfacePoissonSolver with $(summary(solver.conjugate_gradient_poisson_solver))"

@inline top_covered_column(i, j, grid) = immersed_cell(i, j, grid.Nz, grid) |
                                         (Δzᶜᶜᶜ(i, j, grid.Nz, grid) < Δzᶜᶜᶜ(i, j, grid.Nz, grid.underlying_grid))

@inline covered_wet_column(i, j, k, grid) = top_covered_column(i, j, grid) & !inactive_cell(i, j, k, grid)

@inline free_surface_index(i, j, grid::ImmersedTopIBG) = max(topmost_active_index(i, j, grid), 1)

@inline free_surface_spacing(i, j, k, grid::ImmersedTopIBG) =
    ifelse(top_covered_column(i, j, grid), Δzᶜᶜᶜ(i, j, k, grid), Δzᵃᵃᶠ(i, j, grid.Nz+1, grid))

@inline free_surface_inverse_spacing(i, j, k, grid::ImmersedTopIBG) =
    ifelse(top_covered_column(i, j, grid), 1 / Δzᶜᶜᶜ(i, j, k, grid), Δz⁻¹ᵃᵃᶠ(i, j, grid.Nz+1, grid))

# The top flux of the top wet cell is replaced by the homogeneous Robin flux
@inline function Solvers.symmetric_laplacian(i, j, k, grid::ImmersedTopIBG, ϕ, gΔt²::Number)
    top = !inactive_cell(i, j, k, grid) & ((k == grid.Nz) | inactive_cell(i, j, k+1, grid))
    Δzᶠ = free_surface_spacing(i, j, k, grid)
    Az = Azᶜᶜᶠ(i, j, k+1, grid)
    robin_flux = @inbounds - Az * ϕ[i, j, k] / (gΔt² + Δzᶠ / 2)
    top_flux = Az_∂zᶜᶜᶠ(i, j, k+1, grid, ϕ)
    return V∇²ᶜᶜᶜ(i, j, k, grid, ϕ) + ifelse(top, robin_flux - top_flux, zero(grid))
end

@kernel function _set_covered_surface_vertical_velocity!(w, grid, W)
    i, j = @index(Global, NTuple)
    k = free_surface_index(i, j, grid)
    @inbounds w[i, j, k+1] = ifelse(covered_wet_column(i, j, k, grid), W[i, j, 1], w[i, j, k+1])
end

function solve_for_pressure!(pressure, solver::ImmersedTopFreeSurfacePoissonSolver, free_surface, Ũ, Δt)
    grid = Ũ.w.grid
    launch!(architecture(grid), grid, :xy, _set_covered_surface_vertical_velocity!, Ũ.w, grid, solver.surface_vertical_velocity)
    return solve_for_pressure!(pressure, solver.conjugate_gradient_poisson_solver, free_surface, Ũ, Δt)
end

# The free-surface pressure g Δt η sits half a cell above the center of the top wet cell
@kernel function _compute_covered_surface_vertical_velocity!(W, grid, w, pNHSΔt, η, g, Δt)
    i, j = @index(Global, NTuple)
    k = free_surface_index(i, j, grid)
    h = free_surface_spacing(i, j, k, grid) / 2
    @inbounds wᵗ = w[i, j, k+1] - (g * Δt * η[i, j, grid.Nz+1] - pNHSΔt[i, j, k]) / h
    @inbounds W[i, j, 1] = ifelse(covered_wet_column(i, j, k, grid), wᵗ, zero(grid))
end

function correct_surface_vertical_velocity!(solver::ImmersedTopFreeSurfacePoissonSolver, free_surface, w, pNHSΔt, Δt)
    grid = w.grid
    arch = architecture(grid)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)
    η = free_surface.displacement
    W = solver.surface_vertical_velocity
    launch!(arch, grid, :xy, _compute_covered_surface_vertical_velocity!, W, grid, w, pNHSΔt, η, g, Δt)
    launch!(arch, grid, :xy, _compute_surface_vertical_velocity!, w, grid, pNHSΔt)
    return nothing
end

#####
##### Solver selection on grids with an immersed top
#####

nonhydrostatic_pressure_solver(grid::ImmersedTopIBG, free_surface) = ConjugateGradientPoissonSolver(grid)

immersed_top_free_surface_solver(solver, grid) = solver

immersed_top_free_surface_solver(solver::ConjugateGradientPoissonSolver, grid::ImmersedTopIBG) =
    ImmersedTopFreeSurfacePoissonSolver(solver, Field{Center, Center, Nothing}(grid))

immersed_top_free_surface_solver(solver, ::ImmersedTopIBG) =
    throw(ArgumentError("A free surface beneath an immersed top needs a ConjugateGradientPoissonSolver pressure_solver, got $(summary(solver))."))
