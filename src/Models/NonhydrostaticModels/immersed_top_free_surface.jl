using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field
using Oceananigans.ImmersedBoundaries: ImmersedTopIBG, immersed_cell
using Oceananigans.Operators: Azᶜᶜᶠ, Δzᶜᶜᶜ, Vᶜᶜᶜ, divᶜᶜᶜ
using Oceananigans.Solvers: ConjugateGradientSolver, DiagonallyDominantPreconditioner, VolumeInverseNorm,
                            V∇²ᶜᶜᶜ, ZFormulation

import Oceananigans.Solvers: iteration

#####
##### Implicit free surface beneath an immersed top
#####
##### The Robin condition on the nonhydrostatic pressure is applied at the top wet cell `kᵗ` of
##### each column, so the free surface is the pressure at the immersed top in top-covered columns.
#####

"""
$(TYPEDEF)

Conjugate-gradient solver for the nonhydrostatic pressure of a `NonhydrostaticModel` with an implicit
free surface on a grid with an immersed top. The free-surface condition is applied at the top
wet cell of each column.

$(TYPEDFIELDS)
"""
struct ImmersedTopFreeSurfacePoissonSolver{G, R, S, K, W}
    "the grid with an immersed top"
    grid :: G
    "right-hand side of the pressure equation"
    right_hand_side :: R
    "conjugate-gradient solver"
    conjugate_gradient_solver :: S
    "index of the top wet cell of each column, zero in dry columns"
    top_index :: K
    "vertical velocity at the top of each water column"
    surface_vertical_velocity :: W
end

architecture(solver::ImmersedTopFreeSurfacePoissonSolver) = architecture(solver.grid)
iteration(solver::ImmersedTopFreeSurfacePoissonSolver) = iteration(solver.conjugate_gradient_solver)

Base.summary(solver::ImmersedTopFreeSurfacePoissonSolver) =
    "ImmersedTopFreeSurfacePoissonSolver with $(summary(solver.conjugate_gradient_solver.preconditioner)) on $(summary(solver.grid))"

@kernel function _compute_top_index!(kᵗ, grid)
    i, j = @index(Global, NTuple)
    @inbounds kᵗ[i, j] = topmost_active_index(i, j, grid)
end

@inline top_covered_column(i, j, grid) = immersed_cell(i, j, grid.Nz, grid) |
                                         (Δzᶜᶜᶜ(i, j, grid.Nz, grid) < Δzᶜᶜᶜ(i, j, grid.Nz, grid.underlying_grid))

# The top face of a top-covered column is masked, so its vertical velocity is kept in `W`
@inline surface_predictor_velocity(i, j, grid, w, W) =
    @inbounds ifelse(top_covered_column(i, j, grid), W[i, j, 1], w[i, j, grid.Nz+1])

@inline robin_denominator(i, j, k, grid, Δt, g) = g * Δt^2 + Δzᶜᶜᶜ(i, j, k, grid) / 2

"""
$(TYPEDSIGNATURES)

Return an `ImmersedTopFreeSurfacePoissonSolver` that uses the tolerances, iteration limit and preconditioner
of `solver`. An FFT-based preconditioner is replaced by one that includes the free-surface condition at
the top of the grid.
"""
function ImmersedTopFreeSurfacePoissonSolver(solver::ConjugateGradientPoissonSolver)
    grid = solver.grid
    arch = architecture(grid)
    cg = solver.conjugate_gradient_solver
    rhs = CenterField(grid)

    Nx, Ny, _ = size(grid)
    top_index = on_architecture(arch, zeros(Int, Nx, Ny))
    launch!(arch, grid, :xy, _compute_top_index!, top_index, grid)

    conjugate_gradient_solver = ConjugateGradientSolver(compute_immersed_top_free_surface_laplacian!;
                                                        template_field = rhs,
                                                        maxiter = cg.maxiter,
                                                        reltol = cg.reltol,
                                                        abstol = cg.abstol,
                                                        preconditioner = free_surface_preconditioner(cg.preconditioner),
                                                        enforce_gauge_condition! = mask_inactive_cells!,
                                                        residual_norm = VolumeInverseNorm(grid))

    surface_vertical_velocity = Field{Center, Center, Nothing}(grid)

    return ImmersedTopFreeSurfacePoissonSolver(grid, rhs, conjugate_gradient_solver, top_index, surface_vertical_velocity)
end

free_surface_preconditioner(preconditioner) = preconditioner
free_surface_preconditioner(preconditioner::FFTBasedPoissonSolver) = inhomogeneous_z_solver(preconditioner.grid)
free_surface_preconditioner(preconditioner::FourierTridiagonalPoissonSolver) =
    free_surface_preconditioner(preconditioner, preconditioner.tridiagonal_formulation)

free_surface_preconditioner(preconditioner, ::ZFormulation) = inhomogeneous_z_solver(preconditioner.grid)
free_surface_preconditioner(preconditioner, formulation) = DiagonallyDominantPreconditioner()

inhomogeneous_z_solver(grid) = FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation=InhomogeneousFormulation(ZDirection()))

update_free_surface_preconditioner!(preconditioner, free_surface, Ũ, Δt) = nothing
update_free_surface_preconditioner!(preconditioner::FourierTridiagonalPoissonSolver, free_surface, Ũ, Δt) =
    update_fourier_tridiagonal_solver!(preconditioner, free_surface, Ũ, Δt)

# The operator is not singular, so the gauge condition only masks inactive cells
function mask_inactive_cells!(x, r)
    grid = r.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _mask_inactive_cells!, x, r, grid)
    return nothing
end

@kernel function _mask_inactive_cells!(x, r, grid)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    @inbounds x[i, j, k] *= active
    @inbounds r[i, j, k] *= active
end

@kernel function _immersed_top_free_surface_laplacian!(∇²φ, grid, φ, kᵗ, Δt, g)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    top = @inbounds k == kᵗ[i, j]
    robin = @inbounds Azᶜᶜᶠ(i, j, k+1, grid) * φ[i, j, k] / robin_denominator(i, j, k, grid, Δt, g)
    @inbounds ∇²φ[i, j, k] = active * (V∇²ᶜᶜᶜ(i, j, k, grid, φ) - ifelse(top, robin, zero(grid)))
end

function compute_immersed_top_free_surface_laplacian!(∇²φ, φ, kᵗ, Δt, g)
    grid = φ.grid
    arch = architecture(grid)
    fill_halo_regions!(φ)
    launch!(arch, grid, :xyz, _immersed_top_free_surface_laplacian!, ∇²φ, grid, φ, kᵗ, Δt, g)
    return nothing
end

@kernel function _immersed_top_free_surface_source_term!(rhs, grid, Ũ, W, η, kᵗ, Δt, g)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    δ = divᶜᶜᶜ(i, j, k, grid, Ũ.u, Ũ.v, Ũ.w)

    w̃ = surface_predictor_velocity(i, j, grid, Ũ.w, W)
    η★ = @inbounds η[i, j, grid.Nz+1] + Δt * w̃
    Az = Azᶜᶜᶠ(i, j, k+1, grid)
    top = @inbounds k == kᵗ[i, j]
    surface = @inbounds Az * (w̃ - Ũ.w[i, j, k+1] - g * Δt * η★ / robin_denominator(i, j, k, grid, Δt, g))

    @inbounds rhs[i, j, k] = active * (δ * Vᶜᶜᶜ(i, j, k, grid) + ifelse(top, surface, zero(grid)))
end

function solve_for_pressure!(pressure, solver::ImmersedTopFreeSurfacePoissonSolver, free_surface, Ũ, Δt)
    ϵ = eps(eltype(pressure))
    Δt⁺ = max(ϵ, Δt)
    Δt★ = Δt⁺ * isfinite(Δt)
    pressure .*= Δt★

    grid = solver.grid
    arch = architecture(grid)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)
    η = free_surface.displacement
    rhs = solver.right_hand_side
    kᵗ = solver.top_index
    W = solver.surface_vertical_velocity

    launch!(arch, grid, :xyz, _immersed_top_free_surface_source_term!, rhs, grid, Ũ, W, η, kᵗ, Δt, g)

    cg = solver.conjugate_gradient_solver
    update_free_surface_preconditioner!(cg.preconditioner, free_surface, Ũ, Δt)

    return solve!(pressure, cg, rhs, kᵗ, Δt, g)
end

@kernel function _update_immersed_top_free_surface!(η, W, grid, w, φ, kᵗ, Δt, g)
    i, j = @index(Global, NTuple)
    k = @inbounds kᵗ[i, j]
    wet = k > 0
    k = max(k, 1)

    w̃ = surface_predictor_velocity(i, j, grid, w, W)
    h = Δzᶜᶜᶜ(i, j, k, grid) / 2
    ηⁿ = @inbounds η[i, j, grid.Nz+1]
    wᵗ = @inbounds (h * w̃ - g * Δt * ηⁿ + φ[i, j, k]) / robin_denominator(i, j, k, grid, Δt, g)
    wᵗ = ifelse(wet, wᵗ, zero(grid))

    @inbounds W[i, j, 1] = wᵗ
    @inbounds η[i, j, grid.Nz+1] = ηⁿ + Δt * wᵗ
end

function set_top_pressure_boundary_condition!(φ, solver::ImmersedTopFreeSurfacePoissonSolver, free_surface, w̃, Δt)
    grid = solver.grid
    arch = architecture(grid)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)
    η = free_surface.displacement
    launch!(arch, grid, :xy, _update_immersed_top_free_surface!, η, solver.surface_vertical_velocity,
            grid, w̃, φ, solver.top_index, Δt, g)
    return nothing
end

@kernel function _set_immersed_top_surface_vertical_velocity!(w, W, kᵗ)
    i, j = @index(Global, NTuple)
    k = @inbounds kᵗ[i, j]
    @inbounds w[i, j, k+1] = ifelse(k > 0, W[i, j, 1], w[i, j, k+1])
end

function correct_surface_vertical_velocity!(solver::ImmersedTopFreeSurfacePoissonSolver, w, pNHSΔt)
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xy, _set_immersed_top_surface_vertical_velocity!, w, solver.surface_vertical_velocity, solver.top_index)
    return nothing
end

#####
##### Solver and pressure-field selection on grids with an immersed top
#####

nonhydrostatic_pressure_solver(grid::ImmersedTopIBG, free_surface) = ConjugateGradientPoissonSolver(grid)

immersed_top_free_surface_solver(solver, grid, free_surface) = solver
immersed_top_free_surface_solver(solver, ::ImmersedTopIBG, ::Nothing) = solver
immersed_top_free_surface_solver(solver::ConjugateGradientPoissonSolver, ::ImmersedTopIBG, ::Nothing) = solver
immersed_top_free_surface_solver(solver::ConjugateGradientPoissonSolver, ::ImmersedTopIBG, free_surface) = ImmersedTopFreeSurfacePoissonSolver(solver)
immersed_top_free_surface_solver(solver::ImmersedTopFreeSurfacePoissonSolver, ::ImmersedTopIBG, free_surface) = solver

immersed_top_free_surface_solver(::ImmersedTopFreeSurfacePoissonSolver, ::ImmersedTopIBG, ::Nothing) =
    throw(ArgumentError("An ImmersedTopFreeSurfacePoissonSolver needs a free surface."))

immersed_top_free_surface_solver(solver, ::ImmersedTopIBG, free_surface) =
    throw(ArgumentError("A free surface beneath an immersed top needs a ConjugateGradientPoissonSolver pressure_solver, got $(summary(solver))."))

# The free-surface condition enters the pressure equation, not the pressure boundary condition
free_surface_pressure_field(grid::ImmersedTopIBG) = CenterField(grid)
