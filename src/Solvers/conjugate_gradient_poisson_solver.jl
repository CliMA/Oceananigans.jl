using Oceananigans.Operators: Vᶜᶜᶜ, V⁻¹ᶜᶜᶜ, Ax_∂xᶠᶜᶜ, Axᶠᶜᶜ, Ay_∂yᶜᶠᶜ, Ayᶜᶠᶜ, Az_∂zᶜᶜᶠ,
    Azᶜᶜᶠ, Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶜᶠ, δxᶜᶜᶜ, δyᶜᶜᶜ, δzᶜᶜᶜ, Δzᵃᵃᶠ
using Oceananigans.Grids: XYZRegularRG, XYRegularRG, XZRegularRG, YZRegularRG
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Statistics: mean

#####
##### Volume-inverse-weighted residual norm
#####
##### The V∇² system has residuals that scale with cell volume.
##### Using ||V⁻¹r|| instead of ||r|| for convergence makes the
##### criterion independent of cell volume, allowing universal tolerances.
#####

struct VolumeInverseNorm{G}
    grid :: G
end

@kernel function _scale_by_volume_inverse!(r, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds r[i, j, k] = r[i, j, k] * V⁻¹ᶜᶜᶜ(i, j, k, grid)
end

@kernel function _scale_by_volume!(r, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds r[i, j, k] = r[i, j, k] * Vᶜᶜᶜ(i, j, k, grid)
end

function (vin::VolumeInverseNorm)(r)
    grid = vin.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _scale_by_volume_inverse!, r, grid)
    n = norm(r)
    launch!(arch, grid, :xyz, _scale_by_volume!, r, grid)
    return n
end

Base.summary(::VolumeInverseNorm) = "VolumeInverseNorm"

struct ConjugateGradientPoissonSolver{G, R, S}
    grid :: G
    right_hand_side :: R
    conjugate_gradient_solver :: S
end

Architectures.architecture(solver::ConjugateGradientPoissonSolver) = architecture(solver.grid)
iteration(cgps::ConjugateGradientPoissonSolver) = iteration(cgps.conjugate_gradient_solver)

Base.summary(ips::ConjugateGradientPoissonSolver) =
    "ConjugateGradientPoissonSolver with $(summary(ips.conjugate_gradient_solver.preconditioner)) on $(summary(ips.grid))"

function Base.show(io::IO, ips::ConjugateGradientPoissonSolver)
    A = architecture(ips.grid)
    print(io, "ConjugateGradientPoissonSolver:", '\n',
              "├── grid: ", summary(ips.grid), '\n',
              "└── conjugate_gradient_solver: ", summary(ips.conjugate_gradient_solver), '\n',
              "    ├── maxiter: ", prettysummary(ips.conjugate_gradient_solver.maxiter), '\n',
              "    ├── reltol: ", prettysummary(ips.conjugate_gradient_solver.reltol), '\n',
              "    ├── abstol: ", prettysummary(ips.conjugate_gradient_solver.abstol), '\n',
              "    ├── preconditioner: ", prettysummary(ips.conjugate_gradient_solver.preconditioner), '\n',
              "    └── iteration: ", prettysummary(ips.conjugate_gradient_solver.iteration))
end

@inline function V∇²ᶜᶜᶜ(i, j, k, grid, c)
    return δxᶜᶜᶜ(i, j, k, grid, Ax_∂xᶠᶜᶜ, c) +
           δyᶜᶜᶜ(i, j, k, grid, Ay_∂yᶜᶠᶜ, c) +
           δzᶜᶜᶜ(i, j, k, grid, Az_∂zᶜᶜᶠ, c)
end

@kernel function _symmetric_laplacian_operator!(∇²ϕ, grid, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²ϕ[i, j, k] = V∇²ᶜᶜᶜ(i, j, k, grid, ϕ)
end

function compute_symmetric_laplacian!(∇²ϕ, ϕ)
    grid = ϕ.grid
    arch = architecture(grid)
    fill_halo_regions!(ϕ)
    launch!(arch, grid, :xyz, _symmetric_laplacian_operator!, ∇²ϕ, grid, ϕ)
    return nothing
end

# Linear operator for the free-surface pressure Poisson equation.
# Applies V∇² with Neumann BC at the top, then adds the Robin BC diagonal correction
# -Az(Nz+1)/den * ϕ[Nz] at k=Nz (where den = g*Δt² + Δzᶠ/2).
#
# The top ghost is explicitly set to Neumann (ϕ[Nz+1] = ϕ[Nz]) before computing V∇².
# This prevents the MixedBoundaryCondition on p_Δt (set at model construction) from
# polluting the operator during CG iterations with a stale coefficient/inhomogeneity.
struct FreeSurfaceLaplacian end

@kernel function _fill_top_neumann_halo!(ϕ, grid)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz
    @inbounds ϕ[i, j, Nz+1] = ϕ[i, j, Nz]
end

@kernel function _apply_free_surface_correction!(∇²ϕ, grid, ϕ, Δt, g)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz
    Δzᶠ = Δzᵃᵃᶠ(i, j, Nz+1, grid)
    den = g * Δt^2 + Δzᶠ / 2
    Az = Azᶜᶜᶠ(i, j, Nz+1, grid)
    @inbounds ∇²ϕ[i, j, Nz] -= Az / den * ϕ[i, j, Nz]
end

function (::FreeSurfaceLaplacian)(∇²ϕ, ϕ, free_surface, Δt)
    grid = ϕ.grid
    arch = architecture(grid)
    fill_halo_regions!(ϕ)
    launch!(arch, grid, :xy, _fill_top_neumann_halo!, ϕ, grid)
    launch!(arch, grid, :xyz, _symmetric_laplacian_operator!, ∇²ϕ, grid, ϕ)
    g = free_surface.gravitational_acceleration
    launch!(arch, grid, :xy, _apply_free_surface_correction!, ∇²ϕ, grid, ϕ, Δt, g)
    return nothing
end

@kernel function subtract_and_mask!(a, grid, b)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    @inbounds a[i, j, k] = (a[i, j, k] - b) * active
end

function enforce_zero_mean_gauge!(x, r)
    grid = r.grid
    arch = architecture(grid)

    mean_x = mean(x)
    mean_r = mean(r)

    launch!(arch, grid, :xyz, subtract_and_mask!, x, grid, mean_x)
    launch!(arch, grid, :xyz, subtract_and_mask!, r, grid, mean_r)
end

@kernel function cell_volume!(V, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds V[i, j, k] = Vᶜᶜᶜ(i, j, k, grid)
end

function minimum_cell_volume(grid)
    V = CenterField(grid)
    arch = architecture(grid)
    launch!(arch, grid, :xyz, cell_volume!, V, grid)
    return minimum(V)
end

struct DefaultPreconditioner end

"""
    ConjugateGradientPoissonSolver(grid;
                                   preconditioner = DefaultPreconditioner(),
                                   reltol = sqrt(eps(grid)),
                                   abstol = sqrt(eps(grid)),
                                   enforce_gauge_condition! = enforce_zero_mean_gauge!,
                                   kw...)

Creates a `ConjugateGradientPoissonSolver` on `grid` using a `preconditioner`.
`ConjugateGradientPoissonSolver` is iterative, and will stop when both the relative error in the
pressure solution is smaller than `reltol` and the absolute error is smaller than `abstol`. Other
keyword arguments are passed to `ConjugateGradientSolver`.

Convergence is measured using a volume-inverse-weighted residual norm, `||V⁻¹r||₂`, which
normalizes out the cell volume scaling introduced by the symmetric volume-weighted Laplacian
operator `V∇²`. This makes convergence behavior independent of cell volume.

The Poisson solver has a zero mean gauge condition enforced with `enforce_gauge_condition! = enforce_zero_mean_gauge!`,
which pins the pressure field to have a mean of zero.
This is because the pressure field is defined only up to an arbitrary constant, and the zero mean gauge condition
is a common choice to remove this degree of freedom.
"""
function ConjugateGradientPoissonSolver(grid;
                                        linear_operation = compute_symmetric_laplacian!,
                                        preconditioner = DefaultPreconditioner(),
                                        reltol = sqrt(eps(grid)),
                                        abstol = sqrt(eps(grid)),
                                        enforce_gauge_condition! = enforce_zero_mean_gauge!,
                                        kw...)

    if preconditioner isa DefaultPreconditioner # try to make a useful default
        if grid isa ImmersedBoundaryGrid && grid.underlying_grid isa GridWithFFTSolver
            preconditioner = fft_poisson_solver(grid.underlying_grid)
        else
            preconditioner = DiagonallyDominantPreconditioner()
        end
    end

    rhs = CenterField(grid)

    volume_inverse_norm = VolumeInverseNorm(grid)

    conjugate_gradient_solver = ConjugateGradientSolver(linear_operation;
                                                        reltol,
                                                        abstol,
                                                        preconditioner,
                                                        template_field = rhs,
                                                        enforce_gauge_condition!,
                                                        residual_norm = volume_inverse_norm,
                                                        kw...)

    return ConjugateGradientPoissonSolver(grid, rhs, conjugate_gradient_solver)
end

#####
##### Preconditioner selection for free-surface CG solvers
#####

# For grids where x and y are both uniform, use FT with InhomogeneousFormulation in z
# so the preconditioner already encodes the Robin BC (best preconditioner for IBG + free surface).
# Note: per-type (not union) dispatches required because XYRegularRG <: XZRegularRG, causing
# union dispatches to be ambiguous.
fft_free_surface_preconditioner(grid::XYZRegularRG) =
    FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation=InhomogeneousFormulation(ZDirection()))

fft_free_surface_preconditioner(grid::XYRegularRG) =
    FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation=InhomogeneousFormulation(ZDirection()))

# For grids stretched in x or y, the z-direction InhomogeneousFormulation cannot be used
# (the non-tridiagonal directions must be uniform for FFT). Use the standard FT solver in the
# stretched direction (HomogeneousNeumann) as preconditioner; CG corrects for the free surface.
fft_free_surface_preconditioner(grid::XZRegularRG) = FourierTridiagonalPoissonSolver(grid)
fft_free_surface_preconditioner(grid::YZRegularRG) = FourierTridiagonalPoissonSolver(grid)

#####
##### A preconditioner based on the FFT solver
#####

@kernel function fft_preconditioner_rhs!(preconditioner_rhs, rhs, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = rhs[i, j, k] * V⁻¹ᶜᶜᶜ(i, j, k, grid)
end

@kernel function fourier_tridiagonal_preconditioner_rhs!(preconditioner_rhs, ::XDirection, grid, rhs)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = rhs[i, j, k] * V⁻¹ᶜᶜᶜ(i, j, k, grid)
end

@kernel function fourier_tridiagonal_preconditioner_rhs!(preconditioner_rhs, ::YDirection, grid, rhs)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = rhs[i, j, k] * V⁻¹ᶜᶜᶜ(i, j, k, grid)
end

@kernel function fourier_tridiagonal_preconditioner_rhs!(preconditioner_rhs, ::ZDirection, grid, rhs)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = rhs[i, j, k] * V⁻¹ᶜᶜᶜ(i, j, k, grid)
end

function compute_preconditioner_rhs!(solver::FFTBasedPoissonSolver, rhs)
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, fft_preconditioner_rhs!, solver.storage, rhs, grid)
    return nothing
end

function compute_preconditioner_rhs!(solver::FourierTridiagonalPoissonSolver, rhs)
    grid = solver.grid
    arch = architecture(grid)
    tridiagonal_dir = solver.batched_tridiagonal_solver.tridiagonal_direction
    launch!(arch, grid, :xyz, fourier_tridiagonal_preconditioner_rhs!,
            solver.storage, tridiagonal_dir, grid, rhs)
    return nothing
end

const FFTBasedPreconditioner = Union{FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver}

@inline function precondition!(p, preconditioner::FFTBasedPreconditioner, r, args...)
    compute_preconditioner_rhs!(preconditioner, r)
    solve!(p, preconditioner, preconditioner.storage)
    return p
end

#####
##### The "DiagonallyDominantPreconditioner" (Marshall et al 1997)
#####

struct DiagonallyDominantPreconditioner end
Base.summary(::DiagonallyDominantPreconditioner) = "DiagonallyDominantPreconditioner"

@inline function precondition!(p, ::DiagonallyDominantPreconditioner, r, args...)
    grid = r.grid
    arch = architecture(p)
    fill_halo_regions!(r)
    launch!(arch, grid, :xyz, _diagonally_dominant_precondition!, p, grid, r)

    return p
end

# Kernels that calculate coefficients for the preconditioner
@inline Ax⁻(i, j, k, grid) = Axᶠᶜᶜ(i,   j, k, grid) * Δx⁻¹ᶠᶜᶜ(i,   j, k, grid) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline Ax⁺(i, j, k, grid) = Axᶠᶜᶜ(i+1, j, k, grid) * Δx⁻¹ᶠᶜᶜ(i+1, j, k, grid) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline Ay⁻(i, j, k, grid) = Ayᶜᶠᶜ(i, j,   k, grid) * Δy⁻¹ᶜᶠᶜ(i, j,   k, grid) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline Ay⁺(i, j, k, grid) = Ayᶜᶠᶜ(i, j+1, k, grid) * Δy⁻¹ᶜᶠᶜ(i, j+1, k, grid) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline Az⁻(i, j, k, grid) = Azᶜᶜᶠ(i, j, k,   grid) * Δz⁻¹ᶜᶜᶠ(i, j, k,   grid) * V⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline Az⁺(i, j, k, grid) = Azᶜᶜᶠ(i, j, k+1, grid) * Δz⁻¹ᶜᶜᶠ(i, j, k+1, grid) * V⁻¹ᶜᶜᶜ(i, j, k, grid)

@inline Ac(i, j, k, grid) = - Ax⁻(i, j, k, grid) - Ax⁺(i, j, k, grid) -
                              Ay⁻(i, j, k, grid) - Ay⁺(i, j, k, grid) -
                              Az⁻(i, j, k, grid) - Az⁺(i, j, k, grid)

@inline heuristic_residual(i, j, k, grid, r) =
    @inbounds 1 / abs(Ac(i, j, k, grid)) * (r[i, j, k] - 2 * Ax⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i-1, j, k, grid)) * r[i-1, j, k] -
                                                         2 * Ax⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i+1, j, k, grid)) * r[i+1, j, k] -
                                                         2 * Ay⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j-1, k, grid)) * r[i, j-1, k] -
                                                         2 * Ay⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j+1, k, grid)) * r[i, j+1, k] -
                                                         2 * Az⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j, k-1, grid)) * r[i, j, k-1] -
                                                         2 * Az⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j, k+1, grid)) * r[i, j, k+1])

@kernel function _diagonally_dominant_precondition!(p, grid, r)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    @inbounds p[i, j, k] = heuristic_residual(i, j, k, grid, r) * active
end
