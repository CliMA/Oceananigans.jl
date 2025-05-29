using Oceananigans.Operators
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Statistics: mean
using Oceananigans.Grids: XYZRegularRG

using KernelAbstractions: @kernel, @index

import Oceananigans.Architectures: architecture

struct ConjugateGradientPoissonSolver{G, R, S}
    grid :: G
    right_hand_side :: R
    conjugate_gradient_solver :: S
    symmetrized :: Bool
end

architecture(solver::ConjugateGradientPoissonSolver) = architecture(cgps.grid)
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

@kernel function laplacian!(∇²ϕ, grid, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²ϕ[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, ϕ)
end

function compute_laplacian!(∇²ϕ, ϕ)
    grid = ϕ.grid
    arch = architecture(grid)
    fill_halo_regions!(ϕ)
    launch!(arch, grid, :xyz, laplacian!, ∇²ϕ, grid, ϕ)
    return nothing
end

@kernel function multiply_with_sqrt_spacing_operator(a, grid, operator)
    i, j, k = @index(Global, NTuple)
    a[i, j, k] = sqrt(operator(i, j, k, grid)) * a[i, j, k]
end

# @kernel function symmetric_laplacian!(∇²ϕ, grid, ϕ)
#     i, j, k = @index(Global, NTuple)

#     @inbounds ϕ[i, j, k] = sqrt(V⁻¹ᶜᶜᶜ(i, j, k, grid)) * ϕ[i, j, k]
#     @inbounds ∇²ϕ[i, j, k] = sqrt(Vᶜᶜᶜ(i, j, k, grid)) * ∇²ᶜᶜᶜ(i, j, k, grid, ϕ)
#     @inbounds ϕ[i, j, k] = sqrt(Vᶜᶜᶜ(i, j, k, grid)) * ϕ[i, j, k]
# end

function compute_symmetric_laplacian!(∇²ϕ, ϕ)
    grid = ϕ.grid
    arch = architecture(grid)
    
    # launch!(arch, grid, :xyz, symmetric_laplacian!, ∇²ϕ, grid, ϕ)
    launch!(arch, grid, :xyz, multiply_with_sqrt_spacing_operator, ϕ, grid, V⁻¹ᶜᶜᶜ)
    fill_halo_regions!(ϕ)
    launch!(arch, grid, :xyz, laplacian!, ∇²ϕ, grid, ϕ)
    launch!(arch, grid, :xyz, multiply_with_sqrt_spacing_operator, ∇²ϕ, grid, Vᶜᶜᶜ)
    launch!(arch, grid, :xyz, multiply_with_sqrt_spacing_operator, ϕ, grid, Vᶜᶜᶜ)
    return nothing
end

struct DefaultPreconditioner end

function symmetrize_or_not(grid)
    if grid isa ImmersedBoundaryGrid && grid.underlying_grid isa XYZRegularRG
        symmetrized = false
    elseif grid isa XYZRegularRG
        symmetrized = false
    else
        symmetrized = true
    end
    return symmetrized
end

"""
    ConjugateGradientPoissonSolver(grid;
                                   preconditioner = DefaultPreconditioner(),
                                   reltol = sqrt(eps(grid)),
                                   abstol = sqrt(eps(grid)),
                                   kw...)

Creates a `ConjugateGradientPoissonSolver` on `grid` using a `preconditioner`.
`ConjugateGradientPoissonSolver` is iterative, and will stop when both the relative error in the
pressure solution is smaller than `reltol` and the absolute error is smaller than `abstol`. Other
keyword arguments are passed to `ConjugateGradientSolver`.
"""
function ConjugateGradientPoissonSolver(grid;
                                        preconditioner = DefaultPreconditioner(),
                                        reltol = sqrt(eps(grid)),
                                        abstol = sqrt(eps(grid)),
                                        symmetrized = symmetrize_or_not(grid),
                                        enforce_gauge_condition = !symmetrize_or_not(grid),
                                        kw...)

    if preconditioner isa DefaultPreconditioner # try to make a useful default
        if grid isa ImmersedBoundaryGrid && grid.underlying_grid isa GridWithFFTSolver
            preconditioner = fft_poisson_solver(grid.underlying_grid)
        else
            preconditioner = DiagonallyDominantPreconditioner()
        end
    end

    rhs = CenterField(grid)

    linear_operation = symmetrized ? compute_symmetric_laplacian! : compute_laplacian!

    conjugate_gradient_solver = ConjugateGradientSolver(linear_operation;
                                                        reltol,
                                                        abstol,
                                                        preconditioner,
                                                        template_field = rhs,
                                                        enforce_gauge_condition,
                                                        kw...)

    return ConjugateGradientPoissonSolver(grid, rhs, conjugate_gradient_solver, symmetrized)
end

#####
##### A preconditioner based on the FFT solver
#####

@kernel function fft_preconditioner_rhs!(preconditioner_rhs, rhs)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = rhs[i, j, k]
end

@kernel function fourier_tridiagonal_preconditioner_rhs!(preconditioner_rhs, ::XDirection, grid, rhs)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = Δxᶜᶜᶜ(i, j, k, grid) * rhs[i, j, k]
end

@kernel function fourier_tridiagonal_preconditioner_rhs!(preconditioner_rhs, ::YDirection, grid, rhs)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = Δyᶜᶜᶜ(i, j, k, grid) * rhs[i, j, k]
end

@kernel function fourier_tridiagonal_preconditioner_rhs!(preconditioner_rhs, ::ZDirection, grid, rhs)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = Δzᶜᶜᶜ(i, j, k, grid) * rhs[i, j, k]
end

function compute_preconditioner_rhs!(solver::FFTBasedPoissonSolver, rhs)
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, fft_preconditioner_rhs!, solver.storage, rhs)
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

function precondition!(p, preconditioner::FFTBasedPreconditioner, r, args...)
    compute_preconditioner_rhs!(preconditioner, r)
    shift = - sqrt(eps(eltype(r))) # to make the operator strictly negative definite
    # solve!(p, preconditioner, preconditioner.storage, shift)
    solve!(p, preconditioner, preconditioner.storage)
    p .*= -1

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
