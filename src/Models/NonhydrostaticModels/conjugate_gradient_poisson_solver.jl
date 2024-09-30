using Oceananigans.Architectures: device, architecture
using Oceananigans.Solvers: PreconditionedConjugateGradientSolver, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: inactive_cell
using Oceananigans.Operators: divᶜᶜᶜ, ∇²ᶜᶜᶜ 
using Oceananigans.Utils: launch!, prettysummary
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Statistics: mean

using KernelAbstractions: @kernel, @index

import Oceananigans.Solvers: precondition!
import ..Models: iteration

struct ConjugateGradientPoissonSolver{G, R, S}
    grid :: G
    right_hand_side :: R
    conjugate_gradient_solver :: S
end

iteration(cgps::ConjugateGradientPoissonSolver) = cgps.conjugate_gradient_solver.iteration

Base.summary(ips::ConjugateGradientPoissonSolver) =
    summary("ConjugateGradientPoissonSolver on ", summary(ips.grid))

function Base.show(io::IO, ips::ConjugateGradientPoissonSolver)
    A = architecture(ips.grid)
    print(io, "ConjugateGradientPoissonSolver:", '\n',
              "├── grid: ", summary(ips.grid), '\n',
              "│   └── immersed_boundary: ", prettysummary(ips.grid.immersed_boundary), '\n',
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

function ConjugateGradientPoissonSolver(grid;
                                        preconditioner = nothing,
                                        reltol = sqrt(eps(grid)),
                                        abstol = 0,
                                        kw...)

    if isnothing(preconditioner) # make a useful default
        if grid isa ImmersedBoundaryGrid && grid.underlying_grid isa GridWithFFT
            if grid.underlying_grid isa XYZRegularRG
                preconditioner = FFTBasedPoissonSolver(grid.underlying_grid)
            else # it's stretched in one direction
                preconditioner = FourierTridiagonalPoissonSolver(grid.underlying_grid)
            end
        else
            preconditioner = DiagonallyDominantPreconditioner()
        end
    end

    rhs = CenterField(grid)

    conjugate_gradient_solver =
        PreconditionedConjugateGradientSolver(compute_laplacian!;
                                              reltol,
                                              abstol,
                                              preconditioner,
                                              template_field = rhs,
                                              kw...)

    return ConjugateGradientPoissonSolver(grid, rhs, conjugate_gradient_solver)
end

function solve_for_pressure!(pressure, solver::ConjugateGradientPoissonSolver, Δt, U★)
    rhs = solver.right_hand_side
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _compute_source_term!, rhs, grid, Δt, U★)
    return solve!(pressure, solver.conjugate_gradient_solver, rhs)
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
            solver.storage, tridiagonal_dir, rhs)
    return nothing
end

function precondition!(p, solver, rhs, args...)
    compute_preconditioner_rhs!(solver, rhs)
    p = solve!(p, solver)

    P = mean(p)
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, subtract_and_mask!, p, grid, P)

    return p
end

@kernel function subtract_and_mask!(a, grid, b)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    a[i, j, k] = (a[i, j, k] - b) * active
end

#####
##### The "DiagonallyDominantPreconditioner" used by MITgcm
#####

struct DiagonallyDominantPreconditioner end
Base.summary(::DiagonallyDominantPreconditioner) = "DiagonallyDominantPreconditioner"

@inline function precondition!(p, ::DiagonallyDominantPreconditioner, r, args...)
    grid = r.grid
    arch = architecture(p)
    fill_halo_regions!(r)
    launch!(arch, grid, :xyz, _diagonally_dominant_precondition!, p, grid, r)

    P = mean(p)
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, subtract_and_mask!, p, grid, P)

    return p
end

# Kernels that calculate coefficients for the preconditioner
@inline Ax⁻(i, j, k, grid) = Axᶠᶜᶜ(i,   j, k, grid) / Δxᶠᶜᶜ(i,   j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ax⁺(i, j, k, grid) = Axᶠᶜᶜ(i+1, j, k, grid) / Δxᶠᶜᶜ(i+1, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)

@inline Ay⁻(i, j, k, grid) = Ayᶜᶠᶜ(i, j,   k, grid) / Δyᶜᶠᶜ(i, j,   k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ay⁺(i, j, k, grid) = Ayᶜᶠᶜ(i, j+1, k, grid) / Δyᶜᶠᶜ(i, j+1, k, grid) / Vᶜᶜᶜ(i, j, k, grid)

@inline Az⁻(i, j, k, grid) = Azᶜᶜᶠ(i, j, k,   grid) / Δzᶜᶜᶠ(i, j, k,   grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Az⁺(i, j, k, grid) = Azᶜᶜᶠ(i, j, k+1, grid) / Δzᶜᶜᶠ(i, j, k+1, grid) / Vᶜᶜᶜ(i, j, k, grid)

@inline Ac(i, j, k, grid) = - Ax⁻(i, j, k, grid) -
                              Ax⁺(i, j, k, grid) -
                              Ay⁻(i, j, k, grid) -
                              Ay⁺(i, j, k, grid) -
                              Az⁻(i, j, k, grid) -
                              Az⁺(i, j, k, grid)

@inline heuristic_residual(i, j, k, grid, r) =
    @inbounds 1 / Ac(i, j, k, grid) * (r[i, j, k] - 2 * Ax⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i-1, j, k, grid)) * r[i-1, j, k] -
                                                    2 * Ax⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i+1, j, k, grid)) * r[i+1, j, k] -
                                                    2 * Ay⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j-1, k, grid)) * r[i, j-1, k] -
                                                    2 * Ay⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j+1, k, grid)) * r[i, j+1, k] -
                                                    2 * Az⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j, k-1, grid)) * r[i, j, k-1] -
                                                    2 * Az⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j, k+1, grid)) * r[i, j, k+1])

@kernel function _diagonally_dominant_precondition!(p, grid, r)
    i, j, k = @index(Global, NTuple)
    @inbounds p[i, j, k] = heuristic_residual(i, j, k, grid, r)
end
