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

@kernel function compute_source_term!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    δ = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w)
    inactive = !inactive_cell(i, j, k, grid)
    @inbounds rhs[i, j, k] = δ / Δt * inactive
end

function solve_for_pressure!(pressure, solver::ConjugateGradientPoissonSolver, Δt, U★)
    # We may want a criteria like this:
    # min_Δt = eps(typeof(Δt))
    # Δt <= min_Δt && return pressure

    rhs = solver.right_hand_side
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, compute_source_term!, rhs, grid, Δt, U★)

    # Solve pressure Pressure equation for pressure, given rhs
    # @info "Δt before pressure solve: $(Δt)"
    solve!(pressure, solver.conjugate_gradient_solver, rhs)

    return pressure
end

#####
##### A preconditioner based on the FFT solver
#####

@kernel function fft_preconditioner_right_hand_side!(preconditioner_rhs, rhs)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = rhs[i, j, k]
end

function precondition!(p, solver::FFTBasedPoissonSolver, rhs, args...)
    grid = solver.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, fft_preconditioner_right_hand_side!, solver.storage, rhs)
    p = solve!(p, solver, solver.storage)
    return p
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
