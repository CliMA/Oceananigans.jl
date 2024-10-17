using Oceananigans.Operators
using Statistics: mean

using KernelAbstractions: @kernel, @index

import Oceananigans.Architectures: architecture

struct ConjugateGradientPoissonSolver{G, R, S}
    grid :: G
    right_hand_side :: R
    conjugate_gradient_solver :: S
end

architecture(solver::ConjugateGradientPoissonSolver) = architecture(cgps.grid)
iteration(cgps::ConjugateGradientPoissonSolver) = iteration(cgps.conjugate_gradient_solver)

Base.summary(ips::ConjugateGradientPoissonSolver) =
    summary("ConjugateGradientPoissonSolver on ", summary(ips.grid))

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

struct RegularizedLaplacian{D}
    δ :: D
end

function (L::RegularizedLaplacian)(Lϕ, ϕ)
    grid = ϕ.grid
    arch = architecture(grid)
    fill_halo_regions!(ϕ)
    launch!(arch, grid, :xyz, laplacian!, Lϕ, grid, ϕ)

    if !isnothing(L.δ)
        # Add regularization
        ϕ̄ = mean(ϕ)
        parent(Lϕ) .+= L.δ * ϕ̄
    end

    return nothing
end

struct DefaultPreconditioner end

function ConjugateGradientPoissonSolver(grid;
                                        regularization = nothing,
                                        preconditioner = DefaultPreconditioner(),
                                        reltol = sqrt(eps(grid)),
                                        abstol = sqrt(eps(grid)),
                                        kw...)

    if preconditioner isa DefaultPreconditioner # try to make a useful default
        if has_fft_poisson_solver(grid)
            preconditioner = fft_poisson_solver(grid)
        else
            preconditioner = NearDiagonalPoissonPreconditioner()
        end
    end

    rhs = CenterField(grid)
    operator = RegularizedLaplacian(regularization)
    preconditioner = RegularizedPreconditioner(preconditioner, rhs, regularization)

    conjugate_gradient_solver = ConjugateGradientSolver(operator;
                                                        reltol,
                                                        abstol,
                                                        preconditioner,
                                                        template_field = rhs,
                                                        kw...)
        
    return ConjugateGradientPoissonSolver(grid, rhs, conjugate_gradient_solver)
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

struct RegularizedPreconditioner{P, R, D}
    preconditioner :: P
    rhs :: R
    regularization :: D
end

const SolverWithFFT = Union{FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver}
const FFTBasedPreconditioner = RegularizedPreconditioner{<:SolverWithFFT}

function precondition!(p, regularized::FFTBasedPreconditioner, r, args...)
    solver = regularized.preconditioner
    compute_preconditioner_rhs!(solver, r)
    p = solve!(p, solver)

    δ = regularized.regularization
    rhs = regularized.rhs
    mean_p = mean(p)

    if !isnothing(δ)
        mean_rhs = mean(rhs)
        Δp = mean_p - mean_rhs / δ
    else
        Δp = mean_p
    end

    grid = p.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, subtract_and_mask!, p, grid, Δp)

    return p
end

@kernel function subtract_and_mask!(a, grid, b)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    @inbounds a[i, j, k] = (a[i, j, k] - b) * active
end

#####
##### The "NearDiagonalPoissonPreconditioner" (Marshall et al 1997)
#####

struct NearDiagonalPoissonPreconditioner end
const RegularizedNDPP = RegularizedPreconditioner{<:NearDiagonalPoissonPreconditioner}
Base.summary(::NearDiagonalPoissonPreconditioner) = "NearDiagonalPoissonPreconditioner"

@inline function precondition!(p, preconditioner::RegularizedNDPP, r, args...)
    grid = r.grid
    arch = architecture(p)
    fill_halo_regions!(r)
    launch!(arch, grid, :xyz, _diagonally_dominant_precondition!, p, grid, r)

    δ = preconditioner.regularization
    rhs = preconditioner.rhs
    mean_p = mean(p)

    if !isnothing(δ)
        mean_rhs = mean(rhs)
        Δp = mean_p - mean_rhs / δ
    else
        Δp = mean_p
    end

    grid = p.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, subtract_and_mask!, p, grid, Δp)

    return p
end

# Kernels that calculate coefficients for the preconditioner
@inline Ax⁻(i, j, k, grid) = Axᶠᶜᶜ(i,   j, k, grid) / Δxᶠᶜᶜ(i,   j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ax⁺(i, j, k, grid) = Axᶠᶜᶜ(i+1, j, k, grid) / Δxᶠᶜᶜ(i+1, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ay⁻(i, j, k, grid) = Ayᶜᶠᶜ(i, j,   k, grid) / Δyᶜᶠᶜ(i, j,   k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ay⁺(i, j, k, grid) = Ayᶜᶠᶜ(i, j+1, k, grid) / Δyᶜᶠᶜ(i, j+1, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Az⁻(i, j, k, grid) = Azᶜᶜᶠ(i, j, k,   grid) / Δzᶜᶜᶠ(i, j, k,   grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Az⁺(i, j, k, grid) = Azᶜᶜᶠ(i, j, k+1, grid) / Δzᶜᶜᶠ(i, j, k+1, grid) / Vᶜᶜᶜ(i, j, k, grid)

@inline Ac(i, j, k, grid) = - Ax⁻(i, j, k, grid) - Ax⁺(i, j, k, grid) -
                              Ay⁻(i, j, k, grid) - Ay⁺(i, j, k, grid) -
                              Az⁻(i, j, k, grid) - Az⁺(i, j, k, grid)
                              
@inline heuristic_residual(i, j, k, grid, r) =
    @inbounds 1 / Ac(i, j, k, grid) * (r[i, j, k] - 2 * Ax⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i-1, j, k, grid)) * r[i-1, j, k] -
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

