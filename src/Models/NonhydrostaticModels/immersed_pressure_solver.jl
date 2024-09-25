using Oceananigans
using Oceananigans.Operators

using Oceananigans.Architectures: device, architecture
using Oceananigans.Solvers: PreconditionedConjugateGradientSolver, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: inactive_cell
using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.Utils: launch!
using Oceananigans.Models.NonhydrostaticModels: PressureSolver, calculate_pressure_source_term_fft_based_solver!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

using KernelAbstractions: @kernel, @index

import Oceananigans.Solvers: precondition!
import Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

struct ImmersedPoissonSolver{R, G, S}
    rhs :: R
    grid :: G
    pcg_solver :: S
end

@kernel function fft_preconditioner_right_hand_side!(preconditioner_rhs, rhs)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = rhs[i, j, k]
end

# FFTBasedPoissonPreconditioner
function precondition!(p, solver::FFTBasedPoissonSolver, rhs, args...)
    grid = solver.grid
    arch = architecture(grid)

    launch!(arch, grid, :xyz,
            fft_preconditioner_right_hand_side!,
            solver.storage, rhs)

    p = solve!(p, solver, solver.storage)
    
    #=
    # Rescale the 0th eigenvalue of p
    Lx = grid.Lx
    Ly = grid.Ly
    Lz = grid.Lz
    m₀⁻¹ = - Lx*Ly + Ly*Lz + Lz*Lx
    R = mean(parent(rhs))
    parent(p) .+= m₀⁻¹ * R
    mask_immersed_field!(p, zero(grid))
    =#

    return p
end

function ImmersedPoissonSolver(ibg::ImmersedBoundaryGrid;
                               preconditioner = :FFT,
                               reltol = sqrt(eps(grid)),
                               abstol = 0,
                               kw...)

    if preconditioner == :FFT
        arch = architecture(grid)
        preconditioner = PressureSolver(arch, ibg.underlying_grid)
    end

    rhs = CenterField(grid)
    pcg_solver = PreconditionedConjugateGradientSolver(compute_laplacian!;
                                                       reltol,
                                                       abstol,
                                                       preconditioner,
                                                       template_field = rhs,
                                                       kw...)

    return ImmersedPoissonSolver(rhs, grid, pcg_solver)
end

PressureSolver(arch, ibg::ImmersedBoundaryGrid) = ImmersedPoissonSolver(ibg)

@kernel function calculate_pressure_source_term!(rhs, ibg, Δt, U★)
    i, j, k = @index(Global, NTuple)
    δ = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w)
    not_immersed = !immersed_cell(i, j, k, grid)
    @inbounds rhs[i, j, k] = δ / Δt * not_immersed
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

function solve_for_pressure!(pressure, solver::ImmersedPoissonSolver, Δt, U★)
    # TODO: Is this the right criteria?
    min_Δt = eps(typeof(Δt))
    Δt <= min_Δt && return pressure

    rhs = solver.rhs
    ibg = solver.grid
    arch = architecture(grid)
    underlying_grid = ibg.underlying_grid

    # if grid isa ImmersedBoundaryGrid
    #     underlying_grid = grid.underlying_grid
    # else
    #     underlying_grid = grid
    # end

    launch!(arch, underlying_grid, :xyz,
            calculate_pressure_source_term!,
            rhs, underlying_grid, Δt, U★)

    # mask_immersed_field!(rhs, zero(grid))

    # Solve pressure Pressure equation for pressure, given rhs
    # @info "Δt before pressure solve: $(Δt)"
    solve!(pressure, solver.pcg_solver, rhs)

    return pressure
end

#####
##### The "DiagonallyDominantPreconditioner" used by MITgcm
#####

struct DiagonallyDominantPreconditioner end

@inline function precondition!(P_r, ::DiagonallyDominantPreconditioner, r, args...)
    grid = r.grid
    arch = architecture(P_r)

    fill_halo_regions!(r)

    launch!(arch, grid, :xyz, _diagonally_dominant_precondition!!,
            P_r, grid, r)

    return P_r
end

# Kernels that calculate coefficients for the preconditioner
@inline Ax⁻(i, j, k, grid) = Axᶠᶜᶜ(i, j, k, grid) / Δxᶠᶜᶜ(i, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ay⁻(i, j, k, grid) = Ayᶜᶠᶜ(i, j, k, grid) / Δyᶜᶠᶜ(i, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Az⁻(i, j, k, grid) = Azᶜᶜᶠ(i, j, k, grid) / Δzᶜᶜᶠ(i, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ax⁺(i, j, k, grid) = Axᶠᶜᶜ(i+1, j, k, grid) / Δxᶠᶜᶜ(i+1, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ay⁺(i, j, k, grid) = Ayᶜᶠᶜ(i, j+1, k, grid) / Δyᶜᶠᶜ(i, j+1, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Az⁺(i, j, k, grid) = Azᶜᶜᶠ(i, j, k+1, grid) / Δzᶜᶜᶠ(i, j, k+1, grid) / Vᶜᶜᶜ(i, j, k, grid)

@inline Ac(i, j, k, grid) = - (Ax⁻(i, j, k, grid) +
                               Ax⁺(i, j, k, grid) +
                               Ay⁻(i, j, k, grid) +
                               Ay⁺(i, j, k, grid) +
                               Az⁻(i, j, k, grid) +
                               Az⁺(i, j, k, grid))

@inline heuristic_inverse_times_residuals(i, j, k, r, grid) =
    @inbounds 1 / Ac(i, j, k, grid) * (r[i, j, k] - 2 * Ax⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i-1, j, k, grid)) * r[i-1, j, k] -
                                                    2 * Ax⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i+1, j, k, grid)) * r[i+1, j, k] -
                                                    2 * Ay⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j-1, k, grid)) * r[i, j-1, k] -
                                                    2 * Ay⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j+1, k, grid)) * r[i, j+1, k] -
                                                    2 * Az⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j, k-1, grid)) * r[i, j, k-1] -
                                                    2 * Az⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j, k+1, grid)) * r[i, j, k+1])

@kernel function _diagonally_dominant_precondition!!(P_r, grid, r)
    i, j, k = @index(Global, NTuple)
    @inbounds P_r[i, j, k] = heuristic_inverse_times_residuals(i, j, k, r, grid)
end
