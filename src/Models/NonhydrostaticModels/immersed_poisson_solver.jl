using Oceananigans.Architectures: device, architecture
using Oceananigans.Solvers: PreconditionedConjugateGradientSolver, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: inactive_cell
using Oceananigans.Operators: divᶜᶜᶜ, ∇²ᶜᶜᶜ 
using Oceananigans.Utils: launch!, prettysummary
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, immersed_cell
using Statistics: mean

using KernelAbstractions: @kernel, @index

import Oceananigans.Solvers: precondition!

struct ImmersedPoissonSolver{G, R, S}
    grid :: G
    rhs :: R
    pcg_solver :: S
end

Base.summary(ips::ImmersedPoissonSolver) = summary("ImmersedPoissonSolver on ", summary(ips.grid.underlying_grid))

function Base.show(io::IO, ips::ImmersedPoissonSolver)
    A = architecture(ips.grid)
    print(io, "ImmersedPoissonSolver:", '\n',
              "├── grid: ", summary(ips.grid), '\n',
              "│   └── immersed_boundary: ", prettysummary(ips.grid.immersed_boundary), '\n',
              "└── pcg_solver: ", summary(ips.pcg_solver), '\n',
              "    ├── maxiter: ", prettysummary(ips.pcg_solver.maxiter), '\n',
              "    ├── reltol: ", prettysummary(ips.pcg_solver.reltol), '\n',
              "    ├── abstol: ", prettysummary(ips.pcg_solver.abstol), '\n',
              "    ├── preconditioner: ", prettysummary(ips.pcg_solver.preconditioner), '\n',
              "    └── iteration: ", prettysummary(ips.pcg_solver.iteration))
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
    
    return p
end

function ImmersedPoissonSolver(ibg::ImmersedBoundaryGrid;
                               preconditioner = nonhydrostatic_pressure_solver(ibg.underlying_grid),
                               reltol = sqrt(eps(ibg)),
                               abstol = 0,
                               kw...)

    rhs = CenterField(ibg)
    pcg_solver = PreconditionedConjugateGradientSolver(compute_laplacian!;
                                                       reltol,
                                                       abstol,
                                                       preconditioner,
                                                       template_field = rhs,
                                                       kw...)

    return ImmersedPoissonSolver(ibg, rhs, pcg_solver)
end

nonhydrostatic_pressure_solver(arch, ibg::ImmersedBoundaryGrid) = ImmersedPoissonSolver(ibg)

@kernel function calculate_pressure_source_term!(rhs, grid, Δt, U★)
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
    arch = architecture(ibg)
    underlying_grid = ibg.underlying_grid

    launch!(arch, underlying_grid, :xyz,
            calculate_pressure_source_term!, rhs, underlying_grid, Δt, U★)

    # Solve pressure Pressure equation for pressure, given rhs
    # @info "Δt before pressure solve: $(Δt)"
    solve!(pressure, solver.pcg_solver, rhs)

    return pressure
end

#####
##### The "DiagonallyDominantPreconditioner" used by MITgcm
#####

struct DiagonallyDominantPreconditioner end
Base.summary(::DiagonallyDominantPreconditioner) = "DiagonallyDominantPreconditioner"

@inline function precondition!(P_r, ::DiagonallyDominantPreconditioner, r, args...)
    grid = r.grid
    arch = architecture(P_r)
    fill_halo_regions!(r)
    launch!(arch, grid, :xyz, _diagonally_dominant_precondition!, P_r, grid, r)
    return P_r
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

@kernel function _diagonally_dominant_precondition!(P_r, grid, r)
    i, j, k = @index(Global, NTuple)
    @inbounds P_r[i, j, k] = heuristic_residual(i, j, k, grid, r)
end
