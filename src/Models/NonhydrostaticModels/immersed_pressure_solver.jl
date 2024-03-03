using Oceananigans
using Oceananigans.Operators

using Oceananigans.Architectures: device, architecture
using Oceananigans.Solvers: PreconditionedConjugateGradientSolver, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: inactive_cell
using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.Utils: launch!
using Oceananigans.Models.NonhydrostaticModels: PressureSolver, calculate_pressure_source_term_fft_based_solver!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, immersed_cell

using KernelAbstractions: @kernel, @index

import Oceananigans.Solvers: precondition!, solve!, build_preconditioner
import Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

struct ImmersedPoissonSolver{R, G, S, Z}
    rhs :: R
    grid :: G
    pcg_solver :: S
    storage :: Z
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

    return solve!(p, solver, solver.storage)
end

function ImmersedPoissonSolver(grid;
                               preconditioner = nothing,
                               reltol = eps(eltype(grid)),
                               solver_method = :PreconditionedConjugateGradient,
                               abstol = 0,
                               kw...)

    if preconditioner == "FFT"
        arch = architecture(grid)
        preconditioner = PressureSolver(arch, grid)
    end

    pcg_solver, rhs, storage = build_implicit_poisson_solver(Val(solver_method), grid; reltol, abstol, preconditioner, kw...)

    return ImmersedPoissonSolver(rhs, grid, pcg_solver, storage)
end

function build_implicit_poisson_solver(::Val{:PreconditionedConjugateGradient}, grid;
                                       reltol = eps(eltype(grid)),
                                       abstol = 0,
                                       preconditioner = nothing, 
                                       kw...)

    rhs = CenterField(grid)

    pcg_solver = PreconditionedConjugateGradientSolver(compute_laplacian!; reltol, abstol,
                                                       preconditioner,
                                                       template_field = rhs,
                                                       kw...)

    return pcg_solver, rhs, nothing
end

function build_implicit_poisson_solver(::Val{:HeptadiagonalIterativeSolver}, grid;
                                       reltol = eps(eltype(grid)),
                                       abstol = 0,
                                       preconditioner = nothing, 
                                       kw...)

    N = prod(size(grid))

    right_hand_side = arch_array(architecture(grid), zeros(eltype(grid), N))
    storage = deepcopy(right_hand_side)

    tolerance  = reltol 
    coeffs     = compute_poisson_weights(grid)

    pcg_solver = HeptadiagonalIterativeSolver(coeffs; 
                                              template = right_hand_side, 
                                              grid, 
                                              tolerance, 
                                              preconditioner_method = preconditioner,
                                              kw...)

    return pcg_solver, right_hand_side, storage
end

@kernel function _calculate_pressure_source_term!(rhs, grid, Δt, U★, ::Val{false})
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@kernel function _calculate_pressure_source_term!(rhs, grid, Δt, U★, ::Val{true})
    i, j, k = @index(Global, NTuple)
    t = i + grid.Nx * (j - 1 + grid.Ny * (k - 1))
    @inbounds rhs[t] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt * Vᶜᶜᶜ(i, j, k, grid)
end

@inline laplacianᶜᶜᶜ(i, j, k, grid, ϕ) = ∇²ᶜᶜᶜ(i, j, k, grid, ϕ)

@kernel function laplacian!(∇²ϕ, grid, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²ϕ[i, j, k] = laplacianᶜᶜᶜ(i, j, k, grid, ϕ)
end

function compute_laplacian!(∇²ϕ, ϕ)
    grid = ϕ.grid
    arch = architecture(grid)

    fill_halo_regions!(ϕ)

    launch!(arch, grid, :xyz, laplacian!, ∇²ϕ, grid, ϕ)

    return nothing
end

zero_weight_x(i, j, k, grid) = immersed_cell(i-1, j, k, grid) | immersed_cell(i, j, k, grid)
zero_weight_y(i, j, k, grid) = immersed_cell(i, j-1, k, grid) | immersed_cell(i, j, k, grid)
zero_weight_z(i, j, k, grid) = immersed_cell(i, j, k-1, grid) | immersed_cell(i, j, k, grid)

@kernel function _compute_poisson_weights(Ax, Ay, Az, grid)
    i, j, k = @index(Global, NTuple)
    Ax[i, j, k] = ifelse(zero_weight_x(i, j, k, grid), 0, Δzᶠᶜᶜ(i, j, k, grid) * Δyᶠᶜᶜ(i, j, k, grid) / Δxᶠᶜᶜ(i, j, k, grid))
    Ay[i, j, k] = ifelse(zero_weight_y(i, j, k, grid), 0, Δzᶜᶠᶜ(i, j, k, grid) * Δxᶜᶠᶜ(i, j, k, grid) / Δyᶜᶠᶜ(i, j, k, grid))
    Az[i, j, k] = ifelse(zero_weight_z(i, j, k, grid), 0, Δxᶜᶜᶠ(i, j, k, grid) * Δyᶜᶜᶠ(i, j, k, grid) / Δzᶜᶜᶠ(i, j, k, grid))
end

function compute_poisson_weights(grid)
    N = size(grid)
    Ax = arch_array(architecture(grid), zeros(N...))
    Ay = arch_array(architecture(grid), zeros(N...))
    Az = arch_array(architecture(grid), zeros(N...))
    C  = arch_array(architecture(grid), zeros(grid, N...))
    D  = arch_array(architecture(grid), zeros(grid, N...))

    launch!(architecture(grid), grid, :xyz, _compute_poisson_weights, Ax, Ay, Az, grid)
    
    return (Ax, Ay, Az, C, D)
end

linear_rhs(solver) = Val(false)
linear_rhs(::HeptadiagonalIterativeSolver) = Val(true)

function solve_for_pressure!(pressure, solver::ImmersedPoissonSolver, Δt, U★)
    # TODO: Is this the right criteria?
    min_Δt = eps(typeof(Δt))
    Δt <= min_Δt && return pressure

    rhs = solver.rhs
    grid = solver.grid
    arch = architecture(grid)

    if grid isa ImmersedBoundaryGrid
        underlying_grid = grid.underlying_grid
    else
        underlying_grid = grid
    end

    linear_rhs_kernel = linear_rhs(solver.pcg_solver)

    launch!(arch, grid, :xyz, _calculate_pressure_source_term!,
            rhs, underlying_grid, Δt, U★, linear_rhs_kernel)

    # mask_immersed_field!(rhs, zero(grid))

    storage = getstorage(pressure, solver.storage, solver.pcg_solver)

    # Solve pressure Pressure equation for pressure, given rhs
    # @info "Δt before pressure solve: $(Δt)"
    solve!(storage, solver.pcg_solver, rhs)

    reshape_solution!(pressure, storage, solver.pcg_solver)
    
    return pressure
end

solve!(storage, solver::HeptadiagonalIterativeSolver, rhs) = solve!(storage, solver, rhs, 1)

getstorage(pressure, storage, solver) = pressure
getstorage(pressure, storage, solver::HeptadiagonalIterativeSolver) = storage

reshape_solution!(pressure, storage, args...) = nothing
reshape_solution!(pressure, storage, ::HeptadiagonalIterativeSolver) = set!(pressure, reshape(storage, size(pressure)...))

struct DiagonallyDominantThreeDimensionalPreconditioner end

@inline function precondition!(P_r, ::DiagonallyDominantThreeDimensionalPreconditioner, r, args...)
    grid = r.grid
    arch = architecture(P_r)

    fill_halo_regions!(r)

    launch!(arch, grid, :xyz, _DiagonallyDominantThreeDimensional_precondition!,
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

@kernel function _DiagonallyDominantThreeDimensional_precondition!(P_r, grid, r)
    i, j, k = @index(Global, NTuple)
    @inbounds P_r[i, j, k] = heuristic_inverse_times_residuals(i, j, k, r, grid)
end