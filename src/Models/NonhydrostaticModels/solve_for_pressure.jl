using Oceananigans.Operators
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.Architectures: device_event
using Oceananigans.Distributed: DistributedFFTBasedPoissonSolver

using PencilArrays: Permutation

#####
##### Calculate the right-hand-side of the non-hydrostatic pressure Poisson equation.
#####

const ZXYPermutation = Permutation{(3, 1, 2), 3}
const ZYXPermutation = Permutation{(3, 2, 1), 3}

@kernel function calculate_pressure_source_term_fft_based_solver!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@kernel function calculate_permuted_pressure_source_term_fft_based_solver!(rhs, grid, Δt, U★, ::ZXYPermutation)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[k, i, j] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@kernel function calculate_permuted_pressure_source_term_fft_based_solver!(rhs, grid, Δt, U★, ::ZYXPermutation)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[k, j, i] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@kernel function calculate_pressure_source_term_fourier_tridiagonal_solver!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = Δzᶜᶜᶜ(i, j, k, grid) * divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

#####
##### Solve for pressure
#####

function solve_for_pressure!(pressure, solver::DistributedFFTBasedPoissonSolver, Δt, U★)
    rhs = parent(first(solver.storage))
    arch = architecture(solver)
    grid = solver.local_grid

    rhs_event = launch!(arch, grid, :xyz, calculate_permuted_pressure_source_term_fft_based_solver!,
                        rhs, grid, Δt, U★, solver.input_permutation, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    # Solve pressure Poisson equation for pressure, given rhs
    solve!(pressure, solver)

    return pressure
end

function solve_for_pressure!(pressure, solver::FFTBasedPoissonSolver, Δt, U★)

    # Calculate right hand side:
    rhs = solver.storage
    arch = architecture(solver)
    grid = solver.grid

    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fft_based_solver!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    # Solve pressure Poisson given for pressure, given rhs
    solve!(pressure, solver, rhs)

    return nothing
end

function solve_for_pressure!(pressure, solver::FourierTridiagonalPoissonSolver, Δt, U★)

    # Calculate right hand side:
    rhs = solver.source_term
    arch = architecture(solver)
    grid = solver.grid

    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fourier_tridiagonal_solver!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    # Pressure Poisson rhs, scaled by Δzᶜᶜᶜ, is stored in solver.source_term:
    solve!(pressure, solver)

    return nothing
end
