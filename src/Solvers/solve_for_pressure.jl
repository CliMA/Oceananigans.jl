using Oceananigans.Operators
using Oceananigans.Architectures: device_event

#####
##### Calculate the right-hand-side of the Poisson equation for the non-hydrostatic pressure.
#####

@kernel function calculate_pressure_source_term_fft_based_solver!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@kernel function calculate_pressure_source_term_fourier_tridiagonal_solver!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = Δzᵃᵃᶜ(i, j, k, grid) * divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

#####
##### Solve for pressure
#####

function solve_for_pressure!(pressure, solver, Δt, U★)
    rhs = calculate_pressure_poisson_rhs!(solver, Δt, U★)
    solve!(pressure, solver, rhs)
    return nothing
end

function calculate_pressure_poisson_rhs!(solver::FFTBasedPoissonSolver, Δt, U★)
    rhs = solver.storage
    arch = solver.architecture
    grid = solver.grid

    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fft_based_solver!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    return rhs
end

function calculate_pressure_poisson_rhs!(solver::FourierTridiagonalPoissonSolver, Δt, U★)
    rhs = solver.source_term
    arch = solver.architecture
    grid = solver.grid

    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term_fourier_tridiagonal_solver!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    return rhs
end

