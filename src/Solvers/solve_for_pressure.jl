using Oceananigans.Operators

function solve_for_pressure!(pressure, solver, arch, grid, Δt, U★)
    
    calculate_pressure_right_hand_side!(solver, arch, grid, Δt, U★)
    
    solve_poisson_equation!(solver)

    copy_pressure!(pressure, solver, arch, grid)
    
    return nothing
end

#####
##### Calculate the right-hand-side of the Poisson equation for the non-hydrostatic pressure.
#####

@kernel function calculate_pressure_source_term_fft_based_solver!(RHS, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)

    @inbounds RHS[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

source_term_storage(solver::FFTBasedPoissonSolver) = solver.storage
source_term_storage(solver::FourierTridiagonalPoissonSolver) = solver.batched_tridiagonal_solver.f

source_term_kernel(::FFTBasedPoissonSolver) = calculate_pressure_source_term_fft_based_solver!
source_term_kernel(::FourierTridiagonalPoissonSolver) = calculate_pressure_source_term_fourier_tridiagonal_solver!

function calculate_pressure_right_hand_side!(solver, arch, grid, Δt, U★)
    RHS = source_term_storage(solver)
    rhs_event = launch!(arch, grid, :xyz, source_term_kernel(solver), RHS, grid, Δt, U★,
                        dependencies = Event(device(arch)))

    wait(device(arch), rhs_event)

    return nothing
end

#####
##### Copy the non-hydrostatic pressure into `p` from the pressure solver.
#####

@kernel function copy_pressure_kernel!(p, ϕ)
    i, j, k = @index(Global, NTuple)

    @inbounds p[i, j, k] = real(ϕ[i, j, k])
end

solution_storage(solver) = solver.storage

function copy_pressure!(p, solver, arch, grid)
    ϕ = solution_storage(solver)
    copy_event = launch!(arch, grid, :xyz, copy_pressure_kernel!, p, ϕ,
                         dependencies = Event(device(arch)))

    wait(device(arch), copy_event)
end
