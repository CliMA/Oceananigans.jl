import Oceananigans.Solvers: solve_for_pressure!

function solve_for_pressure!(pressure, solver::DistributedFFTBasedPoissonSolver, arch, grid, Δt, U★)

    RHS = first(solver.storage)

    rhs_event = launch!(arch, grid, :xyz,
                        calculate_pressure_right_hand_side!, RHS, arch, grid, Δt, U★,
                        dependencies = Event(device(arch)))

    wait(device(arch), rhs_event)

    solve_poisson_equation!(solver)

    ϕ = first(solver.storage)

    copy_event = launch!(arch, grid, :xyz,
                         copy_pressure!, pressure, ϕ, arch, grid,
                         dependencies = Event(device(arch)))

    wait(device(arch), copy_event)

    return nothing
end
