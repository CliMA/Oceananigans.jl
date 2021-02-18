using Oceananigans.Operators

function solve_for_pressure!(pressure, solver, arch, grid, Δt, U★)

    ϕ = RHS = solver.storage

    rhs_event = launch!(arch, grid, :xyz,
                        calculate_pressure_right_hand_side!, RHS, arch, grid, Δt, U★,
                        dependencies = Event(device(arch)))

    wait(device(arch), rhs_event)

    solve_poisson_equation!(solver)

    copy_event = launch!(arch, grid, :xyz,
                         copy_pressure!, pressure, ϕ, arch, grid,
                         dependencies = Event(device(arch)))

    wait(device(arch), copy_event)

    return nothing
end

function solve_for_pressure!(pressure, solver::PreconditionedConjugateGradientSolver, arch, grid, Δt, U★)
    RHS = solver.settings.RHS
    rhs_event = launch!(arch, grid, :xyz,
                        calculate_pressure_right_hand_side!, RHS, arch, grid, Δt, U★,
                        dependencies = Event(device(arch)))
    wait(device(arch), rhs_event)
    fill_halo_regions!(RHS, solver.settings.bcs, arch, grid)

    x = solver.settings.x
    x .= 0

    solve_poisson_equation!(solver, RHS, x)

    fill_halo_regions!(x, solver.settings.bcs, arch, grid)
    pressure.data .= x

    return nothing
end

"""
Calculate the right-hand-side of the Poisson equation for the non-hydrostatic pressure.
"""
@kernel function calculate_pressure_right_hand_side!(RHS, arch, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)

    @inbounds RHS[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

"""
Copy the non-hydrostatic pressure into `p` from the pressure solver.
"""
@kernel function copy_pressure!(p, ϕ, arch, grid)
    i, j, k = @index(Global, NTuple)

    @inbounds p[i, j, k] = real(ϕ[i, j, k])
end
