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

"""
Calculate the right-hand-side of the Poisson equation for the non-hydrostatic pressure.
"""
@kernel function calculate_pressure_right_hand_side!(RHS, arch, grid::AbstractGrid{FT, TX, TY, TZ}, Δt, U★) where {FT, TX, TY, TZ}
    i, j, k = @index(Global, NTuple)

    @inbounds RHS[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

"""
Copy the non-hydrostatic pressure into `p` from the pressure solver.
"""
@kernel function copy_pressure!(p, ϕ, arch, grid::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    i, j, k = @index(Global, NTuple)

    @inbounds p[i, j, k] = real(ϕ[i, j, k])
end
