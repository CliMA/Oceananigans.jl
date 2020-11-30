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
Calculate the right-hand-side of the Poisson equation for the non-hydrostatic
pressure and in the process apply the permutation along any direction we need
to perform a GPU fast cosine transform algorithm.
"""
@kernel function calculate_pressure_right_hand_side!(RHS, arch, grid::AbstractGrid{FT, TX, TY, TZ}, Δt, U★) where {FT, TX, TY, TZ}
    i, j, k = @index(Global, NTuple)

    i′ = permute_index(arch, TX(), i, grid.Nx)
    j′ = permute_index(arch, TY(), j, grid.Ny)
    k′ = permute_index(arch, TZ(), k, grid.Nz)

    @inbounds RHS[i′, j′, k′] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

"""
Copy the non-hydrostatic pressure into `p` from `solver.storage` and
undo the permutation along any dimensions in which a GPU fast cosine
transform was performed.
"""
@kernel function copy_pressure!(p, ϕ, arch, grid::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    i, j, k = @index(Global, NTuple)

    i′ = unpermute_index(arch, TX(), i, grid.Nx)
    j′ = unpermute_index(arch, TY(), j, grid.Ny)
    k′ = unpermute_index(arch, TZ(), k, grid.Nz)

    @inbounds p[i′, j′, k′] = real(ϕ[i, j, k])
end
