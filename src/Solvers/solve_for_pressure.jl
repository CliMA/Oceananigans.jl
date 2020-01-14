using Oceananigans.Operators

function solve_for_pressure!(pressure, solver, arch, grid, U, G, Δt)
    if solver.type isa Channel && arch isa GPU
        ϕ = RHS = solver.storage.storage1
    else
        ϕ = RHS = solver.storage
    end

    @launch(device(arch), config=launch_config(grid, :xyz),
            calculate_pressure_right_hand_side!(RHS, solver.type, arch, grid, U, G, Δt))

    solve_poisson_equation!(solver, grid)

    @launch(device(arch), config=launch_config(grid, :xyz),
            copy_pressure!(pressure, ϕ, solver.type, arch, grid))

    return nothing
end

"""
Calculate the right-hand-side of the Poisson equation for the non-hydrostatic
pressure and in the process apply the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

along any direction we need to perform a GPU fast cosine transform algorithm.
"""
function calculate_pressure_right_hand_side!(RHS, solver_type, arch, grid, U, G, Δt)
    @loop_xyz i j k grid begin
        i′, j′, k′ = permute_index(solver_type, arch, i, j, k, grid.Nx, grid.Ny, grid.Nz)

        @inbounds RHS[i′, j′, k′] = divᶜᶜᶜ(i, j, k, grid, U.u, U.v, U.w) / Δt +
                                    divᶜᶜᶜ(i, j, k, grid, G.u, G.v, G.w)
    end
    return nothing
end

"""
Copy the non-hydrostatic pressure into `p` from `solver.storage` and
undo the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

along any dimensions in which a GPU fast cosine transform was performed.
"""
function copy_pressure!(p, ϕ, solver_type, arch, grid)
    @loop_xyz i j k grid begin
        i′, j′, k′ = unpermute_index(solver_type, arch, i, j, k, grid.Nx, grid.Ny, grid.Nz)
        @inbounds p[i′, j′, k′] = real(ϕ[i, j, k])
    end
    return nothing
end
