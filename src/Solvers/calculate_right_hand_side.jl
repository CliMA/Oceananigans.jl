"""
Calculate the right-hand-side of the Poisson equation for the non-hydrostatic
pressure and in the process apply the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

in the y- and z-directions which is required by the GPU fast cosine transform algorithm for
reentrant channel model configurations.
"""
function calculate_poisson_right_hand_side!(RHS, solver, grid, U, G, Δt)
    @loop_xyz i j k grid begin
        i′, j′, k′ = permute_index(solver, i, j, k, grid.Nx, grid.Ny, grid.Nz)

        @inbounds RHS[i′, j′, k′] = divᶜᶜᶜ(i, j, k, grid, U.u, U.v, U.w) / Δt +
                                    divᶜᶜᶜ(i, j, k, grid, G.u, G.v, G.w)
    end
    return nothing
end
