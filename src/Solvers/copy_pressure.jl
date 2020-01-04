"""
Copy the non-hydrostatic pressure into `p` from `solver.storage` and
undo the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

along any dimensions in which a GPU fast cosine transform was performed.
"""
function copy_pressure!(p, solver, grid)
    ϕ = solver.storage
    @loop_xyz i j k grid begin
        i′, j′, k′ = unpermute_index(solver, i, j, k, grid.Nx, grid.Ny, grid.Nz)
        @inbounds p[i′, j′, k′] = real(ϕ[i, j, k])
    end
    return nothing
end
