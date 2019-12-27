function copy_pressure!(pNHS, solver, grid, U, G, Δt)
    ϕ = solver.storage
    @loop_xyz i j k grid begin
        i′, j′, k′ = unpermute_index(solver, i, j, k, grid.Nx, grid.Ny, grid.Nz)
        @inbounds pNHS[i′, j′, k′] = real(ϕ[i, j, k])
    end
    return nothing
end
