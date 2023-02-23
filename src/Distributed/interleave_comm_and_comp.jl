using Oceananigans: prognostic_fields

function complete_communication_and_compute_boundary(model, grid::DistributedGrid, arch)

    arch = architecture(grid)

    MPI.Waitall(arch.mpi_requests)
    empty!(arch.mpi_requests)
    arch.mpi_tag[1] = 0

    for field in prognostic_fields(model)
        recv_from_buffers!(field.data, field.boundary_buffers, field.grid)
    end

    # HERE we have to put fill_eventual_halo_corners
    recompute_boundary_tendencies(model)

    return nothing
end

complete_communication_and_compute_boundary(model, grid::DistributedGrid, arch::SynchedDistributedArch) = nothing
recompute_boundary_tendencies() = nothing
