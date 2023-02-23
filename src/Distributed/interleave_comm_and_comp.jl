using Oceananigans: prognostic_fields

function complete_communication_and_compute_boundary(model, grid::DistributedGrid)

    arch = architecture(grid)

    MPI.Waitall(arch.mpi_requests)
    empty!(arch.mpi_requests)
    arch.mpi_tag[1] = 0

    for field in merge(model.velocities, model.tracers)
        recv_from_buffers!(field.data, field.boundary_buffers, grid)
    end
    
    # HERE we have to put fill_eventual_halo_corners
    recompute_boundary_tendencies(model)

    return nothing
end

recompute_boundary_tendencies() = nothing
