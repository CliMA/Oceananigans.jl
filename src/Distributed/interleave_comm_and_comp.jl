using Oceananigans: prognostic_fields

function complete_communication_and_compute_boundary(model, grid::DistributedGrid, arch)

    # We iterate over the fields because we have to clear _ALL_ architectures
    # and split explicit variables live on a different grid
    for field in prognostic_fields(model)
        complete_halo_communication!(field)
    end

    # HERE we have to put fill_eventual_halo_corners
    recompute_boundary_tendencies!(model)

    return nothing
end

complete_communication_and_compute_boundary(model, grid::DistributedGrid, arch::SynchedDistributedArch) = nothing
recompute_boundary_tendencies!() = nothing

function complete_halo_communication!(field)
    arch = architecture(field.grid)

    # Wait for outstanding requests
    if !isempty(arch.mpi_requests) 
        MPI.Waitall(arch.mpi_requests)

        # Reset MPI tag
        arch.mpi_tag[1] -= arch.mpi_tag[1]
    
        # Reset MPI requests
        empty!(arch.mpi_requests)
        recv_from_buffers!(field.data, field.boundary_buffers, field.grid)
    end
    
    return nothing
end