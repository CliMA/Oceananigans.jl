function complete_communication_and_compute_boundary(model, grid::DistributedGrid)

    arch = architecture(grid)

    MPI.Waitall(arch.mpi_requests)
    empty!(arch.mpi_requests)
    arch.mpi_tag[1] = 0

    for side in (:west_and_east, :south_and_north, :bottom_and_top)
        for field in prognostic_fields(model)
            recv_from_buffers!(field.data, field.boundary_buffers, grid, Val(side))    
        end
    end

    # HERE we have to put fill_eventual_halo_corners
    recompute_boundary_tendencies(model)

    return nothing
end

recompute_boundary_tendencies() = nothing
