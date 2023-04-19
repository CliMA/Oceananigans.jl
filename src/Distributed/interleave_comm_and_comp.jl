using Oceananigans: prognostic_fields
using Oceananigans.Grids: halo_size

function complete_communication_and_compute_boundary(model, grid::DistributedGrid, arch)

    # We iterate over the fields because we have to clear _ALL_ architectures
    # and split explicit variables live on a different grid
    for field in prognostic_fields(model)
        complete_halo_communication!(field)
    end

    # HERE we have to put fill_eventual_halo_corners
    compute_boundary_tendencies!(model)

    return nothing
end

complete_communication_and_compute_boundary(model, grid::DistributedGrid, arch::BlockingDistributedArch) = nothing
compute_boundary_tendencies!(model) = nothing

interior_tendency_kernel_size(grid::DistributedGrid)    = interior_tendency_kernel_size(grid,    architecture(grid))
interior_tendency_kernel_offsets(grid::DistributedGrid) = interior_tendency_kernel_offsets(grid, architecture(grid))

interior_tendency_kernel_size(grid, ::BlockingDistributedArch) = :xyz
interior_tendency_kernel_offsets(grid, ::BlockingDistributedArch) = (0, 0, 0)

function interior_tendency_kernel_size(grid, arch)
    Rx, Ry, _ = arch.ranks
    Hx, Hy, _ = halo_size(grid)

    Nx, Ny, Nz = size(grid)
    
    Ax = Rx == 1 ? 0 : Hx
    Ay = Ry == 1 ? 0 : Hy

    return (Nx-2Ax, Ny-2Ay, Nz)
end

function interior_tendency_kernel_offsets(grid, arch)
    Rx, Ry, _ = arch.ranks
    Hx, Hy, _ = halo_size(grid)
    
    Ax = Rx == 1 ? 0 : Hx
    Ay = Ry == 1 ? 0 : Hy

    return (Ax, Ay, 0)
end

"""
    complete_halo_communication!(field)

complete the halo passing of `field` among processors.
"""
function complete_halo_communication!(field)
    arch = architecture(field.grid)

    # Wait for outstanding requests
    if !isempty(arch.mpi_requests) 
        cooperative_waitall!(arch.mpi_requests)

        # Reset MPI tag
        arch.mpi_tag[1] -= arch.mpi_tag[1]
    
        # Reset MPI requests
        empty!(arch.mpi_requests)
    end
    
    recv_from_buffers!(field.data, field.boundary_buffers, field.grid)
    
    return nothing
end