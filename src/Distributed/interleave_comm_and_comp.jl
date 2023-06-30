using Oceananigans: prognostic_fields
using Oceananigans.Grids: halo_size

function complete_communication_and_compute_boundary!(model, ::DistributedGrid, arch)

    # We iterate over the fields because we have to clear _ALL_ architectures
    # and split explicit variables live on a different grid
    for field in prognostic_fields(model)
        complete_halo_communication!(field)
    end

    # Recompute tendencies near the boundary halos
    compute_boundary_tendencies!(model)

    return nothing
end

# Fallback
complete_communication_and_compute_boundary!(model, ::DistributedGrid, ::BlockingDistributedArch) = nothing
complete_communication_and_compute_boundary!(model, grid, arch) = nothing

compute_boundary_tendencies!(model) = nothing

interior_tendency_kernel_parameters(grid) = :xyz

interior_tendency_kernel_parameters(grid::DistributedGrid) = 
            interior_tendency_kernel_parameters(grid, architecture(grid))

interior_tendency_kernel_parameters(grid, ::BlockingDistributedArch) = :xyz

function interior_tendency_kernel_parameters(grid, arch)
    Rx, Ry, _ = arch.ranks
    Hx, Hy, _ = halo_size(grid)

    Nx, Ny, Nz = size(grid)
    
    Sx = Rx == 1 ? Nx : Nx - 2Hx
    Sy = Ry == 1 ? Ny : Ny - 2Hy

    Ox = Rx == 1 ? 0 : Hx
    Oy = Ry == 1 ? 0 : Hy
     
    return KernelParameters((Sx, Sy, Nz), (Ox, Oy, 0))
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
        arch.mpi_tag[] -= arch.mpi_tag[]
    
        # Reset MPI requests
        empty!(arch.mpi_requests)
    end
    
    recv_from_buffers!(field.data, field.boundary_buffers, field.grid)
    
    return nothing
end
