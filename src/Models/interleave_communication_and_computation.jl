using Oceananigans: prognostic_fields
using Oceananigans.Grids
using Oceananigans.Utils: KernelParameters
using Oceananigans.Grids: halo_size, topology, architecture
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedGrid
using Oceananigans.DistributedComputations: synchronize_communication!, SynchronizedDistributed

function complete_communication_and_compute_boundary!(model, ::DistributedGrid, arch)

    # Iterate over the fields to clear _ALL_ possible architectures
    for field in prognostic_fields(model)
        synchronize_communication!(field)
    end

    # Recompute tendencies near the boundary halos
    compute_boundary_tendencies!(model)

    return nothing
end

# Fallback
complete_communication_and_compute_boundary!(model, ::DistributedGrid, ::SynchronizedDistributed) = nothing
complete_communication_and_compute_boundary!(model, grid, arch) = nothing

compute_boundary_tendencies!(model) = nothing

""" Kernel parameters for computing interior tendencies. """
interior_tendency_kernel_parameters(grid) = :xyz # fallback

interior_tendency_kernel_parameters(grid::DistributedGrid) = 
    interior_tendency_kernel_parameters(grid, architecture(grid))

interior_tendency_kernel_parameters(grid, ::SynchronizedDistributed) = :xyz

function interior_tendency_kernel_parameters(grid, arch)
    Rx, Ry, _ = arch.ranks
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)
    Nx, Ny, Nz = size(grid)

    # TODO: add a comment explaining what is going on here
    # Kernel parameters to compute the tendencies in all the interior if the direction is local (`R == 1`) and only in 
    # the part of the domain that does not depend on the halo cells if the direction is partitioned. 
    local_x = Rx == 1
    local_y = Ry == 1
    one_sided_connection_x = Tx == RightConnected || Tx == LeftConnected
    one_sided_connection_y = Ty == RightConnected || Ty == LeftConnected 
    Sx = local_x ? Nx : ( one_sided_connection_x ? Nx - Hx : Nx - 2Hx)
    Sy = local_y ? Ny : ( one_sided_connection_y ? Ny - Hy : Ny - 2Hy)

    Ox = Rx == 1 || Tx == RightConnected ? 0 : Hx
    Oy = Ry == 1 || Ty == RightConnected ? 0 : Hy
     
    return KernelParameters((Sx, Sy, Nz), (Ox, Oy, 0))
end

