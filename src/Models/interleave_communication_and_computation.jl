using Oceananigans: prognostic_fields
using Oceananigans.Grids
using Oceananigans.Utils: KernelParameters
using Oceananigans.Grids: halo_size, topology, architecture
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedGrid
using Oceananigans.DistributedComputations: synchronize_communication!, SynchronizedDistributed

function complete_communication_and_compute_buffer!(model, ::DistributedGrid, arch)

    # Iterate over the fields to clear _ALL_ possible architectures
    for field in prognostic_fields(model)
        synchronize_communication!(field)
    end

    # Recompute tendencies near the buffer halos
    compute_buffer_tendencies!(model)

    return nothing
end

# Fallback
complete_communication_and_compute_buffer!(model, ::DistributedGrid, ::SynchronizedDistributed) = nothing
complete_communication_and_compute_buffer!(model, grid, arch) = nothing

compute_buffer_tendencies!(model) = nothing

""" Kernel parameters for computing interior tendencies. """
interior_tendency_kernel_parameters(arch, grid) = :xyz # fallback
interior_tendency_kernel_parameters(::SynchronizedDistributed, grid) = :xyz

function interior_tendency_kernel_parameters(arch::Distributed, grid)
    Rx, Ry, _ = arch.ranks
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)
    Nx, Ny, Nz = size(grid)

    # Kernel parameters to compute the tendencies in all the interior if the direction is local (`R == 1`) and only in 
    # the part of the domain that does not depend on the halo cells if the direction is partitioned. 
    local_x = Rx == 1
    local_y = Ry == 1
    one_sided_x = Tx == RightConnected || Tx == LeftConnected
    one_sided_y = Ty == RightConnected || Ty == LeftConnected 

    # Sizes
    Sx = if local_x
        Nx
    elseif one_sided_x
        Nx - Hx
    else # two sided
        Nx - 2Hx
    end

    Sy = if local_y
        Ny
    elseif one_sided_y
        Ny - Hy
    else # two sided
        Ny - 2Hy
    end

    # Offsets
    Ox = Rx == 1 || Tx == RightConnected ? 0 : Hx
    Oy = Ry == 1 || Ty == RightConnected ? 0 : Hy

    sizes = (Sx, Sy, Nz)
    offsets = (Ox, Oy, 0)
     
    return KernelParameters(sizes, offsets)
end

