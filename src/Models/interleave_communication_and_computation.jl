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
function interior_tendency_kernel_parameters(arch, grid)
    Nx, Ny, Nz = size(grid)
    return KernelParameters(0:Nx+1, 0:Ny+1, 1:Nz)
end

function interior_tendency_kernel_parameters(::SynchronizedDistributed, grid) 
    Nx, Ny, Nz = size(grid)
    return KernelParameters(0:Nx+1, 0:Ny+1, 1:Nz)
end

function interior_parameters_without_halo(arch, grid)
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
        Nx + 2
    elseif one_sided_x
        Nx - Hx + 2
    else # two sided
        Nx - 2Hx + 2
    end

    Sy = if local_y
        Ny + 2
    elseif one_sided_y
        Ny - Hy + 2
    else # two sided
        Ny - 2Hy + 2
    end

    # Offsets
    Ox = Rx == 1 || Tx == RightConnected ? -1 : Hx-1
    Oy = Ry == 1 || Ty == RightConnected ? -1 : Hy-1

    sizes = (Sx, Sy, Nz)
    offsets = (Ox, Oy, 0)
     
    return sizes, offsets
end

function interior_tendency_kernel_parameters(arch::Distributed, grid)
    return KernelParameters(interior_parameters_without_halo(arch, grid)...)
end

@inline surface_kernel_parameters(arch, grid) = surface_kernel_parameters(grid) 
@inline surface_kernel_parameters(::SynchronizedDistributed, grid) = surface_kernel_parameters(grid) 

# extend w kernel to compute also the boundaries
# If Flat, do not calculate on halos!
@inline function surface_kernel_parameters(grid) 
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)

    ii = ifelse(Tx == Flat, 1:Nx, -Hx+2:Nx+Hx-1)
    jj = ifelse(Ty == Flat, 1:Ny, -Hy+2:Ny+Hy-1)

    return KernelParameters(ii, jj)
end

@inline function surface_kernel_parameters(arch::Distributed, grid) 
    size, offset = interior_parameters_without_halo(arch, grid)
    return KernelParameters(size[1:2], offset[1:2])
end
