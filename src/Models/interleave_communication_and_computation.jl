using Oceananigans: prognostic_fields
using Oceananigans.Grids
using Oceananigans.Utils: KernelParameters
using Oceananigans.Grids: halo_size, topology, architecture
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedGrid
using Oceananigans.DistributedComputations: synchronize_communication!, AsynchronousDistributed

function complete_communication_and_compute_buffer!(model, ::DistributedGrid, ::AsynchronousDistributed)

    # Iterate over the fields to clear _ALL_ possible architectures
    for field in prognostic_fields(model)
        synchronize_communication!(field)
    end

    # Recompute tendencies near the buffer halos
    compute_buffer_tendencies!(model)

    return nothing
end

# Fallback
complete_communication_and_compute_buffer!(model, grid, arch) = nothing
compute_buffer_tendencies!(model) = nothing

""" Kernel parameters for computing interior tendencies. """
interior_tendency_kernel_parameters(arch, grid) = :xyz # fallback

function interior_tendency_kernel_parameters(arch::AsynchronousDistributed, grid)
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

"""
    surface_kernel_parameters(grid)

Return kernel parameters for computing 2D (surface) variables including halo regions.

The returned `KernelParameters` cover the total date ming one halo cell on each side
(indices `-Hx+2:Nx+Hx-1` and `-Hy+2:Ny+Hy-1`), which is sufficient for computing
quantities that require neighbor data (like derivatives and interpolations).
"""
@inline function surface_kernel_parameters(grid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)

    ii = ifelse(Tx == Flat, 1:Nx, -Hx+2:Nx+Hx-1)
    jj = ifelse(Ty == Flat, 1:Ny, -Hy+2:Ny+Hy-1)

    return KernelParameters(ii, jj)
end

"""
    volume_kernel_parameters(grid)

Return kernel parameters for computing 3D (volume) variables including halo regions.
Similar to `surface_kernel_parameters` but for three-dimensional fields.
"""
@inline function volume_kernel_parameters(grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    Tx, Ty, Tz = topology(grid)

    ii = ifelse(Tx == Flat, 1:Nx, -Hx+2:Nx+Hx-1)
    jj = ifelse(Ty == Flat, 1:Ny, -Hy+2:Ny+Hy-1)
    kk = ifelse(Tz == Flat, 1:Nz, -Hz+2:Nz+Hz-1)

    return KernelParameters(ii, jj, kk)
end
