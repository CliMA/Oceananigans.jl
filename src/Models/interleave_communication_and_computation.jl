using Oceananigans: prognostic_fields
using Oceananigans.Grids
using Oceananigans.Utils: KernelParameters, worksize, get_active_cells_map
using Oceananigans.Grids: halo_size, topology, architecture, CommunicatingSide, Sides
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

""" Kernel parameters for computing interior tendencies: the `:core` active cells map of `grid`, if it has one. """
@inline interior_tendency_kernel_parameters(arch, grid) = something(get_active_cells_map(grid, Val(:core)), KernelParameters(worksize(grid), map(zero, worksize(grid))))

function interior_tendency_kernel_parameters(arch::AsynchronousDistributed, grid)
    Rx, Ry, _ = arch.ranks
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)
    Wx, Wy, Wz = worksize(grid)

    # Kernel parameters to compute the tendencies in all the interior if the direction is local (`R == 1`) and only in
    # the part of the domain that does not depend on the halo cells if the direction is partitioned.
    local_x = Rx == 1
    local_y = Ry == 1
    one_sided_x = (Tx <: Sides{<:CommunicatingSide}) != (Tx <: Sides{<:Any, <:CommunicatingSide})
    one_sided_y = (Ty <: Sides{<:CommunicatingSide}) != (Ty <: Sides{<:Any, <:CommunicatingSide})

    # Sizes
    Sx = if local_x
        Wx
    elseif one_sided_x
        Wx - Hx
    else # two sided
        Wx - 2Hx
    end

    Sy = if local_y
        Wy
    elseif one_sided_y
        Wy - Hy
    else # two sided
        Wy - 2Hy
    end

    # Offsets
    Ox = Rx == 1 || !(Tx <: Sides{<:CommunicatingSide}) ? 0 : Hx
    Oy = Ry == 1 || !(Ty <: Sides{<:CommunicatingSide}) ? 0 : Hy

    sizes = (Sx, Sy, Wz)
    offsets = (Ox, Oy, 0)

    return something(get_active_cells_map(grid, Val(:core)), KernelParameters(sizes, offsets))
end
