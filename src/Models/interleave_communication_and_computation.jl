using Oceananigans: prognostic_fields
using Oceananigans.Grids
using Oceananigans.Utils: KernelParameters, worksize, get_active_cells_map
using Oceananigans.Grids: halo_size, topology, architecture, LeftConnectedOnlyTopology
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedGrid
using Oceananigans.DistributedComputations: synchronize_communication!, AsynchronousDistributed

# True for topologies whose buffer region is owned locally (no MPI-communicating side in that direction).
@inline is_local_dimension(T) = T === Bounded           ||
                                T === Periodic          ||
                                T === Flat              ||
                                T === RightFaceFolded   ||
                                T === RightCenterFolded

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
    one_sided_x = Tx == RightConnected || Tx == LeftConnected
    one_sided_y = Ty == RightConnected || Ty <: LeftConnectedOnlyTopology

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
    Ox = Rx == 1 || Tx == RightConnected ? 0 : Hx
    Oy = Ry == 1 || Ty == RightConnected ? 0 : Hy

    sizes = (Sx, Sy, Wz)
    offsets = (Ox, Oy, 0)

    return something(get_active_cells_map(grid, Val(:core)), KernelParameters(sizes, offsets))
end

"""
    buffer_tendency_kernel_parameters(grid, arch)

Return a NamedTuple keyed by region name (`:west_halo_dependent_cells`, `:east_halo_dependent_cells`,
`:south_halo_dependent_cells`, `:north_halo_dependent_cells`) of `KernelParameters` for tendency compute
in each halo-dependent buffer strip. Regions where the local subdomain has no neighbor are set to `nothing`.
"""
@inline function buffer_tendency_kernel_parameters(grid, arch)
    Nx, Ny, Nz = size(grid)
    Wx, Wy, _  = worksize(grid)
    Hx, Hy, _  = halo_size(grid)

    param_west  = (1:Hx,       1:Wy,       1:Nz)
    param_east  = (Wx-Hx+1:Wx, 1:Wy,       1:Nz)
    param_south = (1:Wx,       1:Hy,       1:Nz)
    param_north = (1:Wx,       Wy-Hy+1:Wy, 1:Nz)

    params = (param_west, param_east, param_south, param_north)
    return buffer_parameters(params, grid, arch)
end

"""
    buffer_parameters(parameters, grid, arch)

Wrap a 4-tuple `(west, east, south, north)` of index ranges into a per-region NamedTuple of `KernelParameters`,
with `nothing` for regions where the local subdomain has no neighbor on that side.
"""
@inline function buffer_parameters(parameters, grid, arch)
    TX, TY, _ = topology(grid)

    include_west  = !is_local_dimension(TX) && !(TX === RightConnected)
    include_east  = !is_local_dimension(TX) && !(TX === LeftConnected)
    include_south = !is_local_dimension(TY) && !(TY === RightConnected)
    include_north = !is_local_dimension(TY) && !(TY <: LeftConnectedOnlyTopology)

    p_west, p_east, p_south, p_north = parameters

    return (west_halo_dependent_cells  = include_west  ? KernelParameters(p_west...)  : nothing,
            east_halo_dependent_cells  = include_east  ? KernelParameters(p_east...)  : nothing,
            south_halo_dependent_cells = include_south ? KernelParameters(p_south...) : nothing,
            north_halo_dependent_cells = include_north ? KernelParameters(p_north...) : nothing)
end

"""
    distributed_region_kernel_parameters(grid)

Return a NamedTuple of workspecs keyed by region — `halo_independent_cells` plus
the four `*_halo_dependent_cells` strips (with `nothing` for absent regions). Composes
`interior_tendency_kernel_parameters` for the core and `buffer_tendency_kernel_parameters`
for the halo-dependent strips.
"""
function distributed_region_kernel_parameters(grid)
    arch = architecture(grid)
    return (halo_independent_cells = interior_tendency_kernel_parameters(arch, grid),
            buffer_tendency_kernel_parameters(grid, arch)...)
end
