using Oceananigans.Grids: metrics_precomputed, on_architecture, pop_flat_elements
import Oceananigans.Grids: architecture, size, new_data

struct MultiRegionGrid{FT, TX, TY, TZ, P, G, D, Arch} <: AbstractMultiGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    partition :: P
    local_grids :: G
    devices :: D

    function MultiRegionGrid{FT, TX, TY, TZ}(arch::A, partition::P, local_grids::G, devices::D) where {FT, TX, TY, TZ, P, G, D, A}
        return new{FT, TX, TY, TZ, P, G, D, A}(arch, partition, local_grids, devices)
    end
end

function MultiRegionGrid(global_grid; partition = XPartition(1), devices = nothing)

    global_grid = on_architecture(CPU(), global_grid)
    N      = size(global_grid)
    N      = partition_size(partition, N)
    extent = partition_extent(partition, global_grid)
    
    arch    = infer_architecture(devices)
    devices = validate_devices(partition, devices)
    devices = assign_devices(partition, devices)

    FT         = eltype(global_grid)
    child_arch = underlying_arch(arch)
    TX, TY, TZ = T = topology(global_grid)
    
    args = (global_grid, child_arch, T, N, extent)
    iter = (0, 0, 0, 1, 1)

    local_grids = multi_region_object(devices, construct_grid, args, iter)

    return MultiRegionGrid{FT, TX, TY, TZ}(arch, partition, local_grids, devices)
end

@inline assoc_device(mrg::MultiRegionGrid, idx) = mrg.devices[idx]
@inline assoc_grid(mrg::MultiRegionGrid, idx)   = mrg.local_grids[idx]
@inline architecture(mrg::MultiRegionGrid)      = mrg.architecture
@inline grids(mrg::MultiRegionGrid)             = mrg.local_grids

function construct_grid(grid::RectilinearGrid, child_arch, topo, size, extent)
    halo = halo_size(grid)
    size = pop_flat_elements(size, topo)
    halo = pop_flat_elements(halo, topo)
    FT   = eltype(grid)
    return RectilinearGrid(child_arch, FT; size = size, halo = halo, topology = topo, extent...)
end

function construct_grid(grid::LatitudeLongitudeGrid, child_arch, topo, size, extent)
    halo = halo_size(grid)
    FT   = eltype(grid)
    lon, lat, z = extent
    return LatitudeLongitudeGrid(child_arch, FT; 
                                 size = size, halo = halo,
                                 latitude = lat, longitude = lon, z = z,
                                 precompute_metrics = metrics_precomputed(grid))
end

function construct_grid(ibg::ImmersedBoundaryGrid, child_arch, topo, size, extent)
    boundary = ibg.immersed_boundary
    return ImmersedBoundaryGrid(construct_grid(ibg.grid, child_arch, topo, extent, size), boundary)
end

function Adapt.adapt_structure(to, mrg::MultiRegionGrid)
    TX, TY, TZ = topology(mrg)
    return MultiRegionGrid{TX, TY, TZ}(nothing,
                                       Adapt.adapt(to, partition),
                                       Adapt.adapt(to, local_grids),
                                       Adapt.adapt(to, devices))
end

Base.show(io::IO, mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =  
    print(io, "MultiRegionGrid partitioned on $(underlying_arch(architecture(mrg))): \n",
              "├── grids: $(summary(mrg.local_grids[1])) \n",
              "├── partitioning: $(summary(mrg.partition)) \n",
              "└── architecture: $(arch_summary(mrg))")

arch_summary(mrg::MultiRegionGrid) = "$(architecture(mrg)) $(architecture(mrg) isa MultiGPU ? "on $(length(unique(mrg.devices))) devices" : "")"
