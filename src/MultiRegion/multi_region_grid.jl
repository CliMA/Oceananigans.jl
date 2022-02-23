using Oceananigans.Grids: metrics_precomputed, on_architecture, pop_flat_elements
import Oceananigans.Grids: architecture, size, new_data, halo_size

struct MultiRegionGrid{FT, TX, TY, TZ, P, G, D, Arch} <: AbstractMultiGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    partition :: P
    region_grids :: G
    devices :: D

    function MultiRegionGrid{FT, TX, TY, TZ}(arch::A, partition::P, region_grids::G, devices::D) where {FT, TX, TY, TZ, P, G, D, A}
        return new{FT, TX, TY, TZ, P, G, D, A}(arch, partition, region_grids, devices)
    end
end

function MultiRegionGrid(global_grid; partition = XPartition(1), devices = nothing)

    arch    = devices isa Nothing ? CPU() : GPU()
    devices = validate_devices(partition, devices)
    devices = assign_devices(partition, devices)

    global_grid  = on_architecture(CPU(), global_grid)
    local_size   = MultiRegionObject(partition_size(partition, global_grid), devices)
    local_extent = MultiRegionObject(partition_extent(partition, global_grid), devices)
    
    FT   = eltype(global_grid)
    topo = topology(global_grid)  # Here we should make also topo a MultiRegionObject?
    
    args = (Reference(global_grid), Reference(arch), Reference(topo), local_size, local_extent)

    region_grids = construct_regionally(construct_grid, args...)

    return MultiRegionGrid{FT, topo[1], topo[2], topo[3]}(arch, partition, region_grids, devices)
end

devices(mrg::MultiRegionGrid)      = devices(mrg.region_grids)

getregion(mrg::MultiRegionGrid, i)      = getregion(mrg.region_grids, i)
getdevice(mrg::MultiRegionGrid, i)      = getdevice(mrg.region_grids, i)
switch_device!(mrg::MultiRegionGrid, i) = switch_device!(getdevice(mrg, i))

isregional(mrg::MultiRegionGrid) = true

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
    return ImmersedBoundaryGrid(construct_grid(ibg.grid, child_arch, topo, size, extent), boundary)
end

getmultiproperty(mrg::MultiRegionGrid, x::Symbol) = apply_regionally(Base.getproperty, grids(mrg), x)

Base.show(io::IO, mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =  
    print(io, "MultiRegionGrid{$FT, $TX, $TY, $TZ} partitioned on $(architecture(mrg)): \n",
              "├── grids: $(summary(mrg.region_grids[1])) \n",
              "├── partitioning: $(summary(mrg.partition)) \n",
              "└── devices: $(devices(mrg))")

Base.summary(mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =  
    "MultiRegionGrid{$FT, $TX, $TY, $TZ} with $(summary(mrg.partition)) on $(string(typeof(mrg.region_grids[1]).name.wrapper))"
