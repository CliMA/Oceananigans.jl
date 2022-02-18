import Oceananigans.Grids: architecture, size
using Oceananigans.Grids: metrics_precomputed, on_architecture

struct MultiRegionGrid{FT, TX, TY, TZ, P, G, D, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch}
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
    size   = size(global_grid)
    size   = partition_size(partition, size)
    extent = partition_extent(partition, global_grid)
    
    arch    = infer_architecture(devices)
    devices = validate_devices(partition, devices)
    devices = assign_devices(partition, devices)

    FT         = eltype(global_grid)
    TX, TY, TZ = topology(global_grid)
    
    local_grids = fill_multiregion_grids(arch, devices, global_grid, (TX, TY, TZ), size, extent)

    return MultiRegionGrid{FT, TX, TY, TZ}(arch, partition, local_grids, devices)
end

@inline assoc_device(mrg::MultiRegionGrid, idx) = mrg.devices[idx]
@inline architecture(mrg::MultiRegionGrid) = mrg.architecture

@inline underlying_arch(::MultiGPU) = GPU()
@inline underlying_arch(::GPU) = GPU()
@inline underlying_arch(::CPU) = CPU()

function fill_multiregion_grids(arch, devices, grid, topo, N, extent)
    local_grids = []
    child_arch = underlying_arch(arch)
    for part in 1:length(devices)
        switch_device!(devices[part])
        push!(local_grids, construct_grid(grid, child_arch, topo, extent[part], N[part]))
    end
    return local_grids
end

function construct_grid(grid::RectilinearGrid, child_arch, topo, extent, size)
    halo = halo_size(grid)
    FT   = eltype(grid)
    return RectilinearGrid(child_arch, FT; size = size, halo = halo, topology = topo, extent...)
end

function construct_grid(grid::LatitudeLongitudeGrid, child_arch, topo, extent, size)
    halo = halo_size(grid)
    FT   = eltype(grid)
    lon, lat, z = extent
    return LatitudeLongitudeGrid(child_arch, FT; 
                                 size = size, halo = halo,
                                 latitude = lat, longitude = lon, z = z,
                                 precompute_metrics = metrics_precomputed(grid))
end

function construct_grid(ibg::ImmersedBoundaryGrid, child_arch, topo, extent, size)
    boundary = ibg.immersed_boundary
    return ImmersedBoundaryGrid(construct_grid(ibg.grid, child_arch, topo, extent, size), boundary)
end

Base.show(io::IO, mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =  
    print(io, "MultiRegionalGrid: \n",
              "├── grids: $(summary(mrg.local_grids[1])) \n",
              "├── partitioning: $(summary(mrg.partition)) \n",
              "└── architecture: $(arch_summary(mrg))")

arch_summary(mrg::MultiRegionGrid) = "$(architecture(mrg)) $(architecture(mrg) isa MultiGPU ? "on $(length(unique(mrg.devices))) devices" : "")"
