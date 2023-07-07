using Oceananigans.Grids: metrics_precomputed, on_architecture, pop_flat_elements
import Oceananigans.Grids: architecture, size, new_data, halo_size
import Oceananigans.Grids: with_halo, on_architecture
import Oceananigans.Distributed: reconstruct_global_grid

struct MultiRegionGrid{FT, TX, TY, TZ, P, G, D, Arch} <: AbstractMultiRegionGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    partition :: P
    region_grids :: G
    devices :: D

    MultiRegionGrid{FT, TX, TY, TZ}(arch::A, partition::P,
                                    region_grids::G,
                                    devices::D) where {FT, TX, TY, TZ, P, G, D, A} =
        new{FT, TX, TY, TZ, P, G, D, A}(arch, partition, region_grids, devices)
end

@inline isregional(mrg::MultiRegionGrid)        = true
@inline getdevice(mrg::MultiRegionGrid, i)      = getdevice(mrg.region_grids, i)
@inline switch_device!(mrg::MultiRegionGrid, i) = switch_device!(getdevice(mrg, i))
@inline devices(mrg::MultiRegionGrid)           = devices(mrg.region_grids)
@inline sync_all_devices!(mrg::MultiRegionGrid) = sync_all_devices!(devices(mrg))

@inline  getregion(mrg::MultiRegionGrid, r) = _getregion(mrg.region_grids, r)
@inline _getregion(mrg::MultiRegionGrid, r) =  getregion(mrg.region_grids, r)

@inline Base.length(mrg::MultiRegionGrid)   = Base.length(mrg.region_grids)

const ImmersedMultiRegionGrid = MultiRegionGrid{FT, TX, TY, TZ, P, <:MultiRegionObject{<:Tuple{Vararg{IBG}}}} where {FT, TX, TY, TZ, P, IBG<:ImmersedBoundaryGrid}

"""
    MultiRegionGrid(global_grid; partition = XPartition(2), devices = nothing)

Split a `global_grid` into different regions handled by `devices`.

Positional Arguments
====================

- `global_grid`: the grid to be divided into regions

Keyword Arguments
=================

- `partition`: the partitioning required. The implemented partitioning are `XPartition` 
               (division along the x direction) and `YPartition` (division along the y direction)
- `devices`: the devices to allocate memory on. `nothing` will allocate memory on the `CPU`. For 
             `GPU` computation it is possible to specify the total number of `GPU`s or the specific
             `GPU`s to allocate memory on. The number of devices does not have to match the number of
             regions 
"""
function MultiRegionGrid(global_grid; partition = XPartition(2), devices = nothing, validate = true)

    if length(partition) == 1
        return global_grid
    end

    @warn "MultiRegion functionalities are experimental: help the development by reporting bugs or non-implemented features!"

    arch = architecture(global_grid)
    
    if validate
        devices = validate_devices(partition, arch, devices)
        devices = assign_devices(partition, devices)
    end

    global_grid  = on_architecture(CPU(), global_grid)
    local_size   = MultiRegionObject(partition_size(partition, global_grid), devices)
    local_extent = MultiRegionObject(partition_extent(partition, global_grid), devices)
    local_topo   = MultiRegionObject(partition_topology(partition, global_grid), devices)  
    
    global_topo  = topology(global_grid)

    FT   = eltype(global_grid)
    
    args = (Reference(global_grid), 
            Reference(arch), 
            local_topo, 
            local_size,
            local_extent, 
            Reference(partition), 
            Iterate(1:length(partition)))

    region_grids = construct_regionally(construct_grid, args...)
    
    ## If we are on GPUs we want to enable peer access, which we do by just copying fake arrays between all devices
    maybe_enable_peer_access!(devices)

    return MultiRegionGrid{FT, global_topo[1], global_topo[2], global_topo[3]}(arch, partition, region_grids, devices)
end

function construct_grid(grid::RectilinearGrid, child_arch, topo, size, extent, args...)
    halo = halo_size(grid)
    size = pop_flat_elements(size, topo)
    halo = pop_flat_elements(halo, topo)
    FT   = eltype(grid)

    return RectilinearGrid(child_arch, FT; size = size, halo = halo, topology = topo, extent...)
end

function construct_grid(grid::LatitudeLongitudeGrid, child_arch, topo, size, extent, args...)
    halo = halo_size(grid)
    FT   = eltype(grid)
    lon, lat, z = extent
    return LatitudeLongitudeGrid(child_arch, FT; 
                                 size = size, halo = halo, radius = grid.radius,
                                 latitude = lat, longitude = lon, z = z, topology = topo,
                                 precompute_metrics = metrics_precomputed(grid))
end

function construct_grid(ibg::ImmersedBoundaryGrid, child_arch, topo, local_size, extent, partition, region)
    boundary = partition_immersed_boundary(ibg.immersed_boundary, partition, local_size, region, child_arch)
    return ImmersedBoundaryGrid(construct_grid(ibg.underlying_grid, child_arch, topo, local_size, extent), boundary)
end

partition_immersed_boundary(b, args...) = 
    getnamewrapper(b)(partition_global_array(getproperty(b, propertynames(b)[1]), args...))

function reconstruct_global_grid(mrg)
    size    = reconstruct_size(mrg, mrg.partition)
    extent  = reconstruct_extent(mrg, mrg.partition)
    topo    = topology(mrg)
    switch_device!(mrg.devices[1])
    return construct_grid(mrg.region_grids[1], architecture(mrg), topo, size, extent)
end

"""
    reconstruct_global_grid(mrg::MultiRegionGrid)

Reconstruct the `mrg` global grid associated with the `MultiRegionGrid` on `architecture(mrg)`.
"""
function reconstruct_global_grid(mrg::ImmersedMultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    underlying_mrg = MultiRegionGrid{FT, TX, TY, TZ}(architecture(mrg), 
                                                     mrg.partition, 
                                                     construct_regionally(getproperty, mrg, :underlying_grid), 
                                                     mrg.devices)
                                                     
    global_grid     = on_architecture(CPU(), reconstruct_global_grid(underlying_mrg))
    cpu_mrg         = on_architecture(CPU(), mrg)
    local_boundary  = construct_regionally(getproperty, cpu_mrg, :immersed_boundary)
    local_array     = construct_regionally(getproperty, local_boundary, propertynames(local_boundary[1])[1])
    local_array     = construct_regionally(getinterior, local_array, mrg)
    global_boundary = getnamewrapper(local_boundary[1])(reconstruct_global_array(local_array, mrg.partition, architecture(mrg)))
    return on_architecture(architecture(mrg), ImmersedBoundaryGrid(global_grid, global_boundary))
end

getinterior(array::AbstractArray{T, 2}, grid) where T = array[1:grid.Nx, 1:grid.Ny]
getinterior(array::AbstractArray{T, 3}, grid) where T = array[1:grid.Nx, 1:grid.Ny, 1:grid.Nz]
getinterior(func::Function, grid) = func

"""
    multi_region_object_from_array(a::AbstractArray, grid)

Adapt an array `a` to be compatible with a `MultiRegion` grid.
"""
function multi_region_object_from_array(a::AbstractArray, mrg::MultiRegionGrid)
    local_size = construct_regionally(size, mrg)
    arch = architecture(mrg)
    a    = arch_array(CPU(), a)
    ma   = construct_regionally(partition_global_array, a, mrg.partition, local_size, Iterate(1:length(mrg)), arch)
    return ma
end

# Fallback!
multi_region_object_from_array(a::AbstractArray, grid) = arch_array(architecture(grid), a)

#### 
#### Utilitites for MultiRegionGrid
####

new_data(FT::DataType, mrg::MultiRegionGrid, args...) = construct_regionally(new_data, FT, mrg, args...)

# This is kind of annoying but it is necessary to have compatible MultiRegion and Distributed
function with_halo(new_halo, mrg::MultiRegionGrid) 
    devices   = mrg.devices
    partition = mrg.partition
    cpu_mrg   = on_architecture(CPU(), mrg)

    global_grid = reconstruct_global_grid(cpu_mrg)
    new_global  = with_halo(new_halo, global_grid)
    new_global  = on_architecture(architecture(mrg), new_global)

    return MultiRegionGrid(new_global; partition, devices, validate = false)
end

function on_architecture(::CPU, mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    new_grids = construct_regionally(on_architecture, CPU(), mrg)
    devices   = Tuple(CPU() for i in 1:length(mrg))  
    return MultiRegionGrid{FT, TX, TY, TZ}(CPU(), mrg.partition, new_grids, devices)
end

function on_specific_architecture(arch, dev, grid)
    switch_device!(dev)
    return on_architecture(arch, grid)
end

Base.summary(mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =  
    "MultiRegionGrid{$FT, $TX, $TY, $TZ} with $(summary(mrg.partition)) on $(string(typeof(mrg.region_grids[1]).name.wrapper))"

Base.show(io::IO, mrg::MultiRegionGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =  
    print(io, "MultiRegionGrid{$FT, $TX, $TY, $TZ} partitioned on $(architecture(mrg)): \n",
              "├── grids: $(summary(mrg.region_grids[1])) \n",
              "├── partitioning: $(summary(mrg.partition)) \n",
              "└── devices: $(devices(mrg))")
 
function Base.:(==)(mrg1::MultiRegionGrid, mrg2::MultiRegionGrid)
    #check if grids are of the same type
    vals = construct_regionally(Base.:(==), mrg1, mrg2)
    return all(vals.regional_objects)
end
   
####
#### Get property for `MultiRegionGrid` (gets the properties of region 1)
#### In general getpropert should never be used as a MultiRegionGrid
#### Should be used only in combination with an @apply_regionally
####

grids(mrg::MultiRegionGrid) = mrg.region_grids

getmultiproperty(mrg::MultiRegionGrid, x::Symbol) = construct_regionally(Base.getproperty, grids(mrg), x)

const MRG = MultiRegionGrid

@inline Base.getproperty(mrg::MRG, property::Symbol)                 = get_multi_property(mrg, Val(property))
@inline get_multi_property(mrg::MRG, ::Val{property}) where property = getproperty(getindex(getfield(mrg, :region_grids), 1), property)
@inline get_multi_property(mrg::MRG, ::Val{:architecture})           = getfield(mrg, :architecture)
@inline get_multi_property(mrg::MRG, ::Val{:partition})              = getfield(mrg, :partition)
@inline get_multi_property(mrg::MRG, ::Val{:region_grids})           = getfield(mrg, :region_grids)
@inline get_multi_property(mrg::MRG, ::Val{:devices})                = getfield(mrg, :devices)

@inline function get_multi_property(mrg::ImmersedMultiRegionGrid, ::Val{:underlying_grid})
    global_grid = reconstruct_global_grid(mrg)
    grid        = global_grid.underlying_grid
    return MultiRegionGrid(grid, partition = mrg.partition, devices = mrg.devices, validate = false)
end

function maybe_add_active_cells_map(mrg::MRG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    new_grids = construct_regionally(maybe_add_active_cells_map, mrg)
    return MultiRegionGrid{FT, TX, TY, TZ}(mrg.architecture, mrg.partition, new_grids, mrg.devices)
end
