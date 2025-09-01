using Oceananigans.Grids: metrics_precomputed, on_architecture, pop_flat_elements, grid_name
using Oceananigans.ImmersedBoundaries: GridFittedBottom, PartialCellBottom, GridFittedBoundary

import Oceananigans.Grids: architecture, size, new_data, halo_size
import Oceananigans.Grids: with_halo, on_architecture
import Oceananigans.Grids: destantiate
import Oceananigans.Grids: minimum_xspacing, minimum_yspacing, minimum_zspacing
import Oceananigans.Models.HydrostaticFreeSurfaceModels: default_free_surface
import Oceananigans.DistributedComputations: reconstruct_global_grid

struct MultiRegionGrid{FT, TX, TY, TZ, CZ, P, C, G, Arch} <: AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, Arch}
    architecture :: Arch
    partition :: P
    connectivity :: C
    region_grids :: G

    function MultiRegionGrid{FT, TX, TY, TZ, CZ}(arch::A, partition::P, connectivity::C, region_grids::G) where {FT, TX, TY, TZ, CZ, P, C, G, A}
        return new{FT, TX, TY, TZ, CZ, P, C, G, A}(arch, partition, connectivity, region_grids)
    end
end

const ImmersedMultiRegionGrid{FT, TX, TY, TZ} = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:MultiRegionGrid}

const MultiRegionGrids{FT, TX, TY, TZ} = Union{MultiRegionGrid{FT, TX, TY, TZ}, ImmersedMultiRegionGrid{FT, TX, TY, TZ}}

@inline isregional(mrg::MultiRegionGrids) = true
@inline regions(mrg::MultiRegionGrids) = 1:length(mrg.regional_grids)

@inline  getregion(mrg::MultiRegionGrid, r) = _getregion(mrg.region_grids, r)
@inline _getregion(mrg::MultiRegionGrid, r) =  getregion(mrg.region_grids, r)

# Convenience
@inline Base.getindex(mrg::MultiRegionGrids, r::Int) = getregion(mrg, r)
@inline Base.first(mrg::MultiRegionGrids) = mrg[1]
@inline Base.lastindex(mrg::MultiRegionGrids) = length(mrg)
number_of_regions(mrg::MultiRegionGrids) = lastindex(mrg)

minimum_xspacing(grid::MultiRegionGrid) =
    minimum(minimum_xspacing(grid[r]) for r in 1:number_of_regions(grid))

minimum_yspacing(grid::MultiRegionGrid) =
    minimum(minimum_yspacing(grid[r]) for r in 1:number_of_regions(grid))

minimum_zspacing(grid::MultiRegionGrid) =
    minimum(minimum_zspacing(grid[r]) for r in 1:number_of_regions(grid))

minimum_xspacing(grid::MultiRegionGrid, ℓx, ℓy, ℓz) =
    minimum(minimum_xspacing(grid[r], ℓx, ℓy, ℓz) for r in 1:number_of_regions(grid))

minimum_yspacing(grid::MultiRegionGrid, ℓx, ℓy, ℓz) =
    minimum(minimum_yspacing(grid[r], ℓx, ℓy, ℓz) for r in 1:number_of_regions(grid))

minimum_zspacing(grid::MultiRegionGrid, ℓx, ℓy, ℓz) =
    minimum(minimum_zspacing(grid[r], ℓx, ℓy, ℓz) for r in 1:number_of_regions(grid))

@inline Base.length(mrg::MultiRegionGrid)         = Base.length(mrg.region_grids)
@inline Base.length(mrg::ImmersedMultiRegionGrid) = Base.length(mrg.underlying_grid.region_grids)

# the default free surface solver; see Models.HydrostaticFreeSurfaceModels
default_free_surface(grid::MultiRegionGrid; gravitational_acceleration=g_Earth) =
    SplitExplicitFreeSurface(; substeps=50, gravitational_acceleration)

"""
    MultiRegionGrid(global_grid; partition = XPartition(2))

Split a `global_grid` into different regions.

Positional Arguments
====================

- `global_grid`: the grid to be divided into regions.

Keyword Arguments
=================

- `partition`: the partitioning required. The implemented partitioning are `XPartition`
               (division along the ``x`` direction) and `YPartition` (division along
               the ``y`` direction).

Example
=======

```@example multiregion
julia> using Oceananigans

julia> using Oceananigans.MultiRegion: MultiRegionGrid, XPartition

julia> grid = RectilinearGrid(size=(12, 12), extent=(1, 1), topology=(Bounded, Bounded, Flat));

julia> multi_region_grid = MultiRegionGrid(grid, partition = XPartition(4))
```
"""
function MultiRegionGrid(global_grid; partition = XPartition(2))

    @warn "MultiRegion functionalities are experimental: help the development by reporting bugs or non-implemented features!"

    if length(partition) == 1
        return global_grid
    end

    arch = architecture(global_grid)
    connectivity = Connectivity(partition, global_grid)

    global_grid  = on_architecture(CPU(), global_grid)
    local_size   = MultiRegionObject(partition_size(partition, global_grid))
    local_extent = MultiRegionObject(partition_extent(partition, global_grid))
    local_topo   = MultiRegionObject(partition_topology(partition, global_grid))

    global_topo  = topology(global_grid)

    FT = eltype(global_grid)

    args = (Reference(global_grid),
            Reference(arch),
            local_topo,
            local_size,
            local_extent,
            Reference(partition),
            Iterate(1:length(partition)))

    region_grids = construct_regionally(construct_grid, args...)

    # Propagate the vertical coordinate type in the `MultiRegionGrid`
    CZ = typeof(global_grid.z)

    return MultiRegionGrid{FT, global_topo[1], global_topo[2], global_topo[3], CZ}(arch, partition, connectivity, region_grids)
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

"""
    reconstruct_global_grid(mrg::MultiRegionGrid)

Reconstruct the `mrg` global grid associated with the `MultiRegionGrid` on `architecture(mrg)`.
"""
function reconstruct_global_grid(mrg::MultiRegionGrid)
    size   = reconstruct_size(mrg, mrg.partition)
    extent = reconstruct_extent(mrg, mrg.partition)
    topo   = topology(mrg)
    return construct_grid(mrg.region_grids[1], architecture(mrg), topo, size, extent)
end

#####
##### `ImmersedMultiRegionGrid` functionalities
#####

function reconstruct_global_grid(mrg::ImmersedMultiRegionGrid)
    global_grid     = reconstruct_global_grid(mrg.underlying_grid)
    global_immersed_boundary = reconstruct_global_immersed_boundary(mrg.immersed_boundary)
    global_immersed_boundary = on_architecture(architecture(mrg), global_immersed_boundary)

    return ImmersedBoundaryGrid(global_grid, global_immersed_boundary)
end

reconstruct_global_immersed_boundary(g::GridFittedBottom{<:Field})   =   GridFittedBottom(reconstruct_global_field(g.bottom_height), g.immersed_condition)
reconstruct_global_immersed_boundary(g::PartialCellBottom{<:Field})  =  PartialCellBottom(reconstruct_global_field(g.bottom_height), g.minimum_fractional_cell_height)
reconstruct_global_immersed_boundary(g::GridFittedBoundary{<:Field}) = GridFittedBoundary(reconstruct_global_field(g.mask))

@inline  getregion(mrg::ImmersedMultiRegionGrid{FT, TX, TY, TZ}, r) where {FT, TX, TY, TZ} = ImmersedBoundaryGrid{TX, TY, TZ}(_getregion(mrg.underlying_grid, r),
                                                                                                                              _getregion(mrg.immersed_boundary, r),
                                                                                                                              _getregion(mrg.interior_active_cells, r),
                                                                                                                              _getregion(mrg.active_z_columns, r))

@inline _getregion(mrg::ImmersedMultiRegionGrid{FT, TX, TY, TZ}, r) where {FT, TX, TY, TZ} = ImmersedBoundaryGrid{TX, TY, TZ}(getregion(mrg.underlying_grid, r),
                                                                                                                              getregion(mrg.immersed_boundary, r),
                                                                                                                              getregion(mrg.interior_active_cells, r),
                                                                                                                              getregion(mrg.active_z_columns, r))

"""
    multi_region_object_from_array(a::AbstractArray, mrg::MultiRegionGrid)

Adapt an array `a` to be compatible with a `MultiRegionGrid`.
"""
function multi_region_object_from_array(a::AbstractArray, mrg::MultiRegionGrid)
    local_size = construct_regionally(size, mrg)
    arch = architecture(mrg)
    a  = on_architecture(CPU(), a)
    ma = construct_regionally(partition, a, mrg.partition, local_size, Iterate(1:length(mrg)), arch)
    return ma
end

# Fallback!
multi_region_object_from_array(a::AbstractArray, grid) = on_architecture(architecture(grid), a)

####
#### Utilities for MultiRegionGrid
####

new_data(FT::DataType, mrg::MultiRegionGrids, args...) = construct_regionally(new_data, FT, mrg, args...)

# This is kind of annoying but it is necessary to have compatible MultiRegion and Distributed
function with_halo(new_halo, mrg::MultiRegionGrid)
    partition = mrg.partition
    cpu_mrg   = on_architecture(CPU(), mrg)

    global_grid = reconstruct_global_grid(cpu_mrg)
    new_global  = with_halo(new_halo, global_grid)
    new_global  = on_architecture(architecture(mrg), new_global)

    return MultiRegionGrid(new_global; partition)
end

function on_architecture(arch, mrg::MultiRegionGrid{FT, TX, TY, TZ, CZ}) where {FT, TX, TY, TZ, CZ}
    new_grids = on_architecture(arch, mrg.region_grids)
    return MultiRegionGrid{FT, TX, TY, TZ, CZ}(arch, mrg.partition, mrg.connectivity, new_grids)
end

Base.summary(mrg::MultiRegionGrids{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    "MultiRegionGrid{$FT, $TX, $TY, $TZ} with $(summary(mrg.partition)) on $(string(typeof(mrg.region_grids[1]).name.wrapper))"

function Base.show(io::IO, mrg::MultiRegionGrids{FT}) where FT
    TX, TY, TZ = Oceananigans.Grids.topology_strs(mrg)
    return print(io, "$(grid_name(mrg)){$FT, $TX, $TY, $TZ} partitioned on $(architecture(mrg)): \n",
                     "├── region_grids: $(summary(mrg.region_grids[1])) \n",
                     "├── partition: $(summary(mrg.partition)) \n",
                     "└── connectivity: $(summary(mrg.connectivity))")
end

function Base.:(==)(mrg₁::MultiRegionGrids, mrg₂::MultiRegionGrids)
    #check if grids are of the same type
    vals = construct_regionally(Base.:(==), mrg₁, mrg₂)
    return all(vals.regional_objects)
end

####
#### This works only for homogenous partitioning
####

size(mrg::MultiRegionGrids) = size(getregion(mrg, 1))
halo_size(mrg::MultiRegionGrids) = halo_size(getregion(mrg, 1))

size(mrg::MultiRegionGrids, loc::Tuple, indices::MultiRegionObject) =
    size(getregion(mrg, 1), loc, getregion(indices, 1))

####
#### Get property for `MultiRegionGrid` (gets the properties of region 1)
#### In general getproperty should never be used as a MultiRegionGrid
#### Should be used only in combination with an @apply_regionally
####

grids(mrg::MultiRegionGrid) = mrg.region_grids

getmultiproperty(mrg::MultiRegionGrid, x::Symbol) = construct_regionally(Base.getproperty, grids(mrg), x)

const MRG = MultiRegionGrid

@inline Base.getproperty(mrg::MRG, property::Symbol)                 = get_multi_property(mrg, Val(property))
@inline get_multi_property(mrg::MRG, ::Val{property}) where property = getproperty(getindex(getfield(mrg, :region_grids), 1), property)
@inline get_multi_property(mrg::MRG, ::Val{:architecture})           = getfield(mrg, :architecture)
@inline get_multi_property(mrg::MRG, ::Val{:partition})              = getfield(mrg, :partition)
@inline get_multi_property(mrg::MRG, ::Val{:connectivity})           = getfield(mrg, :connectivity)
@inline get_multi_property(mrg::MRG, ::Val{:region_grids})           = getfield(mrg, :region_grids)
