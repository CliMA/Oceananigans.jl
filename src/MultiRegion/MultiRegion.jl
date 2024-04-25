module MultiRegion

export MultiRegionGrid, MultiRegionField
export XPartition, YPartition, Connectivity
export AbstractRegionSide, East, West, North, South
export CubedSpherePartition, ConformalCubedSphereGrid, CubedSphereField

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Models
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.Utils

using Adapt
using CUDA
using DocStringExtensions
using OffsetArrays

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: Reference, Iterate, getnamewrapper
using Oceananigans.Grids: AbstractUnderlyingGrid

using KernelAbstractions: @kernel, @index

import Base: show, length, size

import Oceananigans.Utils:
                getdevice,
                switch_device!,
                devices,
                isregional,
                getregion,
                _getregion,
                sync_all_devices!

abstract type AbstractMultiRegionGrid{FT, TX, TY, TZ, Arch} <: AbstractUnderlyingGrid{FT, TX, TY, TZ, Arch} end

abstract type AbstractPartition end

abstract type AbstractConnectivity end

abstract type AbstractRegionSide end

struct West <: AbstractRegionSide end
struct East <: AbstractRegionSide end
struct North <: AbstractRegionSide end
struct South <: AbstractRegionSide end

struct XPartition{N} <: AbstractPartition
    div :: N

    function XPartition(sizes)
        if length(sizes) > 1 && all(y -> y == sizes[1], sizes)
            sizes = length(sizes)
        end

        return new{typeof(sizes)}(sizes)
    end
end

struct YPartition{N} <: AbstractPartition
    div :: N

    function YPartition(sizes) 
        if length(sizes) > 1 && all(y -> y == sizes[1], sizes)
            sizes = length(sizes)
        end

        return new{typeof(sizes)}(sizes)
    end
end

include("multi_region_utils.jl")
include("multi_region_connectivity.jl")
include("x_partitions.jl")
include("y_partitions.jl")
include("cubed_sphere_partitions.jl")
include("cubed_sphere_connectivity.jl")
include("multi_region_grid.jl")
include("multi_region_cubed_sphere_grid.jl")
include("cubed_sphere_field.jl")
include("cubed_sphere_boundary_conditions.jl")
include("multi_region_field.jl")
include("multi_region_abstract_operations.jl")
include("multi_region_boundary_conditions.jl")
include("multi_region_reductions.jl")
include("unified_implicit_free_surface_solver.jl")
include("multi_region_split_explicit_free_surface.jl")
include("multi_region_models.jl")
include("multi_region_output_writers.jl")

end #module
