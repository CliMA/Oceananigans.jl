module MultiRegion

export MultiRegionGrid, MultiRegionField
export XPartition

# To delete
export switch_device!, getregion, getdevice

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using CUDA
using Adapt
using OffsetArrays

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

import Base: show, length, size

abstract type AbstractMultiGrid{FT, TX, TY, TZ, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch} end

abstract type AbstractMultiField{TX, TY, TZ, G, F, T} <: AbstractField{TX, TY, TZ, G, T, 3} end

abstract type AbstractPartition end

include("multi_region_utils.jl")
include("multi_region_transformation.jl")
include("multi_region_object.jl")
include("x_partitions.jl")
include("multi_region_grid.jl")
include("multi_region_field.jl")
include("multi_region_boundary_conditions.jl")
include("multi_region_reductions.jl")
# include("delete_me.jl")

end #module