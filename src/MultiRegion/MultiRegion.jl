module MultiRegion

export MultiRegionGrid, MultiRegionField
export XPartition
export assoc_field, assoc_grid, assoc_device

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Architectures
using CUDA
using Adapt
using OffsetArrays

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

import Base: show, length, size

abstract type AbstractMultiGrid{FT, TX, TY, TZ, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch} end

abstract type AbstractMultiField{TX, TY, TZ, G, F, T} <: AbstractField{TX, TY, TZ, G, T, 3} end

abstract type AbstractPartition end

include("multi_region_utils.jl")
include("x_partitions.jl")
include("multi_region_grid.jl")
include("multi_region_field_bcs.jl")
include("multi_region_field.jl")
include("adaptation_for_fields.jl")

end #module