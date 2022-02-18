module MultiRegion

export MultiRegionGrid

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Architectures
using CUDA
using OffsetArrays
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

import Base: show, length, size


include("x_partitions.jl")
include("multi_region_utils.jl")
include("multi_region_grid.jl")

end #module