module Grids

export RegularCartesianGrid, VerticallyStretchedCartesianGrid

using OffsetArrays
using Oceananigans.Architectures

include("grid_utils.jl")
include("regular_cartesian_grid.jl")
include("vertically_stretched_cartesian_grid.jl")
include("show_grids.jl")

end
