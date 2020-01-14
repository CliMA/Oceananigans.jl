module Grids

export RegularCartesianGrid, VerticallyStretchedCartesianGrid

using Oceananigans

include("grid_utils.jl")
include("regular_cartesian_grid.jl")
include("vertically_stretched_cartesian_grid.jl")

end
