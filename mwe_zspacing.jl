using Oceananigans
using Oceananigans.Fields: @compute

grid = RectilinearGrid(size=(3, 3, 3), extent=(1, 1, 1));
bottom(x, y) = - 1/2
grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bottom))
@compute dz = Field(zspacings(grid, Center(), Center(), Center()));

display(interior(dz))

import Oceananigans: rspacings
using Oceananigans.Operators: Δr
function rspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δr_op = KernelFunctionOperation{LX, LY, LZ}(Δr, grid, ℓx, ℓy, ℓz)
    return Δr_op
end

@compute dr = Field(rspacings(grid, Center(), Center(), Center()));
display(interior(dr))
