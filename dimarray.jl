using Oceananigans
using Oceananigans.Fields: AbstractField
using DimensionalData
import DimensionalData: DimArray

grid = RectilinearGrid(size = (4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid)
c = CenterField(grid)

function DimArray(c::Oceananigans.Fields.AbstractField)
    x, y, z = nodes(c)
    return DimArray(interior(c), (X(x), Y(y), Z(z)))
end

C = DimArray(c)
