import DimensionalData

function DimensionalData.dims(grid::AbstractRectilinearGrid, loc)
    x_dim = DimensionalData.X(xnodes(loc[1], grid))
    y_dim = DimensionalData.Y(xnodes(loc[2], grid))
    z_dim = DimensionalData.Z(xnodes(loc[3], grid))
    return (x_dim, y_dim, z_dim)
end
