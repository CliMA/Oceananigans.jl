@inline (bc::BoundaryFunction{:x, Y, Z})(j, k, grid, time, args...) where {Y, Z} =
    bc.func(ynode(Y, j, grid), znode(Z, k, grid), time)

@inline (bc::BoundaryFunction{:y, X, Z})(i, k, grid, time, args...) where {X, Z} =
    bc.func(xnode(X, i, grid), znode(Z, k, grid), time)

@inline (bc::BoundaryFunction{:z, X, Y})(i, j, grid, time, args...) where {X, Y} =
    bc.func(xnode(X, i, grid), ynode(Y, j, grid), time)
