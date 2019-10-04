"""
    Forcing(; kwargs...)

Return a named tuple of forcing functions
for each solution field.
"""
ModelForcing(; u=zerofunk, v=zerofunk, w=zerofunk, T=zerofunk, S=zerofunk) =
    (u=u, v=v, w=w, T=T, S=S)

struct SpatialForcing{Lx, Ly, Lz, F}
    func :: F
end

SpatialForcing(func::Function) = SpatialForcing{Cell, Cell, Cell, typeof(func)}(func)

@inline (forcing::SpatialForcing{X, Y, Z})(i, j, k, grid, time, U, C, params) where {X, Y, Z} =
    @inbounds forcing.func(xnode(X, i, grid), ynode(Y, j, grid), znode(Z, k, grid))
