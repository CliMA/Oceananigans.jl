"""
    ModelForcing(; kwargs...)

Return a named tuple of forcing functions for each solution field.
"""
ModelForcing(; u=zerofunk, v=zerofunk, w=zerofunk, T=zerofunk, S=zerofunk) =
    (u=u, v=v, w=w, T=T, S=S)

"""
    NaturalForcing([location=(Cell, Cell, Cell),] forcing_function::Function)

Construct forcing for a field at `location` from `forcing_function(x, y, z, t)`, 
where `x, y, z` is position and `t` is time.
"""
struct NaturalForcing{Lx, Ly, Lz, F}
    func :: F
end

NaturalForcing(location::Tuple, func::Function) = 
    NaturalForcing{location[1], location[2], location[3], typeof(func)}(func)

"""
    NaturalForcing([location=(Cell, Cell, Cell),] forcing::NaturalForcing)

Construct forcing for a field at `location` using `forcing.func`.
"""
NaturalForcing(location::Tuple, forcing::NaturalForcing) = NaturalForcing(location, forcing.func)

NaturalForcing(func::Function) = NaturalForcing((Cell, Cell, Cell), func)

@inline (forcing::NaturalForcing{X, Y, Z})(i, j, k, grid, time, U, C, params) where {X, Y, Z} =
    @inbounds forcing.func(xnode(X, i, grid), ynode(Y, j, grid), znode(Z, k, grid), time)
