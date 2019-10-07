"""
    SimpleForcing([location=(Cell, Cell, Cell),] forcing_function::Function)

Construct forcing for a field at `location` from `forcing_function(x, y, z, t)`, 
where `x, y, z` is position and `t` is time.
"""
struct SimpleForcing{Lx, Ly, Lz, F}
    func :: F
end

"""
    SimpleForcing([location=(Cell, Cell, Cell),] forcing::SimpleForcing)

Construct forcing for a field at `location` using `forcing.func`.
"""
SimpleForcing(location::Tuple, func::Function) = 
    SimpleForcing{location[1], location[2], location[3], typeof(func)}(func)

SimpleForcing(func::Function) = SimpleForcing((Cell, Cell, Cell), func)

SimpleForcing(location::Tuple, forcing::SimpleForcing) = SimpleForcing(location, forcing.func)

@inline (forcing::SimpleForcing{X, Y, Z})(i, j, k, grid, time, U, C, params) where {X, Y, Z} =
    @inbounds forcing.func(xnode(X, i, grid), ynode(Y, j, grid), znode(Z, k, grid), time)

at_location(location, u::Function) = u
at_location(location, u::SimpleForcing) = SimpleForcing(location, u)

"""
    ModelForcing(; kwargs...)

Return a named tuple of forcing functions for each solution field.
"""
function ModelForcing(; u=zerofunk, v=zerofunk, w=zerofunk, T=zerofunk, S=zerofunk)
    u = at_location((Face, Cell, Cell), u)
    v = at_location((Cell, Face, Cell), v)
    w = at_location((Cell, Cell, Face), w)

    return (u=u, v=v, w=w, T=T, S=S)
end


