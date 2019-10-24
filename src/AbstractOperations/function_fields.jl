"""
    FunctionField{X, Y, Z, C, F, G} <: AbstractLocatedField{X, Y, Z, F, G}

An `AbstractLocatedField` that returns a function evaluated at location `(X, Y, Z)` (and time, if 
`C` is not `Nothing`) when indexed at `i, j, k`.
"""
struct FunctionField{X, Y, Z, C, F, G} <: AbstractLocatedField{X, Y, Z, F, G}
     func :: F
     grid :: G
    clock :: C

    """
        FunctionField{X, Y, Z}(func, grid; clock=nothing) where {X, Y, Z}

    Returns a `FunctionField` on `grid` and at location `X, Y, Z`. 
    
    If `clock` is not specified, then `func` must be a function with signature 
    `func(x, y, z)`. If clock is specified, `func` must be a function with signature 
    `func(x, y, z, t)`, where `t` is internally determined from `clock.time`.

    A FunctionField will return the result of `func(x, y, z [, t])` at `X, Y, Z` on 
    `grid` when indexed at `i, j, k`.
    """
    function FunctionField{X, Y, Z}(func, grid; clock=nothing) where {X, Y, Z}
        return new{X, Y, Z, typeof(clock), typeof(func), typeof(grid)}(func, grid, clock)
    end
end

"""
    FunctionField(L::Tuple, func, grid)
    
Returns a stationary `FunctionField` on `grid` and at location `L = (X, Y, Z)`,
where `func` is callable with signature `func(x, y, z)`.
"""
FunctionField(L::Tuple, func, grid) = FunctionField{L[1], L[2], L[3]}(func, grid)

# Ordinary functions needed for fields
architecture(::FunctionField) = nothing
data(f::FunctionField) = f
Base.parent(f::FunctionField) = f

@inline Base.getindex(f::FunctionField{X, Y, Z, <:Nothing}, i, j, k) where {X, Y, Z} =
    f.func(xnode(X, i, f.grid), ynode(Y, j, f.grid), znode(Z, k, f.grid))

@inline Base.getindex(f::FunctionField{X, Y, Z}, i, j, k) where {X, Y, Z} =
    f.func(xnode(X, i, f.grid), ynode(Y, j, f.grid), znode(Z, k, f.grid), f.clock.time)

@inline (f::FunctionField)(x, y, z) = f.func(x, y, z, f.clock.time)
@inline (f::FunctionField{X, Y, Z, <:Nothing})(x, y, z) where {X, Y, Z} = f.func(x, y, z)

# set! for function fields
set!(u, f::FunctionField) = set!(u, (x, y, z) -> f.func(x, y, z, f.clock.time))
set!(u, f::FunctionField{X, Y, Z, <:Nothing}) where {X, Y, Z} = set!(u, f.func)

Adapt.adapt_structure(to, f::FunctionField{X, Y, Z}) where {X, Y, Z} =
    FunctionField{X, Y, Z}(adapt(to, func), grid, clock)
