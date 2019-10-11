struct FunctionField{X, Y, Z, C, F, G} <: AbstractLocatedField{X, Y, Z, F, G}
     func :: F
     grid :: G
    clock :: C

    function FunctionField{X, Y, Z}(func, grid; clock=nothing) where {X, Y, Z}
        return new{X, Y, Z, typeof(clock), typeof(func), typeof(grid)}(func, grid, clock)
    end
end

FunctionField(L::Tuple, func, grid) = FunctionField{L[1], L[2], L[3]}(func, grid)

architecture(::FunctionField) = nothing
data(f::FunctionField) = f
Base.parent(f::FunctionField) = f

@inline Base.getindex(f::FunctionField{X, Y, Z, <:Nothing}, i, j, k) where {X, Y, Z} =
    f.func(xnode(X, i, f.grid), ynode(Y, j, f.grid), znode(Z, k, f.grid))

@inline Base.getindex(f::FunctionField{X, Y, Z}, i, j, k) where {X, Y, Z} =
    f.func(xnode(X, i, f.grid), ynode(Y, j, f.grid), znode(Z, k, f.grid), f.clock.time)

@inline (f::FunctionField)(x, y, z) = f.func(x, y, z, f.clock.time)
@inline (f::FunctionField{X, Y, Z, <:Nothing})(x, y, z) where {X, Y, Z} = f.func(x, y, z)

set!(u, f::FunctionField) = set!(u, (x, y, z) -> f.func(x, y, z, f.clock.time))
set!(u, f::FunctionField{X, Y, Z, <:Nothing}) where {X, Y, Z} = set!(u, f.func)

function compute(f::FunctionField, arch)
    computed_f = Field(location(f), arch, f.grid)    
    set!(computed_f, f)
    return computed_f
end

Adapt.adapt_structure(to, f::FunctionField{X, Y, Z}) where {X, Y, Z} =
    FunctionField{X, Y, Z}(adapt(to, func), grid, clock)
