struct FunctionField{X, Y, Z, C, F, G} <: AbstractLocatedField{X, Y, Z, F, G}
     func :: F
     grid :: G
    clock :: C
    function FunctionField{X, Y, Z}(data, grid; clock=nothing) where {X, Y, Z}
        return new{X, Y, Z, typeof(clock), typeof(data), typeof(grid)}(data, grid, clock)
    end
end

architecture(::FunctionField) = nothing
data(f::FunctionField) = f
Base.parent(f::FunctionField) = f

@propagate_inbounds getindex(f::FunctionField{X, Y, Z, <:Nothing}, i, j, k) where {X, Y, Z} =
    f.func(xnode(X, i, f.grid), ynode(Y, j, f.grid), znode(Z, k, f.grid))

@propagate_inbounds getindex(f::FunctionField{X, Y, Z}, i, j, k) where {X, Y, Z} =
    f.func(xnode(X, i, f.grid), ynode(Y, j, f.grid), znode(Z, k, f.grid), f.clock.time)

set!(u::Field{X, Y, Z}, f::FunctionField{X, Y, Z, <:Nothing}) where {X, Y, Z} =
    set!(u, f.func)

function compute(f::FunctionField, arch)
    computed_f = Field(location(f), arch, f.grid)    
    set!(computed_f, f)
    return computed_f
end
