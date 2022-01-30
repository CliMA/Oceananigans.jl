using Oceananigans.Fields: OneField
using Oceananigans.Grids: architecture
using Oceananigans.Architectures: arch_array
import Oceananigans.Fields: condition_operand, conditional_length, set!

# For conditional reductions such as mean(u * v, condition = u .> 0))

struct ConditionalOperation{LX, LY, LZ, O, F, G, C, M, T} <: AbstractOperation{LX, LY, LZ, G, T} 
    operand :: O
    func :: F
    grid :: G
    condition :: C
    mask :: M

     function ConditionalOperation{LX, LY, LZ}(operand::O, func::F, grid::G, condition::C, mask::M) where {LX, LY, LZ, O, F, G, C, M}
         T = eltype(operand)
         return new{LX, LY, LZ, O, F, G, C, M, T}(operand, func, grid, condition, mask)
     end
end

"""
    ConditionalOperation{LX, LY, LZ}(operand, func, grid, condition, mask)

Returns an abstract representation of a masking procedure applied when `condition` is satisfied on a field 
described by `func(operand)`.

Positional arguments
====================

- `operand`: The `AbstractField` to be masked (it must have a `grid` property!)

Keyword arguments
=================

- `func`: A unary transformation applied element-wise to the field `operand` at locations where `condition == true`. Default is `identity`

- `condition`: either a function of `(i, j, k, grid, operand)` returning a Boolean,
               or a 3-dimensional Boolean `AbstractArray`. At locations where `condition == false`,
               operand will be masked by `mask`

- `mask`: the scalar mask

`condition_operand` is a conveniece function used to construct a `ConditionalOperation`

`condition_operand(func::Function, operand::AbstractField, condition, mask) = ConditionalOperation(operand; func, condition, mask)`

Example
=======

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.Fields: condition_operand

julia> c = CenterField(RectilinearGrid(size=(2, 1, 1), extent=(1, 1, 1)));

julia> f(i, j, k, grid, c) = i < 2; d = condition_operand(cos, c, f, 10)
ConditionalOperation at (Center, Center, Center)
├── operand: 2×1×1 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 2×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── func: typeof(cos)
├── condition: typeof(f)
└── mask: 10

julia> d[1, 1, 1]
1.0

julia> d[2, 1, 1]
10
```
"""
function ConditionalOperation(operand::AbstractField; func = identity, condition = nothing, mask = 0)
    grid = operand.grid
    LX, LY, LZ = location(operand)
    return ConditionalOperation{LX, LY, LZ}(operand, func, grid, condition, mask)
end

function ConditionalOperation(c::ConditionalOperation; func = nothing, condition = nothing, mask = nothing)
    LX, LY, LZ = location(c)
    return ConditionalOperation{LX, LY, LZ}(c.operand, c.func, c.grid, c.condition, c.mask)
end

@inline condition_operand(func::Function, operand::AbstractField, condition, mask) = ConditionalOperation(operand; func, condition, mask)
@inline condition_operand(func::Function, operand::AbstractField, ::Nothing, mask) = ConditionalOperation(operand; func, condition = truefunc, mask)
@inline function condition_operand(func::Function, operand::AbstractField, condition::AbstractArray, mask) 
    condition = arch_array(architecture(operand.grid), condition)
    return ConditionalOperation(operand; func, condition, mask)
end

@inline condition_operand(func::typeof(identity), c::ConditionalOperation, ::Nothing, mask) = ConditionalOperation(c; mask)
@inline condition_operand(func::Function, c::ConditionalOperation, ::Nothing, mask) = ConditionalOperation(c; func, mask)

@inline truefunc(args...) = true

@inline condition_onefield(c::ConditionalOperation{LX, LY, LZ}, mask) where {LX, LY, LZ} =
                              ConditionalOperation{LX, LY, LZ}(OneField(Int), identity, c.grid, c.condition, mask)

@inline conditional_length(c::ConditionalOperation)       = sum(condition_onefield(c, 0))
@inline conditional_length(c::ConditionalOperation, dims) = sum(condition_onefield(c, 0); dims = dims)

Adapt.adapt_structure(to, c::ConditionalOperation{LX, LY, LZ}) where {LX, LY, LZ} =
            ConditionalOperation{LX, LY, LZ}(adapt(to, c.operand),
                                     adapt(to, c.func), 
                                     adapt(to, c.grid),
                                     adapt(to, c.condition),
                                     adapt(to, c.mask))

@inline function Base.getindex(c::ConditionalOperation, i, j, k) 
    return ifelse(get_condition(c.condition, i, j, k, c.grid, c), 
                  c.func(getindex(c.operand, i, j, k)),
                  c.mask)
end

@inline concretize_condition!(c::ConditionalOperation) = set!(c.operand, c)

function concretize_condition(c::ConditionalOperation)
    f = similar(c.operand)
    set!(f, c)
    return f
end

@inline get_condition(condition, i, j, k, grid, args...)                = condition(i, j, k, grid, args...)
@inline get_condition(condition::AbstractArray, i, j, k, grid, args...) = @inbounds condition[i, j, k]


Base.summary(c::ConditionalOperation) = string("ConditionalOperation of ", summary(c.operand), " with condition ", summary(c.condition))

Base.show(io::IO, operation::ConditionalOperation) =
    print(io,
          "ConditionalOperation at $(location(operation))", '\n',
          "├── operand: ", summary(operation.operand), '\n',
          "├── grid: ", summary(operation.grid), '\n',
          "├── func: ", summary(operation.func), '\n',
          "├── condition: ", summary(operation.condition), '\n',
          "└── mask: ", operation.mask)