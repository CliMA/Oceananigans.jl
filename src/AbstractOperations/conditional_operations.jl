using Oceananigans.Fields: OneField
using Oceananigans.Grids: architecture

import Oceananigans.Architectures: on_architecture
import Oceananigans.Fields: condition_operand, conditional_length, set!, compute_at!, indices

# For conditional reductions such as mean(u * v, condition = u .> 0))
struct ConditionalOperation{LX, LY, LZ, F, C, O, G, M, T} <: AbstractOperation{LX, LY, LZ, G, T}
    operand :: O
    func :: F
    grid :: G
    condition :: C
    mask :: M

    function ConditionalOperation{LX, LY, LZ}(operand::O, func, grid::G,
                                              condition::C, mask::M) where {LX, LY, LZ, O, G, C, M}
        if func === Base.identity
            func = nothing
        end
        T = eltype(operand)
        F = typeof(func)
        return new{LX, LY, LZ, F, C, O, G, M, T}(operand, func, grid, condition, mask)
    end
end

"""
    ConditionalOperation(operand::AbstractField;
                         func = nothing,
                         condition = nothing,
                         mask = 0)

Return an abstract representation of a masking procedure applied when `condition` is satisfied on a field
described by `func(operand)`.

Positional arguments
====================

- `operand`: The `AbstractField` to be masked.

Keyword arguments
=================

- `func`: A unary transformation applied element-wise to the field `operand` at locations where
          `condition == true`. Default is `nothing` which applies no transformation.

- `condition`: either a function of `(i, j, k, grid, operand)` returning a Boolean,
               or a 3-dimensional Boolean `AbstractArray`. At locations where `condition == false`,
               operand will be masked by `mask`.

- `mask`: the scalar mask. Default: 0.

`condition_operand` is a convenience function used to construct a `ConditionalOperation`

`condition_operand(func::Function, operand::AbstractField, condition, mask) = ConditionalOperation(operand; func, condition, mask)`

Example
=======

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.Fields: condition_operand

julia> c = CenterField(RectilinearGrid(size=(2, 1, 1), extent=(1, 1, 1)));

julia> add_2(c) = c + 2
add_2 (generic function with 1 method)

julia> f(i, j, k, grid, c) = i < 2; d = condition_operand(add_2, c, f, 10.0)
ConditionalOperation at (Center, Center, Center)
├── operand: 2×1×1 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 2×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×1×1 halo
├── func: add_2 (generic function with 1 method)
├── condition: f (generic function with 1 method)
└── mask: 10.0

julia> d[1, 1, 1]
2.0

julia> d[2, 1, 1]
10.0
```
"""
function ConditionalOperation(operand::AbstractField;
                              func = nothing,
                              condition = nothing,
                              mask = zero(eltype(operand)))
    LX, LY, LZ = location(operand)
    return ConditionalOperation{LX, LY, LZ}(operand, func, operand.grid, condition, mask)
end

function ConditionalOperation(c::ConditionalOperation;
                              func = c.func,
                              condition = c.condition,
                              mask = c.mask)

    LX, LY, LZ = location(c)
    return ConditionalOperation{LX, LY, LZ}(c.operand, func, c.grid, condition, mask)
end

@inline function Base.getindex(co::ConditionalOperation, i, j, k)
    conditioned = evaluate_condition(co.condition, i, j, k, c.grid, c)
    value = getindex(co.operand, i, j, k)
    func_value = co.func(value)
    return ifelse(conditioned, value, c.mask)
end

# Some special cases
const NoFuncCO = ConditionalOperation{<:Any, <:Any, <:Any, Nothing}
const NoConditionCO = ConditionalOperation{<:Any, <:Any, <:Any, <:Any, Nothing}
const NoFuncNoConditionCO = ConditionalOperation{<:Any, <:Any, <:Any, Nothing, Nothing}

using Base: @propagate_inbounds
@propagate_inbounds function Base.getindex(co::NoConditionCO, i, j, k)
    value = getindex(co.operand, i, j, k)
    return co.func(value)
end

@propagate_inbounds Base.getindex(co::NoFuncNoConditionCO, i, j, k) = getindex(co.operand, i, j, k)

@propagate_inbounds function Base.getindex(co::NoFuncCO, i, j, k)
    conditioned = evaluate_condition(co.condition, i, j, k, co.grid, co)
    value = getindex(co.operand, i, j, k)
    return ifelse(conditioned, value, co.mask)
end

# Conditions: general, nothing, array
@inline evaluate_condition(condition, i, j, k, grid, args...) = condition(i, j, k, grid, args...)
@inline evaluate_condition(::Nothing, i, j, k, grid, args...) = true
@propagate_inbounds evaluate_condition(condition::AbstractArray, i, j, k, grid, args...) = condition[i, j, k]

@inline condition_operand(func, op, condition, mask) = ConditionalOperation(op; func, condition, mask)

@inline function condition_operand(func, operand, condition::AbstractArray, mask)
    condition = on_architecture(architecture(operand.grid), condition)
    return ConditionalOperation(operand; func, condition, mask)
end

@inline materialize_condition!(c::ConditionalOperation) = set!(c.operand, c)

function materialize_condition(c::ConditionalOperation)
    f = similar(c.operand)
    set!(f, c)
    return f
end

@inline function conditional_one(c::ConditionalOperation, mask)
    LX, LY, LZ = location(c)
    one_field = OneField(Int)
    return ConditionalOperation{LX, LY, LZ}(one_field, nothing, c.grid, c.condition, mask)
end

@inline conditional_length(c::ConditionalOperation) = sum(conditional_one(c, 0))
@inline conditional_length(c::ConditionalOperation, dims) = sum(conditional_one(c, 0); dims = dims)

Adapt.adapt_structure(to, c::ConditionalOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    ConditionalOperation{LX, LY, LZ}(adapt(to, c.operand),
                                     adapt(to, c.func),
                                     adapt(to, c.grid),
                                     adapt(to, c.condition),
                                     adapt(to, c.mask))

on_architecture(to, c::ConditionalOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    ConditionalOperation{LX, LY, LZ}(on_architecture(to, c.operand),
                                     on_architecture(to, c.func),
                                     on_architecture(to, c.grid),
                                     on_architecture(to, c.condition),
                                     on_architecture(to, c.mask))

Base.summary(c::ConditionalOperation) = string("ConditionalOperation of ", summary(c.operand), " with condition ", summary(c.condition))

compute_at!(c::ConditionalOperation, time) = compute_at!(c.operand, time)
indices(c::ConditionalOperation) = indices(c.operand)

Base.show(io::IO, operation::ConditionalOperation) =
    print(io, "ConditionalOperation at $(location(operation))", '\n',
              "├── operand: ", summary(operation.operand), '\n',
              "├── grid: ", summary(operation.grid), '\n',
              "├── func: ", summary(operation.func), '\n',
              "├── condition: ", summary(operation.condition), '\n',
              "└── mask: ", operation.mask)

