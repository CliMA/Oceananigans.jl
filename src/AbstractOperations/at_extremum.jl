using GPUArraysCore: @allowscalar
using Oceananigans.Grids: xnode, ynode, znode, λnode, φnode, rnode

"""
    at_maximum([f=identity,] c, coordinate; condition=nothing, mask=-Inf)

Return `coordinate` evaluated where `f(c)` is maximum.

`coordinate` may be

- a `Symbol` naming a coordinate of the grid, such as `:x`, `:y`, `:z`, `:λ`, `:φ`, or `:r`,
  which is evaluated at the location of `c`;
- an `AbstractField`, which is interpolated to the location of `c`;
- a `Tuple` of the above.

The maximum is located with `findmax(f, c; condition, mask)`, which runs on the architecture
of `c`; only the extremum's index and the requested values are transferred to the host.

Example
=======

```jldoctest at_maximum
using Oceananigans

grid = RectilinearGrid(size=(1, 1, 4), extent=(1, 1, 4))
T = CenterField(grid)
set!(T, (x, y, z) -> - (z + 1.4)^2)

at_maximum(T, :z)

# output

-1.5
```

A field at another location is interpolated to the location of the maximum:

```jldoctest at_maximum
w = ZFaceField(grid)
set!(w, (x, y, z) -> z)

at_maximum(T, (:z, w))

# output

(-1.5, -1.5)
```
"""
at_maximum(f, c::AbstractField, coordinate; kwargs...) =
    value_at(last(findmax(f, c; kwargs...)), c, coordinate)

"""
    at_minimum([f=identity,] c, coordinate; condition=nothing, mask=Inf)

Return `coordinate` evaluated where `f(c)` is minimum. See [`at_maximum`](@ref).
"""
at_minimum(f, c::AbstractField, coordinate; kwargs...) =
    value_at(last(findmin(f, c; kwargs...)), c, coordinate)

at_maximum(c::AbstractField, coordinate; kwargs...) = at_maximum(identity, c, coordinate; kwargs...)
at_minimum(c::AbstractField, coordinate; kwargs...) = at_minimum(identity, c, coordinate; kwargs...)

value_at(I, c, coordinates::Tuple) = map(coordinate -> value_at(I, c, coordinate), coordinates)
value_at(I, c, coordinate::Symbol) = value_at(I, c, Val(coordinate))

function value_at(I, c, field::AbstractField)
    interpolated = interpolate_operation(instantiated_location(c), field)
    return @allowscalar interpolated[Tuple(I)...]
end

for (name, node) in ((:x, :xnode), (:y, :ynode), (:z, :znode),
                     (:λ, :λnode), (:φ, :φnode), (:r, :rnode))
    @eval value_at(I, c, ::Val{$(QuoteNode(name))}) =
        @allowscalar $node(Tuple(I)..., c.grid, instantiated_location(c)...)
end

value_at(I, c, ::Val{coordinate}) where coordinate =
    throw(ArgumentError("Unknown coordinate :$coordinate; expected one of :x, :y, :z, :λ, :φ, :r"))
