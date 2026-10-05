"""
    RegriddedOperation(source, destination_grid)
    RegriddedOperation(destination, regridder, source)

Return a lazy operation that conservatively regrids `source` onto the grid of
`destination`.

`RegriddedOperation` is the host-side orchestration object. When it is computed,
it first computes `source` and then uses `regridder` to update `destination`.
When an enclosing operation is evaluated in a CPU or GPU kernel, adaptation
replaces the source and regridder with `nothing`. Regridding happens before
the enclosing kernel is launched; the kernel only reads
`destination[i, j, k]`.

The two-argument constructor allocates a destination field at the same location
as `source` and constructs the conservative regridder. Use the three-argument
constructor to supply and reuse an explicitly constructed `destination` and
`regridder`.

The two-argument constructor and computation methods are available when
ConservativeRegridding.jl is loaded.

# Example

```julia
regridded = RegriddedOperation(source, destination_grid)
output = Field(regridded)
```
"""
struct RegriddedOperation{LX, LY, LZ, G, T, S, D, R} <: AbstractOperation{LX, LY, LZ, G, T}
    grid :: G
    source :: S
    destination :: D
    regridder :: R
end

function RegriddedOperation(destination::AbstractField{LX, LY, LZ, G, T},
                            regridder,
                            source::AbstractField) where {LX, LY, LZ, G, T}
    S, D, R = typeof(source), typeof(destination), typeof(regridder)
    return RegriddedOperation{LX, LY, LZ, G, T, S, D, R}(destination.grid,
                                                          source, destination, regridder)
end

@inline Base.getindex(operation::RegriddedOperation, i, j, k) =
    @inbounds operation.destination[i, j, k]

indices(operation::RegriddedOperation) = indices(operation.destination)

function Adapt.adapt_structure(to,
                               operation::RegriddedOperation{LX, LY, LZ, G, T}) where {LX, LY, LZ, G, T}
    grid = Adapt.adapt(to, operation.grid)
    destination = Adapt.adapt(to, operation.destination)
    D = typeof(destination)
    return RegriddedOperation{LX, LY, LZ, typeof(grid), T, Nothing, D, Nothing}(grid,
                                                                                 nothing,
                                                                                 destination,
                                                                                 nothing)
end

Architectures.on_architecture(to, operation::RegriddedOperation) =
    RegriddedOperation(on_architecture(to, operation.destination),
                       on_architecture(to, operation.regridder),
                       on_architecture(to, operation.source))
