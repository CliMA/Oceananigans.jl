module OceananigansConservativeRegriddingExt

using Adapt: Adapt
using ConservativeRegridding: regrid!
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: AbstractField

import Oceananigans.Architectures: on_architecture
import Oceananigans.Fields: ConservativeRegriddedField, compute_at!, indices

struct ConservativeRegridOperation{LX, LY, LZ, G, T, S, D, R} <: AbstractOperation{LX, LY, LZ, G, T}
    grid :: G
    source :: S
    destination :: D
    regridder :: R
end

struct ConservativeRegridKernelOperation{LX, LY, LZ, G, T, D} <: AbstractOperation{LX, LY, LZ, G, T}
    grid :: G
    destination :: D
end

function ConservativeRegridOperation(destination::AbstractField{LX, LY, LZ, G, T},
                                     regridder,
                                     source) where {LX, LY, LZ, G, T}
    S, D, R = typeof(source), typeof(destination), typeof(regridder)
    return ConservativeRegridOperation{LX, LY, LZ, G, T, S, D, R}(destination.grid,
                                                                  source, destination, regridder)
end

ConservativeRegriddedField(destination::AbstractField, regridder, source::AbstractField) =
    ConservativeRegridOperation(destination, regridder, source)

const ConservativeRegridLookup = Union{ConservativeRegridOperation, ConservativeRegridKernelOperation}

@inline Base.getindex(operation::ConservativeRegridLookup, i, j, k) =
    @inbounds operation.destination[i, j, k]

indices(operation::ConservativeRegridOperation) = indices(operation.destination)

function compute_at!(operation::ConservativeRegridOperation, time)
    compute_at!(operation.source, time)
    regrid!(operation.destination, operation.regridder, operation.source)
    return nothing
end

function Adapt.adapt_structure(to,
                               operation::ConservativeRegridOperation{LX, LY, LZ, G, T}) where {LX, LY, LZ, G, T}
    grid = Adapt.adapt(to, operation.grid)
    destination = Adapt.adapt(to, operation.destination)
    D = typeof(destination)
    return ConservativeRegridKernelOperation{LX, LY, LZ, typeof(grid), T, D}(grid, destination)
end

on_architecture(to, operation::ConservativeRegridOperation) =
    ConservativeRegridOperation(on_architecture(to, operation.destination),
                                on_architecture(to, operation.regridder),
                                on_architecture(to, operation.source))

end # module
