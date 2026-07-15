module OceananigansConservativeRegriddingExt

using Adapt: Adapt
using ConservativeRegridding: Regridder, regrid!
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: Architectures, architecture, CPU
using Oceananigans.Fields: Fields, AbstractField, Field
using Oceananigans.ImmersedBoundaries: underlying_grid

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

function ConservativeRegriddedField(source::AbstractField{LX, LY, LZ}, destination_grid) where {LX, LY, LZ}
    source_architecture = architecture(source)
    destination_grid = on_architecture(source_architecture, destination_grid)
    destination = Field{LX, LY, LZ}(destination_grid)

    source_grid = on_architecture(CPU(), underlying_grid(source))
    regrid_grid = on_architecture(CPU(), underlying_grid(destination_grid))
    regridder = Regridder(regrid_grid, source_grid)
    regridder = on_architecture(source_architecture, regridder)

    return ConservativeRegridOperation(destination, regridder, source)
end

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
