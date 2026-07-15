module OceananigansConservativeRegriddingExt

using Adapt: Adapt
using ConservativeRegridding: Regridder, regrid!
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: Architectures, architecture, CPU
using Oceananigans.Fields: Fields, AbstractField, Field
using Oceananigans.ImmersedBoundaries: underlying_grid

struct RegriddedOperation{LX, LY, LZ, G, T, S, D, R} <: AbstractOperation{LX, LY, LZ, G, T}
    grid :: G
    source :: S
    destination :: D
    regridder :: R
end

struct RegriddedKernelOperation{LX, LY, LZ, G, T, D} <: AbstractOperation{LX, LY, LZ, G, T}
    grid :: G
    destination :: D
end

function RegriddedOperation(destination::AbstractField{LX, LY, LZ, G, T},
                           regridder,
                           source) where {LX, LY, LZ, G, T}
    S, D, R = typeof(source), typeof(destination), typeof(regridder)
    return RegriddedOperation{LX, LY, LZ, G, T, S, D, R}(destination.grid,
                                                         source, destination, regridder)
end

Fields.RegriddedField(destination::AbstractField, regridder, source::AbstractField) =
    RegriddedOperation(destination, regridder, source)

function Fields.RegriddedField(source::AbstractField{LX, LY, LZ}, destination_grid) where {LX, LY, LZ}
    source_architecture = architecture(source)
    destination_grid = Architectures.on_architecture(source_architecture, destination_grid)
    destination = Field{LX, LY, LZ}(destination_grid)

    source_grid = Architectures.on_architecture(CPU(), underlying_grid(source))
    regrid_grid = Architectures.on_architecture(CPU(), underlying_grid(destination_grid))
    regridder = Regridder(regrid_grid, source_grid)
    regridder = Architectures.on_architecture(source_architecture, regridder)

    return RegriddedOperation(destination, regridder, source)
end

const RegriddedLookup = Union{RegriddedOperation, RegriddedKernelOperation}

@inline Base.getindex(operation::RegriddedLookup, i, j, k) =
    @inbounds operation.destination[i, j, k]

Fields.indices(operation::RegriddedOperation) = Fields.indices(operation.destination)

function Fields.compute_at!(operation::RegriddedOperation, time)
    Fields.compute_at!(operation.source, time)
    regrid!(operation.destination, operation.regridder, operation.source)
    return nothing
end

function Adapt.adapt_structure(to,
                               operation::RegriddedOperation{LX, LY, LZ, G, T}) where {LX, LY, LZ, G, T}
    grid = Adapt.adapt(to, operation.grid)
    destination = Adapt.adapt(to, operation.destination)
    D = typeof(destination)
    return RegriddedKernelOperation{LX, LY, LZ, typeof(grid), T, D}(grid, destination)
end

Architectures.on_architecture(to, operation::RegriddedOperation) =
    RegriddedOperation(Architectures.on_architecture(to, operation.destination),
                       Architectures.on_architecture(to, operation.regridder),
                       Architectures.on_architecture(to, operation.source))

end # module
