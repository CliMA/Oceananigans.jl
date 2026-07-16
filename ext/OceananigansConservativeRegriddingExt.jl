module OceananigansConservativeRegriddingExt

using ConservativeRegridding: Regridder, regrid!
import Oceananigans.AbstractOperations: RegriddedOperation
using Oceananigans.Architectures: Architectures, architecture, CPU
using Oceananigans.Fields: Fields, AbstractField, Field
using Oceananigans.ImmersedBoundaries: underlying_grid

function RegriddedOperation(source::AbstractField{LX, LY, LZ}, destination_grid) where {LX, LY, LZ}
    source_architecture = architecture(source)
    destination_grid = Architectures.on_architecture(source_architecture, destination_grid)
    destination = Field{LX, LY, LZ}(destination_grid)

    source_grid = Architectures.on_architecture(CPU(), underlying_grid(source))
    regrid_grid = Architectures.on_architecture(CPU(), underlying_grid(destination_grid))
    regridder = Regridder(regrid_grid, source_grid)
    regridder = Architectures.on_architecture(source_architecture, regridder)

    return RegriddedOperation(destination, regridder, source)
end

function Fields.compute_at!(operation::RegriddedOperation, time)
    Fields.compute_at!(operation.source, time)
    regrid!(operation.destination, operation.regridder, operation.source)
    return nothing
end

end # module
