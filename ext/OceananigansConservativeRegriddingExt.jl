module OceananigansConservativeRegriddingExt

using ConservativeRegridding: Regridder, regrid!
using Oceananigans.AbstractOperations: AbstractOperations, RegriddedOperation
using Oceananigans.Architectures: Architectures, architecture, CPU
using Oceananigans.Fields: Fields, AbstractField, Field
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, underlying_grid
using SparseArrays: dropzeros!

function AbstractOperations.RegriddedOperation(source::AbstractField{LX, LY, LZ}, destination_grid) where {LX, LY, LZ}
    source_architecture = architecture(source)
    validate_destination_grid(source.grid, destination_grid)

    destination_grid = Architectures.on_architecture(source_architecture, destination_grid)
    destination = Field{LX, LY, LZ}(destination_grid)

    source_grid = regridding_grid_on_cpu(source.grid)
    regrid_grid = regridding_grid_on_cpu(destination_grid)
    regridder = active_column_regridder(regrid_grid, source_grid)
    regridder = Architectures.on_architecture(source_architecture, regridder)

    return RegriddedOperation(destination, regridder, source)
end

validate_destination_grid(source_grid, destination_grid) = nothing

function validate_destination_grid(source_grid::ImmersedBoundaryGrid, destination_grid)
    destination_grid isa ImmersedBoundaryGrid && return nothing

    msg = """Regridding from an ImmersedBoundaryGrid to a non-immersed grid is not supported.
             The source grid has inactive cells. The destination grid must also be an
             ImmersedBoundaryGrid so later operations know which destination cells are active."""

    return throw(ArgumentError(msg))
end

regridding_grid_on_cpu(grid) = Architectures.on_architecture(CPU(), grid)

function regridding_grid_on_cpu(grid::ImmersedBoundaryGrid)
    grid = Architectures.on_architecture(CPU(), grid)
    return ImmersedBoundaryGrid(grid.underlying_grid, grid.immersed_boundary; active_z_columns=true)
end

active_column_regridder(destination_grid, source_grid) =
    Regridder(underlying_grid(destination_grid), underlying_grid(source_grid))

function active_column_regridder(destination_grid::ImmersedBoundaryGrid, source_grid::ImmersedBoundaryGrid)
    regridder = active_column_regridder(destination_grid.underlying_grid, source_grid.underlying_grid)
    mask_inactive_columns!(regridder, destination_grid; dims=1)
    mask_inactive_columns!(regridder, source_grid; dims=2)
    return regridder
end

function active_column_mask(grid::ImmersedBoundaryGrid)
    active_columns = grid.active_z_columns
    isnothing(active_columns) && throw(ArgumentError("Immersed regridding requires `active_z_columns`."))

    mask = falses(grid.Nx * grid.Ny)

    for (i, j) in active_columns
        n = Int(i) + (Int(j) - 1) * grid.Nx
        @inbounds mask[n] = true
    end

    return mask
end

function mask_inactive_columns!(regridder, grid::ImmersedBoundaryGrid; dims)
    active_columns = active_column_mask(grid)
    inactive_columns = findall(!, active_columns)

    if dims == 1
        regridder.intersections[inactive_columns, :] .= 0
    elseif dims == 2
        regridder.intersections[:, inactive_columns] .= 0
    else
        throw(ArgumentError("Inactive conservative regridding columns can only be masked along dims=1 or dims=2."))
    end

    dropzeros!(regridder.intersections)
    return regridder
end

function Fields.compute_at!(operation::RegriddedOperation, time)
    Fields.compute_at!(operation.source, time)
    regrid!(operation.destination, operation.regridder, operation.source)
    return nothing
end

end # module
