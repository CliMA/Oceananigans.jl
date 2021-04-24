#####
##### Support for Fields on the MultiRegion
#####

using Statistics
using OffsetArrays: OffsetArray
using Oceananigans.Fields: AbstractField, AbstractDataField, AbstractReducedField, Field

import Oceananigans.Fields: validate_field_data, interior
import Oceananigans.Models.HydrostaticFreeSurfaceModels: validate_vertical_velocity_boundary_conditions
import Oceananigans.BoundaryConditions: FieldBoundaryConditions

"""
    struct MultiRegionTuple{E, F}

Contains a tuple of `regions::F` with elements of type `E`.
"""
struct MultiRegionTuple{E, F}
    regions :: F

    function MultiRegionTuple(regions::F) where F <: Tuple
        E = typeof(regions[1])
        return new{E, F}(regions)
    end
end

@inline Base.getindex(t::MultiRegionTuple, i::Int) = @inbounds t.regions[i]

#####
##### Dispatch the world
#####

# Flavors of MultiRegionField
const MultiRegionField                = Field{               X, Y, Z, A, D, <:MultiRegionGrid} where {X, Y, Z, A, D}
const MultiRegionAbstractField        = AbstractField{       X, Y, Z, A,    <:MultiRegionGrid} where {X, Y, Z, A}
const MultiRegionAbstractDataField    = AbstractDataField{   X, Y, Z, A,    <:MultiRegionGrid} where {X, Y, Z, A}
const MultiRegionAbstractReducedField = AbstractReducedField{X, Y, Z, A, D, <:MultiRegionGrid} where {X, Y, Z, A, D}

const AbstractMultiRegionField = Union{MultiRegionField,
                                       MultiRegionAbstractField,
                                       MultiRegionAbstractDataField,
                                       MultiRegionAbstractReducedField}

function FieldBoundaryConditions(multi_region_grid::MultiRegionGrid, loc; user_defined_bcs...)

    regions = Tuple(
        inject_cubed_sphere_exchange_boundary_conditions(
            FieldBoundaryConditions(regional_grid, loc; user_defined_bcs...),
            region_index,
            multi_region_grid.connectivity
        )
        for (region_index, regional_grid) in enumerate(multi_region_grid.regions)
    )

    return MultiRegionTuple(regions)
end

#####
##### Utils
#####

@inline function interior(field::AbstractMultiRegionField)
    regions = Tuple(interior(regional_field) for regional_field in regions(field))
    return MultiRegionTuple(regions)
end

#####
##### MultiRegion reductions
#####

Base.minimum(field::AbstractMultiRegionField; dims=:)    = minimum(minimum(regional_field; dims=dims) for regional_field in regions(field))
Base.maximum(field::AbstractMultiRegionField; dims=:)    = maximum(maximum(regional_field; dims=dims) for regional_field in regions(field))
Statistics.mean(field::AbstractMultiRegionField; dims=:) = mean(mean(regional_field; dims=dims)       for regional_field in regions(field))

Base.minimum(f, field::AbstractMultiRegionField; dims=:)    = minimum(minimum(f, regional_field; dims=dims) for regional_field in regions(field))
Base.maximum(f, field::AbstractMultiRegionField; dims=:)    = maximum(maximum(f, regional_field; dims=dims) for regional_field in regions(field))
Statistics.mean(f, field::AbstractMultiRegionField; dims=:) = mean(mean(f, regional_field; dims=dims)       for regional_field in regions(field))

#####
##### Validating cubed sphere stuff
#####

function validate_field_data(X, Y, Z, data, grid::MultiRegionGrid)

    for (regional_data, regional_grid) in zip(data.regions, grid.regions)
        validate_field_data(X, Y, Z, regional_data, regional_grid)
    end

    return nothing
end

validate_vertical_velocity_boundary_conditions(w::AbstractMultiRegionField) =
    [validate_vertical_velocity_boundary_conditions(regional_w) for regional_w in regions(w)]
