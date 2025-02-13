using Adapt, CUDA
using Oceananigans: instantiated_location
using Oceananigans.Fields: Center, Face
using Oceananigans.AbstractOperations: GridMetricOperation, Ax, Ay, Az
using Oceananigans.BoundaryConditions: BoundaryCondition, Open, PerturbationAdvection

import Adapt: adapt_structure
import Oceananigans.BoundaryConditions: update_boundary_condition!

struct BoundaryAdjacentMean{FF, BV}
    flux_field :: FF
         value :: BV

   BoundaryAdjacentMean(grid, side; 
                        flux_field::FF = boundary_reduced_field(Val(side), grid),
                        value::BV = Ref(zero(grid))) where {FF, BV} = new{FF, BV}(value)
end

@inline (bam::BoundaryAdjacentMean)(args...) = bam.value[]

Adapt.adapt_structure(to, mo::BoundaryAdjacentMean) = 
    BoundaryAdjacentMean(nothing, adapt(to, mo.value[]))

@inline boundary_reduced_field(::Union{Val{:west}, Val{:east}}, grid)   = Field{Center, Nothing, Nothing}(grid)
@inline boundary_reduced_field(::Union{Val{:south}, Val{:north}}, grid) = Field{Nothing, Center, Nothing}(grid)
@inline boundary_reduced_field(::Union{Val{:bottom}, Val{:top}}, grid)  = Field{Nothing, Nothing, Center}(grid)

@inline boundary_normal_area(::Union{Val{:west}, Val{:east}}, grid)   = GridMetricOperation((Face, Center, Center), Ax, grid)
@inline boundary_normal_area(::Union{Val{:south}, Val{:north}}, grid) = GridMetricOperation((Center, Face, Center), Ay, grid)
@inline boundary_normal_area(::Union{Val{:bottom}, Val{:top}}, grid)  = GridMetricOperation((Center, Center, Face), Az, grid)

@inline boundary_adjacent_indices(::Val{:east}, grid, loc) = size(grid, 1), 1, 1
@inline boundary_adjacent_indices(val_side::Val{:west}, grid, loc) = first_interior_index(val_side, loc), 1, 1

@inline boundary_adjacent_indices(::Val{:north}, grid, loc) = 1, size(grid, 2), 1
@inline boundary_adjacent_indices(val_side::Val{:south}, grid, loc) = 1, first_interior_index(val_side, loc), 1

@inline boundary_adjacent_indices(::Val{:top}, grid, loc) = 1, 1, size(grid, 3)
@inline boundary_adjacent_indices(val_side::Val{:bottom}, grid, loc) = 1, 1, first_interior_index(val_side, loc)

@inline first_interior_index(::Union{Val{:west}, Val{:east}}, ::Tuple{Center, <:Any, <:Any}) = 1
@inline first_interior_index(::Union{Val{:west}, Val{:east}}, ::Tuple{Face, <:Any, <:Any}) = 2

@inline first_interior_index(::Union{Val{:south}, Val{:north}}, ::Tuple{<:Any, Center, <:Any}) = 1
@inline first_interior_index(::Union{Val{:south}, Val{:north}}, ::Tuple{<:Any, Face, <:Any}) = 2

@inline first_interior_index(::Union{Val{:bottom}, Val{:top}}, ::Tuple{<:Any, <:Any, Center}) = 1
@inline first_interior_index(::Union{Val{:bottom}, Val{:top}}, ::Tuple{<:Any, <:Any, Face}) = 2

function compute_boundary_mean!(bam::BoundaryAdjacentMean, val_side, u)
    grid = u.grid

    loc = instantiated_location(u)
    iB, jB, kB = boundary_adjacent_indices(val_side, grid, loc)
    An = boundary_normal_area(val_side, grid)

    # get the total flux
    sum!(bam.flux_field, u * An)

    bam.value[] = CUDA.@allowscalar bam.flux_field[iB, jB, kB]

    # get the normalizing area
    sum!(bam.flux_field, An)

    bam.value[] ./= CUDA.@allowscalar bam.flux_field[iB, jB, kB]

    return nothing
end

# let this get updated in boundary conditions

const MOOBC = BoundaryCondition{<:Open, <:BoundaryAdjacentMean}

@inline update_boundary_condition!(bc::MOOBC, val_side, u, model) =
    compute_boundary_mean!(bc.condition, val_side, u)
