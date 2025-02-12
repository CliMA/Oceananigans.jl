using Adapt
using Oceananigans: instantiated_location
using Oceananigans.Fields: Center, Face
using Oceananigans.AbstractOperations: GridMetricOperation, Ax, Ay, Az
using Oceananigans.BoundaryConditions: BoundaryCondition, Open, PerturbationAdvection

import Adapt: adapt_structure
import Oceananigans.BoundaryConditions: update_boundary_condition!

struct BoundaryAdjacentMean{V}
   value :: V

   BoundaryAdjacentMean(; value::BV = Ref(0.0)) where BV = new{BV}(value)
end

@inline (bam::BoundaryAdjacentMean)(args...) = bam.value[]

Adapt.adapt_structure(to, mo::BoundaryAdjacentMean) = 
    BoundaryAdjacentMean(; value = adapt(to, mo.value[]))

const MOPABC = BoundaryCondition{<:Open{<:PerturbationAdvection}, <:BoundaryAdjacentMean}

@inline boundary_normal_area(::Union{Val{:west}, Val{:east}}, grid)   = GridMetricOperation((Face, Center, Center), Ax, grid)
@inline boundary_normal_area(::Union{Val{:south}, Val{:north}}, grid) = GridMetricOperation((Center, Face, Center), Ay, grid)
@inline boundary_normal_area(::Union{Val{:bottom}, Val{:top}}, grid)  = GridMetricOperation((Center, Center, Face), Az, grid)

@inline boundary_adjacent_indices(::Val{:east}, grid, loc) = (size(grid, 1), 1, 1), (2, 3)
@inline boundary_adjacent_indices(val_side::Val{:west}, grid, loc) = (first_interior_index(val_side, loc), 1, 1), (2, 3)

@inline boundary_adjacent_indices(::Val{:north}, grid, loc) = (1, size(grid, 2), 1), (1, 3)
@inline boundary_adjacent_indices(val_side::Val{:south}, grid, loc) = (1, first_interior_index(val_side, loc), 1), (1, 3)

@inline boundary_adjacent_indices(::Val{:top}, grid, loc) = (1, 1, size(grid, 3)), (2, 3)
@inline boundary_adjacent_indices(val_side::Val{:bottom}, grid, loc) = (1, 1, first_interior_index(val_side, loc)), (2, 3)

@inline first_interior_index(::Union{Val{:west}, Val{:east}}, ::Tuple{Center, <:Any, <:Any}) = 1
@inline first_interior_index(::Union{Val{:west}, Val{:east}}, ::Tuple{Face, <:Any, <:Any}) = 2

@inline first_interior_index(::Union{Val{:south}, Val{:north}}, ::Tuple{<:Any, Center, <:Any}) = 1
@inline first_interior_index(::Union{Val{:south}, Val{:north}}, ::Tuple{<:Any, Face, <:Any}) = 2

@inline first_interior_index(::Union{Val{:bottom}, Val{:top}}, ::Tuple{<:Any, <:Any, Center}) = 1
@inline first_interior_index(::Union{Val{:bottom}, Val{:top}}, ::Tuple{<:Any, <:Any, Face}) = 2

function update_boundary_condition!(bc::MOPABC, val_side, u, model)
    grid = model.grid
    loc = instantiated_location(u)

    An = boundary_normal_area(val_side, grid)
    
    (iB, jB, kB), dims = boundary_adjacent_indices(val_side, grid, loc)

    total_area = CUDA.@allowscalar sum(An; dims)[iB, jB, kB]

    Ū = sum(u * An; dims) / total_area

    bc.condition.value[] = CUDA.@allowscalar Ū[iB, jB, kB]
    return nothing
end