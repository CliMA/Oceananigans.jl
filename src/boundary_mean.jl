using Adapt
using Oceananigans: location
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

@inline boundary_adjacent_index(::Val{:east}, grid, loc) = (size(grid, 1), 1, 1), (2, 3)
@inline boundary_adjacent_index(side::Val{:west}, grid, loc) = (first_interior_index(side, loc, grid), 1, 1), (2, 3)

@inline boundary_adjacent_index(::Val{:north}, grid, loc) = (1, size(grid, 2), 1), (1, 3)
@inline boundary_adjacent_index(side::Val{:south}, grid, loc) = (1, first_interior_index(side, loc, grid), 1), (1, 3)

@inline boundary_adjacent_index(::Val{:top}, grid, loc)    = (1, 1, size(grid, 3)), (2, 3)
@inline boundary_adjacent_index(side::Val{:bottom}, grid, loc) = (1, 1, first_interior_index(side, loc, grid)), (2, 3)

@inline first_interior_index(::Union{Val{:west}, Val{:east}}, ::Tuple{<:Center, <:Any, <:Any}, grid) = 1
@inline first_interior_index(::Union{Val{:west}, Val{:east}}, ::Tuple{Face, <:Any, <:Any}, grid) = 2

@inline first_interior_index(::Union{Val{:south}, Val{:north}}, ::Tuple{<:Any, <:Center, <:Any}, grid) = 1
@inline first_interior_index(::Union{Val{:south}, Val{:north}}, ::Tuple{<:Any, Face, <:Any}, grid) = 2

@inline first_interior_index(::Union{Val{:bottom}, Val{:top}}, ::Tuple{<:Any, <:Any, <:Center}, grid) = 1
@inline first_interior_index(::Union{Val{:bottom}, Val{:top}}, ::Tuple{<:Any, <:Any, Face}, grid) = 2

function update_boundary_condition!(bc::MOPABC, val_side, u, model)
    grid = model.grid
    loc = location(u)

    An = boundary_normal_area(val_side, grid)

    (i, j, k), dims = boundary_adjacent_index(val_side, grid, loc)
    
    total_area = CUDA.@allowscalar sum(An; dims)[i, j, k]

    Ū = sum(u * An; dims) / total_area

    bc.condition.value[] = CUDA.@allowscalar Ū[i, j, k]

    return nothing
end