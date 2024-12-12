using Adapt
using Oceananigans.AbstractOperations: GridMetricOperation, Ax, Ay, Az
using Oceananigans.BoundaryConditions: BoundaryCondition, Open, PerturbationAdvection

import Adapt: adapt_structure
import Oceananigans.BoundaryConditions: update_boundary_condition!

struct BoundaryNormalMeanVelocity{BV}
   boundary_velocity :: BV

   BoundaryNormalMeanVelocity(; value::BV = Ref(0.0)) where BV = new{BV}(value)
end

@inline (value::BoundaryNormalMeanVelocity)(args...) = value.boundary_velocity[]

Adapt.adapt_structure(to, mo::BoundaryNormalMeanVelocity) = 
    BoundaryNormalMeanVelocity(adapt(to, mo.mean_outflow_velocity[]))

const MOPABC = BoundaryCondition{<:Open{<:PerturbationAdvection}, <:BoundaryNormalMeanVelocity}

@inline boundary_normal_area(::Union{Val{:west}, Val{:east}}, grid)   = GridMetricOperation((Face, Center, Center), Ax, grid)
@inline boundary_normal_area(::Union{Val{:south}, Val{:north}}, grid) = GridMetricOperation((Center, Face, Center), Ay, grid)
@inline boundary_normal_area(::Union{Val{:bottom}, Val{:top}}, grid)  = GridMetricOperation((Center, Center, Face), Az, grid)

@inline boundary_adjacent_index(::Val{:east}, grid) = (size(grid, 1), 1, 1), (2, 3)
@inline boundary_adjacent_index(::Val{:west}, grid) = (2, 1, 1), (2, 3)

@inline boundary_adjacent_index(::Val{:north}, grid) = (1, size(grid, 2), 1), (1, 3)
@inline boundary_adjacent_index(::Val{:south}, grid) = (1, 2, 1), (1, 3)

@inline boundary_adjacent_index(::Val{:top}, grid)    = (1, 1, size(grid, 3)), (2, 3)
@inline boundary_adjacent_index(::Val{:bottom}, grid) = (1, 1, 2), (2, 3)

function update_boundary_condition!(bc::MOPABC, val_side, u, model)
    grid = model.grid

    An = boundary_normal_area(val_side, grid)

    (i, j, k), dims = boundary_adjacent_index(val_side, grid)
    
    total_area = sum(An; dims)[i, j, k]

    Ū = sum(u * An; dims) / total_area

    bc.condition.boundary_velocity[] = Ū[i, j, k]
end