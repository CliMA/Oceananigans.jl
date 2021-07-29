using Oceananigans.BoundaryConditions: AbstractBoundaryConditionClassification 

import Oceananigans.BoundaryConditions: getbc, regularize_boundary_condition

abstract type AbstractImmersedBCClassification <: AbstractBoundaryConditionClassification end

struct ImmersedFlux <: AbstractImmersedBCCClassification end

ImmersedFluxBoundaryCondition(val; kwargs...) = BoundaryCondition(ImmersedFlux, val; kwargs...)

const IBC = BoundaryCondition{<:AbstractImmersedBCClassification}
const IFBC = BoundaryCondition{<:ImmersedFlux}

@inline getbc(ibc::IFBC{<:Number}, i, j, k, grid, clock, model_fields) = ibc.condition
@inline getbc(ibc::IFBC{<:Array}, i, j, k, grid, clock, model_fields) = @inbounds ibc.condition[i, j, k]
@inline getbc(ibc::IFBC, i, j, k, grid, clock, model_fields, args...) = ibc.condition(i, j, k, grid, clock, model_fields)

function regularize_boundary_condition(bc::IBC, topo, loc, dim, I, prognostic_field_names)

    boundary_func = bc.condition
    LX, LY, LZ = loc

    indices, interps = index_and_interp_dependencies(LX, LY, LZ,
                                                     boundary_func.field_dependencies,
                                                     prognostic_field_names)

    regularized_boundary_func = ContinuousBoundaryFunction{LX, LY, LZ, Nothing}(boundary_func.func,
                                                                                boundary_func.parameters,
                                                                                boundary_func.field_dependencies,
                                                                                indices, interps)

    return BoundaryCondition(bc.classification, regularized_boundary_func)
end


