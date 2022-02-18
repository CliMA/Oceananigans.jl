using Oceananigans.BoundaryConditions: AbstractBoundaryConditionClassification, BoundaryCondition
import Oceananigans.BoundaryConditions: bc_str

struct Connected <: AbstractBoundaryConditionClassification end

ConnectedBoundaryCondition(neighbour) = BoundaryCondition(Connected, neighbour)

@inline bc_str(bc::BoundaryCondition{<:Connected}) = "Connected"

function inject_regional_bcs(region, p::XPartition; bcs)
    
    if region == 1
        east = ConnectedBoundaryCondition(2)
        west = bcs.west
    elseif region == length(p)
        west = ConnectedBoundaryCondition(length(p)-1)
        east = bcs.east
    else
        west = ConnectedBoundaryCondition(region + 1)
        east = ConnectedBoundaryCondition(region - 1)
    end

    return FieldBoundaryConditions(west = west, 
                                   east = east, 
                                   south = bcs.south,
                                   north = bcs.north, 
                                   top = bcs.top, 
                                   bottom = bcs.bottom,
                                   immersed = bcs.immersed)
end