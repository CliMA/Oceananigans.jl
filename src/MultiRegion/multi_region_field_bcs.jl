using Oceananigans.BoundaryConditions: 
                AbstractBoundaryConditionClassification, 
                BoundaryCondition,  
                FieldBoundaryConditions

import Oceananigans.Fields:
            validate_boundary_condition_location

import Oceananigans.BoundaryConditions: bc_str

import Oceananigans.BoundaryConditions: 
                                fill_west_halo!, 
                                fill_east_halo!, 
                                fill_north_halo!, 
                                fill_south_halo!,
                                fill_bottom_halo!,
                                fill_top_halo!,
                                fill_halo_regions!,
                                validate_boundary_condition_topology

struct Connected <: AbstractBoundaryConditionClassification end

ConnectedBoundaryCondition(neighbour) = BoundaryCondition(Connected, neighbour)
const CBC  = BoundaryCondition{<:Connected}

@inline bc_str(bc::BoundaryCondition{<:Connected}) = "Connected"

function inject_regional_bcs(region, p::XPartition, bcs)
    
    if region == 1
        east = ConnectedBoundaryCondition(2)
        west = ConnectedBoundaryCondition(length(p))
    elseif region == length(p)
        west = ConnectedBoundaryCondition(length(p)-1)
        east = ConnectedBoundaryCondition(1)
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

# Everything goes for Connected
validate_boundary_condition_location(::CBC, ::Union{Center, Face}, side) = nothing 
validate_boundary_condition_topology(::CBC, topo::Periodic, side)  = nothing
validate_boundary_condition_topology(::CBC, topo::Flat,     side)  = nothing


# fill_halo_regions!(f::MultiRegionField, args..., kwrags...) = apply_regionally!(fill_halo_regions!, f, args...; kwargs...)
# fill_halo_regions!(f::MultiRegionObject, args..., kwrags...) = apply_regionally!(fill_halo_regions!, f, args...; kwargs...)


# function fill_west_halo!(c, ::CBC, arch, dep, grid, args...; kw...)
  
#   return event
# end

# function fill_east_halo!(c, ::CBC, arch, dep, grid, args...; kw...)
#   c_parent = parent(c)
#   yz_size = size(c_parent)[[2, 3]]
#   event = launch!(arch, grid, yz_size, fill_periodic_east_halo!, c_parent, grid.Hx, grid.Nx; dependencies=dep, kw...)
#   return event
# endß
