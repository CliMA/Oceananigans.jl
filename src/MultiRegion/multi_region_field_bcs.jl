using KernelAbstractions: @kernel, @index, NoneEvent, Event
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Architectures: device, arch_array
using Oceananigans.Utils: launch!

using Oceananigans.BoundaryConditions: 
                AbstractBoundaryConditionClassification, 
                BoundaryCondition,  
                default_auxiliary_field_boundary_condition,
                NoFluxBoundaryCondition,
                fill_bottom_and_top_halo!,
                fill_south_and_north_halo!,
                fill_west_and_east_halo!

import Oceananigans.Fields:
            validate_boundary_condition_location

import Oceananigans.BoundaryConditions: bc_str

using Oceananigans.BoundaryConditions: PeriodicBoundaryCondition

import Oceananigans.BoundaryConditions: 
            fill_west_halo!, 
            fill_east_halo!, 
            fill_halo_regions!,
            apply_x_bcs!,
            apply_y_bcs!,
            apply_x_east_bc!,
            apply_x_west_bc!,
            apply_y_south_bc!,
            apply_y_north_bc!,
            FieldBoundaryConditions,
            validate_boundary_condition_topology

struct Connected <: AbstractBoundaryConditionClassification end

ConnectedBoundaryCondition(neighbor) = BoundaryCondition(Connected, neighbor)
const CBC  = BoundaryCondition{<:Connected}

@inline bc_str(bc::BoundaryCondition{<:Connected}) = "Connected"

function FieldBoundaryConditions(mrg::MultiRegionGrid, loc; 
                                west = default_auxiliary_field_boundary_condition(topology(mrg, 1)(), loc[1]()),
                                east = default_auxiliary_field_boundary_condition(topology(mrg, 1)(), loc[1]()),
                               south = default_auxiliary_field_boundary_condition(topology(mrg, 2)(), loc[2]()),
                               north = default_auxiliary_field_boundary_condition(topology(mrg, 2)(), loc[2]()),
                              bottom = default_auxiliary_field_boundary_condition(topology(mrg, 3)(), loc[3]()),
                                 top = default_auxiliary_field_boundary_condition(topology(mrg, 3)(), loc[3]()),
                            immersed = NoFluxBoundaryCondition())

    west  = apply_regionally(inject_west_boundary,  allregions(mrg), mrg.partition, west)
    east  = apply_regionally(inject_east_boundary,  allregions(mrg), mrg.partition, east)
    south = apply_regionally(inject_south_boundary, allregions(mrg), mrg.partition, south)
    north = apply_regionally(inject_north_boundary, allregions(mrg), mrg.partition, north)
    
    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

fill_halo_regions!(f::MultiRegionField, args...; kwargs...) = fill_halo_regions!(f.data, f.boundary_conditions, architecture(f), f.grid, args...; kwargs...)

function fill_halo_regions!(f::MultiRegionObject, bcs, arch, mrg::MultiRegionGrid, args...; kwargs...)
    # Everything in apply_regionally occurs asynchronously. Therefore, synchronize
    
    # Apply top and bottom boundary conditions as usual
    @sync apply_regionally!(fill_bottom_and_top_halo!, f, bcs.bottom, bcs.top, arch, device_event(arch), mrg, args...; kwargs...) 
    
    # Find neighbor and pass it to the fill_halo functions
    x_neighb = apply_regionally(find_neighbors, bcs.west,  bcs.east, f.regions)
    y_neighb = apply_regionally(find_neighbors, bcs.south, bcs.north, f.regions)

    # Fill x- and y-direction halos
    @sync apply_regionally!(fill_south_and_north_halo!, f, bcs.south, bcs.north, arch, device_event(arch), mrg, y_neighb, args...; kwargs...) 
    @sync apply_regionally!(fill_west_and_east_halo!  , f, bcs.west, bcs.east, arch, device_event(arch), mrg, x_neighb, args...; kwargs...) 
    
    return nothing
end

find_neighbors(left, right, regions) = (find_neighbor(left, regions), find_neighbor(right, regions))
find_neighbor(bc, regions)           = nothing
find_neighbor(bc::CBC, regions)      = regions[bc.condition]

function fill_west_halo!(c, bc::CBC, arch, dep, grid, neighb, args...; kwargs...)
    H = halo_size(grid)[1]
    N = size(grid)[1]
    w = neighb[1]

    switch_device!(getdevice(w))
    src = deepcopy(parent(w)[N+1:N+H, :, :])

    switch_device!(getdevice(c))
    dst = arch_array(arch, zeros(length(H), size(parent(c), 2), size(parent(c), 3)))
    copyto!(dst, src)

    p  = view(parent(c), 1:H, :, :)
    p .= dst

    return NoneEvent()
end

function fill_east_halo!(c, bc::CBC, arch, dep, grid, neighb, args...; kwargs...)
    H = halo_size(grid)[1]
    N = size(grid)[1]
    e = neighb[2]

    switch_device!(getdevice(e))
    src = deepcopy(parent(e)[H+1:2H, :, :])

    switch_device!(getdevice(c))
    dst = arch_array(arch, zeros(length(H), size(parent(c), 2), size(parent(c), 3)))
    copyto!(dst, src)

    p  = view(parent(c), N+H+1:N+2H, :, :)
    p .= dst

    return NoneEvent()
end
  
# Everything goes for Connected
validate_boundary_condition_location(::MultiRegionObject, ::Center, side)       = nothing 
validate_boundary_condition_location(::MultiRegionObject, ::Face, side)         = nothing 
validate_boundary_condition_topology(::MultiRegionObject, topo::Periodic, side) = nothing
validate_boundary_condition_topology(::MultiRegionObject, topo::Flat,     side) = nothing

# Don't "apply fluxes" across Connected boundaries
@inline apply_x_east_bc!(  Gc, loc, ::CBC, args...) = nothing
@inline apply_x_west_bc!(  Gc, loc, ::CBC, args...) = nothing
@inline apply_y_north_bc!( Gc, loc, ::CBC, args...) = nothing
@inline apply_y_south_bc!( Gc, loc, ::CBC, args...) = nothing

apply_x_bcs!(Gc, ::AbstractGrid, c, ::CBC, ::CBC, ::AbstractArchitecture, args...) = NoneEvent()
apply_y_bcs!(Gc, ::AbstractGrid, c, ::CBC, ::CBC, ::AbstractArchitecture, args...) = NoneEvent()
