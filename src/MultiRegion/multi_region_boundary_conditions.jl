using Oceananigans.Architectures: arch_array
using Oceananigans.Operators: assumed_field_location

using Oceananigans.BoundaryConditions: CBC

import Oceananigans.BoundaryConditions: 
            fill_west_halo!, 
            fill_east_halo!, 
            fill_halo_regions!,
            regularize_field_boundary_conditions

@inline bc_str(::MultiRegionObject) = "MultiRegion Boundary Conditions"

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions, mrg::MultiRegionGrid, field_name::Symbol, prognostic_field_names=nothing)
    loc = assumed_field_location(field_name)
    return FieldBoundaryConditions(mrg, loc)
end

fill_halo_regions!(c::MultiRegionObject, ::Nothing, args...; kwargs...) = nothing

fill_halo_regions!(c::MultiRegionObject, bcs, arch, mrg::MultiRegionGrid, args...; kwargs...) =
    apply_regionally!(fill_halo_regions!, c, bcs, arch, mrg, Reference(c.regions), args...; kwargs...)

function fill_west_halo!(c, bc::CBC, arch, dep, grid, neighb, args...; kwargs...)
    H = halo_size(grid)[1]
    N = size(grid)[1]
    w = neighb[bc.condition]

    switch_device!(getdevice(w))
    src = deepcopy(parent(w)[N+1:N+H, :, :])

    switch_device!(getdevice(c))
    dst = arch_array(arch, zeros(length(H), size(parent(c), 2), size(parent(c), 3)))

    copyto!(dst, src)
    synchronize()

    p  = view(parent(c), 1:H, :, :)
    p .= dst

    return nothing
end

function fill_east_halo!(c, bc::CBC, arch, dep, grid, neighb, args...; kwargs...)
    H = halo_size(grid)[1]
    N = size(grid)[1]
    e = neighb[bc.condition]


    switch_device!(getdevice(e))
    src = deepcopy(parent(e)[H+1:2H, :, :])

    switch_device!(getdevice(c))
    dst = arch_array(arch, zeros(length(H), size(parent(c), 2), size(parent(c), 3)))
    
    copyto!(dst, src)
    synchronize()
    
    p  = view(parent(c), N+H+1:N+2H, :, :)
    p .= dst

    return nothing
end
  
# Everything goes for Connected
validate_boundary_condition_location(::MultiRegionObject, ::Center, side)       = nothing 
validate_boundary_condition_location(::MultiRegionObject, ::Face, side)         = nothing 
validate_boundary_condition_topology(::MultiRegionObject, topo::Periodic, side) = nothing
validate_boundary_condition_topology(::MultiRegionObject, topo::Flat,     side) = nothing
