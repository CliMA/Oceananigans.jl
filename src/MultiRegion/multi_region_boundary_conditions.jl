using Oceananigans.Architectures: arch_array
using Oceananigans.Operators: assumed_field_location
using Oceananigans.Fields: reduced_dimensions

using Oceananigans.BoundaryConditions: CBCT, CBC
import Oceananigans.Fields: fill_halo_regions_field_tuple!, extract_field_bcs

import Oceananigans.BoundaryConditions: 
            fill_west_halo!, 
            fill_east_halo!, 
            fill_halo_regions!,
            fill_west_and_east_halo!,
            regularize_field_boundary_conditions

@inline bc_str(::MultiRegionObject) = "MultiRegion Boundary Conditions"

function fill_halo_regions_field_tuple!(full_fields, grid::MultiRegionGrid, args...; kwargs...) 
    for field in full_fields
        fill_halo_regions!(field, args...; kwargs...)
    end
end

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions, mrg::MultiRegionGrid, field_name::Symbol, prognostic_field_names=nothing)
    loc = assumed_field_location(field_name)
    return FieldBoundaryConditions(mrg, loc)
end

function fill_halo_regions!(field::MultiRegionField, args...; kwargs...)
    reduced_dims = reduced_dimensions(field)

    return fill_halo_regions!(field.data,
                              field.boundary_conditions,
                              field.grid,
                              field.boundary_buffers,
                              args...;
                              reduced_dimensions = reduced_dims,
                              kwargs...)
end

@inline extract_field_bcs(field::MultiRegionField) = field.boundary_conditions

fill_halo_regions!(c::MultiRegionObject, ::Nothing, args...; kwargs...) = nothing

fill_halo_regions!(c::MultiRegionObject, bcs, mrg::MultiRegionGrid, buffers, args...; kwargs...) =
    apply_regionally!(fill_halo_regions!, c, bcs, mrg, Reference(c.regions), Reference(buffers.regions), args...; kwargs...)

fill_halo_regions!(c::NTuple, bcs, mrg::MultiRegionGrid, args...; kwargs...) = 
    apply_regionally!(fill_halo_regions!, c, bcs, mrg, Reference(Tuple(obj.regions for obj in c)), args...; kwargs...)

function fill_west_and_east_halo!(c, west_bc::CBCT, east_bc::CBCT, arch, dep, grid, args...; kwargs...) 
    fill_west_halo!(c, west_bc, arch, dep, grid, args...; kwargs...)
    fill_east_halo!(c, east_bc, arch, dep, grid, args...; kwargs...)
    return NoneEvent()
end   


function fill_west_halo!(c, bc::CBC, arch, dep, grid, neighbors, buffers, args...; kwargs...)
    H = halo_size(grid)[1]
    N = size(grid)[1]
    w = neighbors[bc.condition.from_rank]

    dst = buffers[bc.condition.rank].west
    src = buffers[bc.condition.from_rank].west

    switch_device!(getdevice(w))
    src .= (parent(w)[N+1:N+H, :, :])
    synchronize()
    
    switch_device!(getdevice(c))
    copyto!(dst, src)
    
    p  = view(parent(c), 1:H, :, :)
    p .= dst

    return nothing
end

function fill_east_halo!(c, bc::CBC, arch, dep, grid, neighbors, buffers, args...; kwargs...)
    H = halo_size(grid)[1]
    N = size(grid)[1]
    e = neighbors[bc.condition.from_rank]
    
    dst = buffers[bc.condition.rank].east
    src = buffers[bc.condition.from_rank].east

    switch_device!(getdevice(e))
    src .= (parent(e)[H+1:2H, :, :])
    synchronize()
    
    switch_device!(getdevice(c))    
    copyto!(dst, src)
    
    p  = view(parent(c), N+H+1:N+2H, :, :)
    p .= dst

    return nothing
end

function fill_west_halo!(c::NTuple, bc::NTuple{M, CBC}, arch, dep, grid, neighbors, args...; kwargs...) where M
    H = halo_size(grid)[1]
    N = size(grid)[1]

    dst = arch_array(arch, zeros(M, H, size(parent(c[1]), 2), size(parent(c[1]), 3)))

    switch_device!(getdevice(neighbors[1][bc[1].condition]))
    src = arch_array(arch, zeros(M, H, size(parent(c[1]), 2), size(parent(c[1]), 3)))
    
    for n in 1:M
        w = neighbors[n][bc[n].condition]
        src[n, :, :, :] .= parent(w)[N+1:N+H, :, :]
    end

    synchronize()
    
    switch_device!(getdevice(c[1]))
    copyto!(dst, src)
    for n in 1:M
        p  = view(parent(c[n]), 1:H, :, :)
        p .= dst[n, :, :, :]
    end

    synchronize()

    return nothing
end

function fill_east_halo!(c::NTuple, bc::NTuple{M, CBC}, arch, dep, grid, neighbors, args...; kwargs...) where M
    H = halo_size(grid)[1]
    N = size(grid)[1]

    dst = arch_array(arch, zeros(M, H, size(parent(c[1]), 2), size(parent(c[1]), 3)))

    switch_device!(getdevice(neighbors[1][bc[1].condition]))
    src = arch_array(arch, zeros(M, H, size(parent(c[1]), 2), size(parent(c[1]), 3)))
    
    for n in 1:M
        e = neighbors[n][bc[n].condition]
        src[n, :, :, :] .= parent(e)[H+1:2H, :, :]
    end

    synchronize()
    
    switch_device!(getdevice(c[1]))
    copyto!(dst, src)
    for n in 1:M
        p  = view(parent(c[n]),  N+H+1:N+2H, :, :)
        p .= dst[n, :, :, :]
    end

    synchronize()

    return nothing
end
  


# Everything goes for Connected
validate_boundary_condition_location(::MultiRegionObject, ::Center, side)       = nothing 
validate_boundary_condition_location(::MultiRegionObject, ::Face, side)         = nothing 
validate_boundary_condition_topology(::MultiRegionObject, topo::Periodic, side) = nothing
validate_boundary_condition_topology(::MultiRegionObject, topo::Flat,     side) = nothing
