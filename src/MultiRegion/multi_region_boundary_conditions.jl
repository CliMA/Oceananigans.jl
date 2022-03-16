using Oceananigans.Architectures: arch_array
using Oceananigans.Operators: assumed_field_location
using Oceananigans.Fields: reduced_dimensions

using Oceananigans.BoundaryConditions: CBCT, CBC
import Oceananigans.Fields: fill_halo_regions_field_tuple!, extract_field_bcs, extract_field_data

import Oceananigans.BoundaryConditions: 
            fill_west_halo!, 
            fill_east_halo!, 
            fill_halo_regions!,
            fill_west_and_east_halo!

@inline bc_str(::MultiRegionObject) = "MultiRegion Boundary Conditions"

@inline extract_field_buffers(field::Field)        = field.boundary_buffers
@inline extract_field_bcs(field::MultiRegionField) = field.boundary_conditions

@inline function fill_halo_regions_field_tuple!(full_fields, grid::MultiRegionGrid, args...; kwargs...) 
    for field in full_fields
        fill_halo_regions!(field, args...; kwargs...)
    end
end

# @inline fill_halo_regions_field_tuple!(full_fields, grid::MultiRegionGrid, args...; kwargs...) =
#     fill_halo_regions!(extract_field_data.(full_fields), extract_field_bcs.(full_fields), grid, args...; kwargs...)

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

fill_halo_regions!(c::MultiRegionObject, ::Nothing, args...; kwargs...) = nothing

fill_halo_regions!(c::MultiRegionObject, bcs, mrg::MultiRegionGrid, buffers, args...; kwargs...) =
    apply_regionally!(fill_halo_regions!, c, bcs, mrg, Reference(c.regions), Reference(buffers.regions), args...; kwargs...)

fill_halo_regions!(c::NTuple, bcs, mrg::MultiRegionGrid, buffers, args...; kwargs...) = 
    apply_regionally!(fill_halo_regions!, c, bcs, mrg, Reference(Tuple(obj.regions for obj in c)), Reference(Tuple(obj.buffers for obj in c)), args...; kwargs...)

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
    src .= (parent(w)[N+1:N+H, :, :])[:]
    synchronize()
    
    switch_device!(getdevice(c))
    copyto!(dst, src)
    
    p  = view(parent(c), 1:H, :, :)
    p .= reshape(dst, size(p))

    return nothing
end

function fill_east_halo!(c, bc::CBC, arch, dep, grid, neighbors, buffers, args...; kwargs...)
    H = halo_size(grid)[1]
    N = size(grid)[1]
    e = neighbors[bc.condition.from_rank]
    
    dst = buffers[bc.condition.rank].east
    src = buffers[bc.condition.from_rank].east

    switch_device!(getdevice(e))
    src .= (parent(e)[H+1:2H, :, :])[:]
    synchronize()
    
    switch_device!(getdevice(c))    
    copyto!(dst, src)
    
    p  = view(parent(c), N+H+1:N+2H, :, :)
    p .= reshape(dst, size(p))

    return nothing
end

function fill_west_halo!(c::NTuple, bc::NTuple{M, CBC}, arch, dep, grid, neighbors, buffers, args...; kwargs...) where M
    H = halo_size(grid)[1]
    N = size(grid)[1]

    dst = []
    src = []
    for n in M
        push!(dst, buffers[n][bc[n].condition.rank].west...)
        push!(src, buffers[n][bc[n].condition.from_rank].west...)
    end
    
    switch_device!(getdevice(neighbors[1][bc[1].condition.from_rank]))
    
    @sync for n in 1:M
        @async begin
            w = neighbors[n][bc[n].condition.from_rank]
            src[n] .= parent(w)[N+1:N+H, :, :]
        end
    end
    synchronize()
    
    switch_device!(getdevice(c[1]))
    copyto!(dst, src)

    @sync for n in 1:M
        @async begin
            p  = view(parent(c[n]), 1:H, :, :)
            p .= dst[n]
        end
    end

    synchronize()

    return nothing
end

function fill_east_halo!(c::NTuple, bc::NTuple{M, CBC}, arch, dep, grid, neighbors, args...; kwargs...) where M
    H = halo_size(grid)[1]
    N = size(grid)[1]

    dst = arch_array(arch, zeros(M, H, size(parent(c[1]), 2), size(parent(c[1]), 3)))

    switch_device!(getdevice(neighbors[1][bc[1].condition.from_rank]))
    src = arch_array(arch, zeros(M, H, size(parent(c[1]), 2), size(parent(c[1]), 3)))
    
    for n in 1:M
        e = neighbors[n][bc[n].condition.from_rank]
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
