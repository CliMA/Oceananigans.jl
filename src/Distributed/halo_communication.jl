using KernelAbstractions: @kernel, @index, priority!
using OffsetArrays: OffsetArray
using CUDA: synchronize
using CUDA: cuStreamGetFlags, stream, priority_range, CUstream_flags_enum, CuStream, stream!

import Oceananigans.Utils: sync_device!
using Oceananigans.Fields: fill_west_and_east_send_buffers!, 
                           fill_south_and_north_send_buffers!, 
                           fill_west_send_buffers!,
                           fill_east_send_buffers!,
                           fill_south_send_buffers!,
                           fill_north_send_buffers!,
                           fill_southwest_send_buffers!,
                           fill_southeast_send_buffers!,
                           fill_northwest_send_buffers!,
                           fill_northeast_send_buffers!,
                           recv_from_buffers!, 
                           reduced_dimensions, 
                           instantiated_location

import Oceananigans.Fields: tupled_fill_halo_regions!

using Oceananigans.BoundaryConditions:
    fill_halo_size,
    fill_halo_offset,
    permute_boundary_conditions,
    PBCT, DCBCT, DCBC

import Oceananigans.BoundaryConditions: 
    fill_halo_regions!, fill_first, fill_halo_event!,
    fill_west_halo!, fill_east_halo!, fill_south_halo!,
    fill_north_halo!, fill_bottom_halo!, fill_top_halo!,
    fill_west_and_east_halo!, 
    fill_south_and_north_halo!,
    fill_bottom_and_top_halo!

@inline sync_device!(::CPU)                 = nothing
@inline sync_device!(::GPU)                 = synchronize()
@inline sync_device!(arch::DistributedArch) = sync_device!(arch.child_architecture)

#####
##### MPI tags for halo communication BCs
#####

sides  = (:west, :east, :south, :north, :southwest, :southeast, :northwest, :northeast)
side_id = Dict(side => n-1 for (n, side) in enumerate(sides))

opposite_side = Dict(
    :west => :east, 
    :east => :west,
    :south => :north,
    :north => :south,
    :southwest => :northeast, 
    :southeast => :northwest, 
    :northwest => :southeast, 
    :northeast => :southwest, 
)

# Define functions that return unique send and recv MPI tags for each side.
# It's an integer where
#   digit 1-2: an identifier for the field that is reset each timestep
#   digit 3: an identifier for the field's location 
#   digit 4: the side
#   digits 5-6: the "from" rank
#   digits 7-8: the "to" rank

RANK_DIGITS = 2
ID_DIGITS   = 1

# REMEMBER!!! This won't work for tracers!!! (It assumes you are passing maximum 4 at a time)
@inline loc_id(::Nothing, tag) = tag + 5
@inline loc_id(::Face,    tag) = tag
@inline loc_id(::Center,  tag) = tag
@inline location_id(X, Y, Z, tag) = loc_id(Z, tag)

for side in sides
    side_str = string(side)
    send_tag_fn_name = Symbol("$(side)_send_tag")
    recv_tag_fn_name = Symbol("$(side)_recv_tag")
    @eval begin
        # REMEMBER, we need to reset the tag not more than once every four passes!!
        function $send_tag_fn_name(arch, location, local_rank, rank_to_send_to)
            side_digit  = side_id[Symbol($side_str)]
            field_id    = string(location_id(location..., arch.mpi_tag[1]) + side_digit, pad=ID_DIGITS)
            from_digits = string(local_rank, pad=RANK_DIGITS)
            to_digits   = string(rank_to_send_to, pad=RANK_DIGITS)
            return parse(Int, field_id * from_digits * to_digits)
        end

        function $recv_tag_fn_name(arch, location, local_rank, rank_to_recv_from)
            side_digit  = side_id[opposite_side[Symbol($side_str)]]
            field_id    = string(location_id(location..., arch.mpi_tag[1]) + side_digit, pad=ID_DIGITS)
            from_digits = string(rank_to_recv_from, pad=RANK_DIGITS)
            to_digits   = string(local_rank, pad=RANK_DIGITS)
            return parse(Int, field_id * from_digits * to_digits)
        end
    end
end

#####
##### Filling halos for halo communication boundary conditions
#####

function tupled_fill_halo_regions!(full_fields, grid::DistributedGrid, args...; kwargs...) 
    for field in full_fields
        fill_halo_regions!(field, args...; kwargs...)
    end
end

function fill_halo_regions!(field::DistributedField, args...; kwargs...)
    reduced_dims = reduced_dimensions(field)

    return fill_halo_regions!(field.data,
                              field.boundary_conditions,
                              field.indices,
                              instantiated_location(field),
                              field.grid,
                              field.boundary_buffers,
                              args...;
                              reduced_dimensions = reduced_dims,
                              kwargs...)
end

function fill_halo_regions!(c::OffsetArray, bcs, indices, loc, grid::DistributedGrid, buffers, args...; kwargs...)
    arch       = architecture(grid)
    halo_tuple = permute_boundary_conditions(bcs)
    
    for task = 1:3
        fill_halo_event!(task, halo_tuple, c, indices, loc, arch, grid, buffers, args...; kwargs...)
    end
    
    fill_corners!(arch.connectivity, c, indices, loc, arch, grid, buffers, args...; kwargs...)
    arch.mpi_tag[1] += 1

    return nothing
end

for (side, dir) in zip([:southwest, :southeast, :northwest, :northeast], [1, 2, 3, 3])
    fill_corner_halo! = Symbol("fill_$(side)_halo!")
    send_side_halo  = Symbol("send_$(side)_halo")
    recv_and_fill_side_halo! = Symbol("recv_and_fill_$(side)_halo!")
    fill_side_send_buffers! = Symbol("fill_$(side)_send_buffers!")    

    @eval begin
        $fill_corner_halo!(::Nothing, args...; kwargs...) = nothing

        function $fill_corner_halo!(corner, c, indices, loc, arch, grid, buffers, args...; kwargs...) 
            child_arch = child_architecture(arch)
            local_rank = arch.local_rank

            recv_req = $recv_and_fill_side_halo!(c, grid, arch, loc[$dir], loc, local_rank, corner, buffers)
            $fill_side_send_buffers!(c, buffers, grid)
            sync_device!(child_arch)

            send_req = $send_side_halo(c, grid, arch, loc[$dir], loc, local_rank, corner, buffers)
            
            return [send_req, recv_req]
        end
    end
end

# If more than one direction is communicating we need to add a corner passing routine!
function fill_corners!(connectivity, c, indices, loc, arch, grid, buffers, args...; blocking = true, kwargs...)
    
    requests = MPI.Request[]

    reqsw = fill_southwest_halo!(connectivity.southwest, c, indices, loc, arch, grid, buffers, args...; kwargs...)
    reqse = fill_southeast_halo!(connectivity.southeast, c, indices, loc, arch, grid, buffers, args...; kwargs...)
    reqnw = fill_northwest_halo!(connectivity.northwest, c, indices, loc, arch, grid, buffers, args...; kwargs...)
    reqne = fill_northeast_halo!(connectivity.northeast, c, indices, loc, arch, grid, buffers, args...; kwargs...)

    !isnothing(reqsw) && push!(requests, reqsw...)
    !isnothing(reqse) && push!(requests, reqse...)
    !isnothing(reqnw) && push!(requests, reqnw...)
    !isnothing(reqne) && push!(requests, reqne...)

    if isempty(requests)
        return nothing
    end

    if !blocking && !(arch isa BlockingDistributedArch)
        push!(arch.mpi_requests, requests...)
        return nothing
    end

    # Syncronous MPI fill_halo_event!
    cooperative_waitall!(requests)

    # Reset MPI tag
    arch.mpi_tag[1] -= arch.mpi_tag[1]
    recv_from_buffers!(c, buffers, grid, Val(:corners))    

    return nothing
end

@inline mpi_communication_side(::Val{fill_southwest_halo!}) = :southwest
@inline mpi_communication_side(::Val{fill_southeast_halo!}) = :southeast
@inline mpi_communication_side(::Val{fill_northwest_halo!}) = :northwest
@inline mpi_communication_side(::Val{fill_northeast_halo!}) = :northeast

@inline mpi_communication_side(::Val{fill_west_and_east_halo!})   = :west_and_east
@inline mpi_communication_side(::Val{fill_south_and_north_halo!}) = :south_and_north
@inline mpi_communication_side(::Val{fill_bottom_and_top_halo!})  = :bottom_and_top

cooperative_wait(req::MPI.Request) = MPI.Waitall(req)
cooperative_waitall!(req::Array{MPI.Request}) = MPI.Waitall(req)

function fill_halo_event!(task, halo_tuple, c, indices, loc, arch::DistributedArch, grid::DistributedGrid, buffers, args...; blocking = true, kwargs...)
    fill_halo!  = halo_tuple[1][task]
    bc_left     = halo_tuple[2][task]
    bc_right    = halo_tuple[3][task]

    # Calculate size and offset of the fill_halo kernel
    size   = fill_halo_size(c, fill_halo!, indices, bc_left, loc, grid)
    offset = fill_halo_offset(size, fill_halo!, indices)

    requests = fill_halo!(c, bc_left, bc_right, size, offset, loc, arch, grid, buffers, args...; kwargs...)

    # if `isnothing(requests)`, `fill_halo!` did not involve MPI 
    if isnothing(requests)
        return nothing
    end

    # Overlapping communication and computation, store requests in a `MPI.Request`
    # pool to be waited upon after tendency calculation
    if !blocking && !(arch isa BlockingDistributedArch)
        push!(arch.mpi_requests, requests...)
        return nothing
    end

    # Syncronous MPI fill_halo_event!
    cooperative_waitall!(requests)
    # Reset MPI tag
    arch.mpi_tag[1] -= arch.mpi_tag[1]

    buffer_side = mpi_communication_side(Val(fill_halo!))
    recv_from_buffers!(c, buffers, grid, Val(buffer_side))    

    return nothing
end

#####
##### fill_west_and_east_halo!   }
##### fill_south_and_north_halo! } for when both halos are communicative (Single communicating halos are to be implemented)
##### fill_bottom_and_top_halo!  }
#####

for (side, opposite_side, dir) in zip([:west, :south, :bottom], [:east, :north, :top], [1, 2, 3])
    fill_both_halo! = Symbol("fill_$(side)_and_$(opposite_side)_halo!")
    fill_side_halo! = Symbol("fill_$(side)_halo!")
    send_side_halo  = Symbol("send_$(side)_halo")
    fill_opposite_side_halo! = Symbol("fill_$(opposite_side)_halo!")
    send_opposite_side_halo  = Symbol("send_$(opposite_side)_halo")
    recv_and_fill_side_halo! = Symbol("recv_and_fill_$(side)_halo!")
    recv_and_fill_opposite_side_halo! = Symbol("recv_and_fill_$(opposite_side)_halo!")
    fill_all_send_buffers! = Symbol("fill_$(side)_and_$(opposite_side)_send_buffers!")
    fill_side_send_buffers! = Symbol("fill_$(side)_send_buffers!")
    fill_opposite_side_send_buffers! = Symbol("fill_$(opposite_side)_send_buffers!")

    @eval begin
        function $fill_both_halo!(c, bc_side::DCBCT, bc_opposite_side::DCBCT, size, offset, loc, arch::DistributedArch, 
                                  grid::DistributedGrid, buffers, args...; kwargs...)

            @assert bc_side.condition.from == bc_opposite_side.condition.from  # Extra protection in case of bugs
            local_rank = bc_side.condition.from

            recv_req1 = $recv_and_fill_side_halo!(c, grid, arch, loc[$dir], loc, local_rank, bc_side.condition.to, buffers)
            recv_req2 = $recv_and_fill_opposite_side_halo!(c, grid, arch, loc[$dir], loc, local_rank, bc_opposite_side.condition.to, buffers)

            # This has to be synchronized!!
            $fill_all_send_buffers!(c, buffers, grid)

            sync_device!(child_architecture(arch))

            send_req1 = $send_side_halo(c, grid, arch, loc[$dir], loc, local_rank, bc_side.condition.to, buffers)
            send_req2 = $send_opposite_side_halo(c, grid, arch, loc[$dir], loc, local_rank, bc_opposite_side.condition.to, buffers)

            return [send_req1, send_req2, recv_req1, recv_req2]
        end

        function $fill_both_halo!(c, bc_side::DCBCT, bc_opposite_side, size, offset, loc, arch::DistributedArch, 
                                  grid::DistributedGrid, buffers, args...; kwargs...)

            child_arch = child_architecture(arch)
            local_rank = bc_side.condition.from

            recv_req = $recv_and_fill_side_halo!(c, grid, arch, loc[$dir], loc, local_rank, bc_side.condition.to, buffers)

            $fill_opposite_side_halo!(c, bc_opposite_side, size, offset, loc, arch, grid, buffers, args...; kwargs...)
            $fill_side_send_buffers!(c, buffers, grid)

            sync_device!(child_arch)

            send_req = $send_side_halo(c, grid, arch, loc[$dir], loc, local_rank, bc_side.condition.to, buffers)
            
            return [send_req, recv_req]
        end

        function $fill_both_halo!(c, bc_side, bc_opposite_side::DCBCT, size, offset, loc, arch::DistributedArch, 
                                  grid::DistributedGrid, buffers, args...; kwargs...)

            child_arch = child_architecture(arch)
            local_rank = bc_opposite_side.condition.from

            recv_req = $recv_and_fill_opposite_side_halo!(c, grid, arch, loc[$dir], loc, local_rank, bc_opposite_side.condition.to, buffers)

            $fill_side_halo!(c, bc_side, size, offset, loc, arch, grid, buffers, args...; kwargs...)
            $fill_opposite_side_send_buffers!(c, buffers, grid)

            sync_device!(child_arch)

            send_req = $send_opposite_side_halo(c, grid, arch, loc[$dir], loc, local_rank, bc_opposite_side.condition.to, buffers)

            return [send_req, recv_req]
        end
    end
end

#####
##### Sending halos
#####

for side in sides
    side_str = string(side)
    send_side_halo = Symbol("send_$(side)_halo")
    underlying_side_boundary = Symbol("underlying_$(side)_boundary")
    side_send_tag = Symbol("$(side)_send_tag")
    get_side_send_buffer = Symbol("get_$(side)_send_buffer")

    @eval begin
        function $send_side_halo(c, grid, arch, side_location, location, local_rank, rank_to_send_to, buffers)
            send_buffer = $get_side_send_buffer(c, grid, side_location, buffers, arch)
            send_tag = $side_send_tag(arch, location, local_rank, rank_to_send_to)

            @debug "Sending " * $side_str * " halo: local_rank=$local_rank, rank_to_send_to=$rank_to_send_to, send_tag=$send_tag"
            
            send_req = MPI.Isend(send_buffer, rank_to_send_to, send_tag, arch.communicator)

            return send_req
        end

        @inline $get_side_send_buffer(c, grid, side_location, buffers, ::ViewsDistributedArch) = $underlying_side_boundary(c, grid, side_location)
        @inline $get_side_send_buffer(c, grid, side_location, buffers, arch)                   = buffers.$side.send     
    end
end

#####
##### Receiving and filling halos (buffer is a view so it gets filled upon receive)
#####

for side in sides
    side_str = string(side)
    recv_and_fill_side_halo! = Symbol("recv_and_fill_$(side)_halo!")
    underlying_side_halo = Symbol("underlying_$(side)_halo")
    side_recv_tag = Symbol("$(side)_recv_tag")
    get_side_recv_buffer = Symbol("get_$(side)_recv_buffer")

    @eval begin
        function $recv_and_fill_side_halo!(c, grid, arch, side_location, location, local_rank, rank_to_recv_from, buffers)
            recv_buffer = $get_side_recv_buffer(c, grid, side_location, buffers, arch)
            recv_tag = $side_recv_tag(arch, location, local_rank, rank_to_recv_from)

            @debug "Receiving " * $side_str * " halo: local_rank=$local_rank, rank_to_recv_from=$rank_to_recv_from, recv_tag=$recv_tag"
            recv_req = MPI.Irecv!(recv_buffer, rank_to_recv_from, recv_tag, arch.communicator)

            return recv_req
        end

        @inline $get_side_recv_buffer(c, grid, side_location, buffers, ::ViewsDistributedArch) = $underlying_side_halo(c, grid, side_location)
        @inline $get_side_recv_buffer(c, grid, side_location, buffers, arch)                   = buffers.$side.recv
    end
end