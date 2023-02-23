using KernelAbstractions: @kernel, @index
using OffsetArrays: OffsetArray
using CUDA: synchronize
import Oceananigans.Utils: sync_device!
using Oceananigans.Fields: fill_west_and_east_send_buffers!, 
                           fill_south_and_north_send_buffers!, 
                           fill_west_send_buffers!,
                           fill_east_send_buffers!,
                           fill_south_send_buffers!,
                           fill_north_send_buffers!,
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

@inline sync_device!(::GPU) = synchronize()

#####
##### MPI tags for halo communication BCs
#####

sides  = (:west, :east, :south, :north, :top, :bottom)
side_id = Dict(side => n for (n, side) in enumerate(sides))

opposite_side = Dict(
    :west => :east, :east => :west,
    :south => :north, :north => :south,
    :bottom => :top, :top => :bottom
)

# Define functions that return unique send and recv MPI tags for each side.
# It's an integer where
#   digit 1-2: an identifier for the field that is reset each timestep
#   digit 3: the side
#   digits 4-6: the "from" rank
#   digits 7-9: the "to" rank

RANK_DIGITS = 2
ID_DIGITS = 3

for side in sides
    side_str = string(side)
    send_tag_fn_name = Symbol("$(side)_send_tag")
    recv_tag_fn_name = Symbol("$(side)_recv_tag")
    @eval begin
        function $send_tag_fn_name(arch, local_rank, rank_to_send_to)
            field_id    = string(arch.mpi_tag[1], pad=ID_DIGITS)
            from_digits = string(local_rank, pad=RANK_DIGITS)
            to_digits   = string(rank_to_send_to, pad=RANK_DIGITS)
            side_digit  = string(side_id[Symbol($side_str)])
            return parse(Int, field_id * side_digit * from_digits * to_digits)
        end

        function $recv_tag_fn_name(arch, local_rank, rank_to_recv_from)
            field_id    = string(arch.mpi_tag[1], pad=ID_DIGITS)
            from_digits = string(rank_to_recv_from, pad=RANK_DIGITS)
            to_digits   = string(local_rank, pad=RANK_DIGITS)
            side_digit  = string(side_id[opposite_side[Symbol($side_str)]])
            return parse(Int, field_id * side_digit * from_digits * to_digits)
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

    # fill_eventual_corners!(halo_tuple, c, indices, loc, arch, grid, buffers, args...; kwargs...)
    arch.mpi_tag[1] += 1

    return nothing
end

# If more than one direction is communicating we need to repeat one fill halo to fill the freaking corners!
function fill_eventual_corners!(halo_tuple, c, indices, loc, arch, grid, buffers, args...; kwargs...)
    hbc_left  = filter(bc -> bc isa DCBC, halo_tuple[2])
    hbc_right = filter(bc -> bc isa DCBC, halo_tuple[3])

    # 2D/3D Parallelization when `length(hbc_left) > 1 || length(hbc_right) > 1`
    if length(hbc_left) > 1 
        idx = findfirst(bc -> bc isa DCBC, halo_tuple[2])
        fill_halo_event!(idx, halo_tuple, c, indices, loc, arch, grid, buffers, args...; kwargs...)
        return nothing
    end

    if length(hbc_right) > 1 
        idx = findfirst(bc -> bc isa DCBC, halo_tuple[3])
        fill_halo_event!(idx, halo_tuple, c, indices, loc, arch, grid, buffers, args...; kwargs...)
        return nothing
    end
end

@inline mpi_communication_side(::Val{fill_west_and_east_halo!})   = :west_and_east
@inline mpi_communication_side(::Val{fill_south_and_north_halo!}) = :south_and_north
@inline mpi_communication_side(::Val{fill_bottom_and_top_halo!})  = :bottom_and_top

function fill_halo_event!(task, halo_tuple, c, indices, loc, arch::DistributedArch, grid::DistributedGrid, buffers, args...; async = false, kwargs...)
    fill_halo!  = halo_tuple[1][task]
    bc_left     = halo_tuple[2][task]
    bc_right    = halo_tuple[3][task]

    # Calculate size and offset of the fill_halo kernel
    size   = fill_halo_size(c, fill_halo!, indices, bc_left, loc, grid)
    offset = fill_halo_offset(size, fill_halo!, indices)

    requests = fill_halo!(c, bc_left, bc_right, size, offset, loc, arch, grid, buffers, args...; kwargs...)

    if isnothing(requests)
        return nothing
    end

    if async && !(arch isa SynchedDistributedArch)
        push!(arch.mpi_requests, requests...)
        return nothing
    end

    MPI.Waitall(requests)
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

            # This has to be synchronized!!
            $fill_all_send_buffers!(c, buffers, grid)

            sync_device!(child_architecture(arch))

            recv_req1 = $recv_and_fill_side_halo!(c, grid, arch, loc[$dir], local_rank, bc_side.condition.to, buffers)
            recv_req2 = $recv_and_fill_opposite_side_halo!(c, grid, arch, loc[$dir], local_rank, bc_opposite_side.condition.to, buffers)

            send_req1 = $send_side_halo(c, grid, arch, loc[$dir], local_rank, bc_side.condition.to, buffers)
            send_req2 = $send_opposite_side_halo(c, grid, arch, loc[$dir], local_rank, bc_opposite_side.condition.to, buffers)

            return [send_req1, send_req2, recv_req1, recv_req2]
        end

        function $fill_both_halo!(c, bc_side::DCBCT, bc_opposite_side, size, offset, loc, arch::DistributedArch, 
                                  grid::DistributedGrid, buffers, args...; kwargs...)

            child_arch = child_architecture(arch)
            local_rank = bc_side.condition.from

            $fill_opposite_side_halo!(c, bc_opposite_side, size, offset, loc, arch, grid, buffers, args...; kwargs...)
            $fill_side_send_buffers!(c, buffers, grid)

            sync_device!(child_arch)

            recv_req = $recv_and_fill_side_halo!(c, grid, arch, loc[$dir], local_rank, bc_side.condition.to, buffers)
            send_req = $send_side_halo(c, grid, arch, loc[$dir], local_rank, bc_side.condition.to, buffers)
            
            return [send_req, recv_req]
        end

        function $fill_both_halo!(c, bc_side, bc_opposite_side::DCBCT, size, offset, loc, arch::DistributedArch, 
                                  grid::DistributedGrid, buffers, args...; kwargs...)

            child_arch = child_architecture(arch)
            local_rank = bc_opposite_side.condition.from

            $fill_side_halo!(c, bc_side, size, offset, loc, arch, grid, buffers, args...; kwargs...)
            $fill_opposite_side_send_buffers!(c, buffers, grid)

            sync_device!(child_arch)

            recv_req = $recv_and_fill_opposite_side_halo!(c, grid, arch, loc[$dir], local_rank, bc_opposite_side.condition.to, buffers)
            send_req = $send_opposite_side_halo(c, grid, arch, loc[$dir], local_rank, bc_opposite_side.condition.to, buffers)

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
        function $send_side_halo(c, grid, arch, side_location, local_rank, rank_to_send_to, buffers)
            send_buffer = $get_side_send_buffer(c, grid, side_location, buffers, arch)
            send_tag = $side_send_tag(arch, local_rank, rank_to_send_to)

            @debug "Sending " * $side_str * " halo: local_rank=$local_rank, rank_to_send_to=$rank_to_send_to, send_tag=$send_tag"
            send_req = MPI.Isend(send_buffer, rank_to_send_to, send_tag, arch.communicator)

            return send_req
        end

        @inline $get_side_send_buffer(c, grid, side_location, buffers, ::ViewsDistributedArch) = $underlying_side_boundary(c, grid, side_location)
        @inline $get_side_send_buffer(c, grid, side_location, buffers, arch)             = buffers.$side.send     
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
        function $recv_and_fill_side_halo!(c, grid, arch, side_location, local_rank, rank_to_recv_from, buffers)
            recv_buffer = $get_side_recv_buffer(c, grid, side_location, buffers, arch)
            recv_tag = $side_recv_tag(arch, local_rank, rank_to_recv_from)

            @debug "Receiving " * $side_str * " halo: local_rank=$local_rank, rank_to_recv_from=$rank_to_recv_from, recv_tag=$recv_tag"
            recv_req = MPI.Irecv!(recv_buffer, rank_to_recv_from, recv_tag, arch.communicator)

            return recv_req
        end

        @inline $get_side_recv_buffer(c, grid, side_location, buffers, ::ViewsDistributedArch) = $underlying_side_halo(c, grid, side_location)
        @inline $get_side_recv_buffer(c, grid, side_location, buffers, arch)             = buffers.$side.recv
    end
end