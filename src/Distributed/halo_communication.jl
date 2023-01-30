using KernelAbstractions: @kernel, @index, Event, MultiEvent
using OffsetArrays: OffsetArray

import Oceananigans.Fields: tupled_fill_halo_regions!

using Oceananigans.BoundaryConditions:
    fill_west_and_east_halo!, 
    fill_south_and_north_halo!,
    fill_bottom_and_top_halo!,
    fill_halo_size,
    fill_halo_offset



import Oceananigans.BoundaryConditions:
    fill_halo_regions!,
    fill_west_halo!, fill_east_halo!, fill_south_halo!,
    fill_north_halo!, fill_bottom_halo!, fill_top_halo!

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
#   digit 1: the side
#   digits 2-4: the "from" rank
#   digits 5-7: the "to" rank

RANK_DIGITS = 3

for side in sides
    side_str = string(side)
    send_tag_fn_name = Symbol("$(side)_send_tag")
    recv_tag_fn_name = Symbol("$(side)_recv_tag")
    @eval begin
        function $send_tag_fn_name(local_rank, rank_to_send_to)
            from_digits = string(local_rank, pad=RANK_DIGITS)
            to_digits = string(rank_to_send_to, pad=RANK_DIGITS)
            side_digit = string(side_id[Symbol($side_str)])
            return parse(Int, from_digits * to_digits * side_digit)
        end

        function $recv_tag_fn_name(local_rank, rank_to_recv_from)
            from_digits = string(rank_to_recv_from, pad=RANK_DIGITS)
            to_digits = string(local_rank, pad=RANK_DIGITS)
            side_digit = string(side_id[opposite_side[Symbol($side_str)]])
            return parse(Int, from_digits * to_digits * side_digit)
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

# REMEMBER! indices, kernel_size (:yz, :xz and :xy) and offset are not used in distributed halo filling because sliced fields are not supported yet
# TODO: distributed halo filling does not support windowed fields at the moment
# TODO: combination of communicating and other boundary conditions in one direction are not implemented yet!
function fill_halo_regions!(c::OffsetArray, bcs, indices, loc, grid::DistributedGrid, args...; kwargs...)
    arch    = architecture(grid)
    barrier = Event(device(child_architecture(arch)))

    size_x = fill_halo_size(c, fill_west_and_east_halo!,   indices, bcs.west,   loc, grid)
    size_y = fill_halo_size(c, fill_south_and_north_halo!, indices, bcs.south,  loc, grid)
    size_z = fill_halo_size(c, fill_bottom_and_top_halo!,  indices, bcs.bottom, loc, grid)

    offset_x = fill_halo_offset(size_x, fill_west_and_east_halo!,   indices)
    offset_y = fill_halo_offset(size_y, fill_south_and_north_halo!, indices)
    offset_z = fill_halo_offset(size_z, fill_bottom_and_top_halo!,  indices)

    x_events_requests =   fill_west_and_east_halos!(c, bcs.west,   bcs.east,  size_x, offset_x, loc, arch, barrier, grid, args...; kwargs...)
    y_events_requests = fill_south_and_north_halos!(c, bcs.south,  bcs.north, size_y, offset_y, loc, arch, barrier, grid, args...; kwargs...)
    z_events_requests =  fill_bottom_and_top_halos!(c, bcs.bottom, bcs.top,   size_z, offset_z, loc, arch, barrier, grid, args...; kwargs...)

    events_and_requests = [x_events_requests..., y_events_requests..., z_events_requests...]

    mpi_requests = filter(e -> e isa MPI.Request, events_and_requests) |> Array{MPI.Request}
    # Length check needed until this PR is merged: https://github.com/JuliaParallel/MPI.jl/pull/458
    length(mpi_requests) > 0 && MPI.Waitall!(mpi_requests)

    events = filter(e -> e isa Event, events_and_requests)
    wait(device(child_architecture(arch)), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### fill_west_and_east_halos!   }
##### fill_south_and_north_halos! } for non-communicating boundary conditions (fallback)
##### fill_bottom_and_top_halos!  }
#####

for (side, opposite_side) in zip([:west, :south, :bottom], [:east, :north, :top])
    fill_both_halos! = Symbol("fill_$(side)_and_$(opposite_side)_halos!")
    fill_both_halo!  = Symbol("fill_$(side)_and_$(opposite_side)_halo!")

    @eval begin
        function $fill_both_halos!(c, bc_side, bc_opposite_side, size, offset, loc, arch, barrier, grid, args...; kwargs...)
                event = $fill_both_halo!(c, bc_side, bc_opposite_side, size, offset, loc, child_architecture(arch), barrier, grid, args...; kwargs...)
            return [event]
        end
    end
end

#####
##### fill_west_and_east_halos!   }
##### fill_south_and_north_halos! } for when both halos are communicative
##### fill_bottom_and_top_halos!  }
#####

const CBCT = Union{HaloCommunicationBC, NTuple{<:Any, <:HaloCommunicationBC}}

for (side, opposite_side, dir) in zip([:west, :south, :bottom], [:east, :north, :top], [1, 2, 3])
    fill_both_halos! = Symbol("fill_$(side)_and_$(opposite_side)_halos!")
    send_side_halo = Symbol("send_$(side)_halo")
    send_opposite_side_halo = Symbol("send_$(opposite_side)_halo")
    recv_and_fill_side_halo! = Symbol("recv_and_fill_$(side)_halo!")
    recv_and_fill_opposite_side_halo! = Symbol("recv_and_fill_$(opposite_side)_halo!")

    @eval begin
        function $fill_both_halos!(c, bc_side::CBCT, bc_opposite_side::CBCT, size, offset, loc, arch, 
                                   barrier, grid, args...; kwargs...)

            @assert bc_side.condition.from == bc_opposite_side.condition.from  # Extra protection in case of bugs
            local_rank = bc_side.condition.from

            send_req1 = $send_side_halo(c, grid, loc[$dir], local_rank, bc_side.condition.to)
            send_req2 = $send_opposite_side_halo(c, grid, loc[$dir], local_rank, bc_opposite_side.condition.to)

            recv_req1 = $recv_and_fill_side_halo!(c, grid, loc[$dir], local_rank, bc_side.condition.to)
            recv_req2 = $recv_and_fill_opposite_side_halo!(c, grid, loc[$dir], local_rank, bc_opposite_side.condition.to)

            return send_req1, send_req2, recv_req1, recv_req2
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

    @eval begin
        function $send_side_halo(c, grid, side_location, local_rank, rank_to_send_to)
            send_buffer = $underlying_side_boundary(c, grid, side_location)
            send_tag = $side_send_tag(local_rank, rank_to_send_to)

            @debug "Sending " * $side_str * " halo: local_rank=$local_rank, rank_to_send_to=$rank_to_send_to, send_tag=$send_tag"
            send_req = MPI.Isend(send_buffer, rank_to_send_to, send_tag, MPI.COMM_WORLD)

            return send_req
        end
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

    @eval begin
        function $recv_and_fill_side_halo!(c, grid, side_location, local_rank, rank_to_recv_from)
            recv_buffer = $underlying_side_halo(c, grid, side_location)
            recv_tag = $side_recv_tag(local_rank, rank_to_recv_from)

            @debug "Receiving " * $side_str * " halo: local_rank=$local_rank, rank_to_recv_from=$rank_to_recv_from, recv_tag=$recv_tag"
            recv_req = MPI.Irecv!(recv_buffer, rank_to_recv_from, recv_tag, MPI.COMM_WORLD)

            return recv_req
        end
    end
end
