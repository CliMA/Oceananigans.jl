using KernelAbstractions: @kernel, @index, Event, MultiEvent

import Oceananigans.BoundaryConditions:
    fill_halo_regions!,
    fill_west_halo!, fill_east_halo!, fill_south_halo!,
    fill_north_halo!, fill_bottom_halo!, fill_top_halo!

#####
##### MPI tags for halo communication BCs
#####

sides  = (:west, :east, :south, :north, :top, :bottom)

side_id = Dict(
    :east => 1, :west => 2,
    :north => 3, :south => 4,
    :top => 5, :bottom => 6
)

opposite_side = Dict(
    :east => :west, :west => :east,
    :north => :south, :south => :north,
    :top => :bottom, :bottom => :top
)

# Unfortunately can't call MPI.Comm_size(MPI.COMM_WORLD) before MPI.Init().
MAX_RANKS = 10^3
RANK_DIGITS = 3

# Define functions that return unique send and recv MPI tags for each side.
# It's an integer where
#   digit 1: the side
#   digits 2-4: the from rank
#   digits 5-7: the to rank

for side in sides
    side_str = string(side)
    send_tag_fn_name = Symbol("$(side)_send_tag")
    recv_tag_fn_name = Symbol("$(side)_recv_tag")
    @eval begin
        function $send_tag_fn_name(my_rank, rank_to_send_to)
            from_digits = string(my_rank, pad=RANK_DIGITS)
            to_digits = string(rank_to_send_to, pad=RANK_DIGITS)
            side_digit = string(side_id[Symbol($side_str)])
            return parse(Int, from_digits * to_digits * side_digit)
        end

        function $recv_tag_fn_name(my_rank, rank_to_recv_from)
            from_digits = string(rank_to_recv_from, pad=RANK_DIGITS)
            to_digits = string(my_rank, pad=RANK_DIGITS)
            side_digit = string(side_id[opposite_side[Symbol($side_str)]])
            return parse(Int, from_digits * to_digits * side_digit)
        end
    end
end

#####
##### Filling halos for halo communication boundary conditions
#####

fill_halo_regions!(field::AbstractField{LX, LY, LZ}, arch, args...) where {LX, LY, LZ} =
    fill_halo_regions!(field.data, field.boundary_conditions, arch, field.grid, (LX, LY, LZ), args...)

function fill_halo_regions!(c::AbstractArray, bcs, arch, grid, c_location, args...)

    barrier = Event(device(child_architecture(arch)))

    east_event, west_event = fill_east_and_west_halos!(c, bcs.east, bcs.west, arch, barrier, grid, c_location, args...)
    north_event, south_event = fill_north_and_south_halos!(c, bcs.north, bcs.south, arch, barrier, grid, c_location, args...)
    top_event, bottom_event = fill_top_and_bottom_halos!(c, bcs.top, bcs.bottom, arch, barrier, grid, c_location, args...)

    events = [east_event, west_event, north_event, south_event, top_event, bottom_event]
    events = filter(e -> e isa Event, events)
    wait(device(child_architecture(arch)), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### fill_east_and_west_halos!   }
##### fill_north_and_south_halos! } for non-communicating boundary conditions (fallback)
##### fill_top_and_bottom_halos!  }
#####

for (side, opposite_side) in zip([:east, :north, :top], [:west, :south, :bottom])
    fill_both_halos! = Symbol("fill_$(side)_and_$(opposite_side)_halos!")
    fill_side_halo! = Symbol("fill_$(side)_halo!")
    fill_opposite_side_halo! = Symbol("fill_$(opposite_side)_halo!")

    @eval begin
        function $fill_both_halos!(c, bc_side, bc_opposite_side, arch, barrier, grid, args...)
            event_side = $fill_side_halo!(c, bc_side, child_architecture(arch), barrier, grid, args...)
            event_opposite_side = $fill_opposite_side_halo!(c, bc_opposite_side, child_architecture(arch), barrier, grid, args...)
            return event_side, event_opposite_side
        end
    end
end

#####
##### fill_east_and_west_halos!   }
##### fill_north_and_south_halos! } for when both halos are communicative
##### fill_top_and_bottom_halos!  }
#####

for (side, opposite_side) in zip([:east, :north, :top], [:west, :south, :bottom])
    fill_both_halos! = Symbol("fill_$(side)_and_$(opposite_side)_halos!")
    send_side_halo = Symbol("send_$(side)_halo")
    send_opposite_side_halo = Symbol("send_$(opposite_side)_halo")
    recv_and_fill_side_halo! = Symbol("recv_and_fill_$(side)_halo!")
    recv_and_fill_opposite_side_halo! = Symbol("recv_and_fill_$(opposite_side)_halo!")

    @eval begin
        function $fill_both_halos!(c, bc_side::HaloCommunicationBC, bc_opposite_side::HaloCommunicationBC, arch, barrier, grid, c_location, args...)
            @assert bc_side.condition.from == bc_opposite_side.condition.from  # Extra protection in case of bugs
            my_rank = bc_side.condition.from

            $send_side_halo(c, grid, c_location, my_rank, bc_side.condition.to)
            $send_opposite_side_halo(c, grid, c_location, my_rank, bc_opposite_side.condition.to)

            $recv_and_fill_side_halo!(c, grid, c_location, my_rank, bc_side.condition.to)
            $recv_and_fill_opposite_side_halo!(c, grid, c_location, my_rank, bc_opposite_side.condition.to)

            return nothing, nothing
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
        function $send_side_halo(c, grid, c_location, my_rank, rank_to_send_to)
            send_buffer = $underlying_side_boundary(c, grid, c_location)
            send_tag = $side_send_tag(my_rank, rank_to_send_to)

            @debug "Sending " * $side_str * " halo: my_rank=$my_rank, rank_to_send_to=$rank_to_send_to, send_tag=$send_tag"
            status = MPI.Isend(send_buffer, rank_to_send_to, send_tag, MPI.COMM_WORLD)

            return status
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
        function $recv_and_fill_side_halo!(c, grid, c_location, my_rank, rank_to_recv_from)
            recv_buffer = $underlying_side_halo(c, grid, c_location)
            recv_tag = $side_recv_tag(my_rank, rank_to_recv_from)

            @debug "Receiving " * $side_str * " halo: my_rank=$my_rank, rank_to_recv_from=$rank_to_recv_from, recv_tag=$recv_tag"
            MPI.Recv!(recv_buffer, rank_to_recv_from, recv_tag, MPI.COMM_WORLD)

            return nothing
        end
    end
end
