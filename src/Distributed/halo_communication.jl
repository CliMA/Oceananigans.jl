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
const MAX_RANKS = 10^3
RANK_DIGITS = 3

# Define functions that return unique send and recv MPI tags for each side.
for side in sides
    side_str = string(side)
    send_tag_fn_name = Symbol(side, :_send_tag)
    recv_tag_fn_name = Symbol(side, :_recv_tag)
    @eval begin
        function $send_tag_fn_name(bc)
            from_digits = string(bc.condition.from, pad=RANK_DIGITS)
            to_digits = string(bc.condition.to, pad=RANK_DIGITS)
            side_digit = string(side_id[Symbol($side_str)])
            return parse(Int, from_digits * to_digits * side_digit)
        end

        function $recv_tag_fn_name(bc)
            from_digits = string(bc.condition.from, pad=RANK_DIGITS)
            to_digits = string(bc.condition.to, pad=RANK_DIGITS)
            side_digit = string(side_id[opposite_side[Symbol($side_str)]])
            return parse(Int, to_digits * from_digits * side_digit)
        end
    end
end

#####
##### Filling halos for halo communication boundary conditions
#####

@inline   west_send_buffer(c, N, H) = c.parent[N+1:N+H, :, :]
@inline   east_send_buffer(c, N, H) = c.parent[1+H:2H,  :, :]
@inline  south_send_buffer(c, N, H) = c.parent[:, N+1:N+H, :]
@inline  north_send_buffer(c, N, H) = c.parent[:, 1+H:2H,  :]
@inline    top_send_buffer(c, N, H) = c.parent[:, :,  1+H:2H]
@inline bottom_send_buffer(c, N, H) = c.parent[:, :, N+1:N+H]

@inline west_recv_buffer(grid)  = zeros(grid.Hx, grid.Ny + 2grid.Hy, grid.Nz + 2grid.Hz)
@inline south_recv_buffer(grid) = zeros(grid.Nx + 2grid.Hx, grid.Hy, grid.Nz + 2grid.Hz)
@inline top_recv_buffer(grid)   = zeros(grid.Nx + 2grid.Hx, grid.Ny + 2grid.Hy, grid.Hz)

const   east_recv_buffer =  west_recv_buffer
const  north_recv_buffer = south_recv_buffer
const bottom_recv_buffer =   top_recv_buffer

@inline   copy_recv_buffer_into_west_halo!(c, N, H, buf) = (c.parent[    1:H,    :, :] .= buf)
@inline   copy_recv_buffer_into_east_halo!(c, N, H, buf) = (c.parent[N+H+1:N+2H, :, :] .= buf)
@inline  copy_recv_buffer_into_south_halo!(c, N, H, buf) = (c.parent[:,     1:H,    :] .= buf)
@inline  copy_recv_buffer_into_north_halo!(c, N, H, buf) = (c.parent[:, N+H+1:N+2H, :] .= buf)
@inline copy_recv_buffer_into_bottom_halo!(c, N, H, buf) = (c.parent[:, :,     1:H   ] .= buf)
@inline    copy_recv_buffer_into_top_halo!(c, N, H, buf) = (c.parent[:, :, N+H+1:N+2H] .= buf)

function fill_halo_regions!(c::AbstractArray, bcs, arch::AbstractMultiArchitecture, grid, args...)

    barrier = Event(device(child_architecture(arch)))

    east_event, west_event = fill_east_and_west_halos!(c, bcs.east, bcs.west, arch, barrier, grid, args...)
    # north_event, south_event = fill_north_and_south_halos!(c, bcs.north, bcs.south, arch, barrier, grid, args...)
    # top_event, bottom_event = fill_top_and_bottom_halos!(c, bcs.east, bcs.west, arch, barrier, grid, args...)

    events = [east_event, west_event] # , north_event, south_event, top_event, bottom_event]
    events = filter(e -> e isa Event, events)
    wait(device(child_architecture(arch)), MultiEvent(Tuple(events)))

    return nothing
end

function fill_east_and_west_halos!(c, east_bc, west_bc, arch, barrier, grid, args...)
    east_event = fill_east_halo!(c, east_bc, child_architecture(arch), barrier, grid, args...)
    west_event = fill_west_halo!(c, west_bc, child_architecture(arch), barrier, grid, args...)
    return east_event, west_event
end

function fill_east_and_west_halos!(c, east_bc::HaloCommunicationBC, west_bc::HaloCommunicationBC, arch, barrier, grid, args...)
    # 1 -> send east halo to eastern rank and fill east halo from eastern rank's west halo.
    # 2 -> send west halo to western rank and fill west halo from western rank's east halo.

    @assert east_bc.condition.from == west_bc.condition.from
    my_rank = east_bc.condition.from

    rank_to_send_to1 = east_bc.condition.to
    rank_to_send_to2 = west_bc.condition.to

    send_buffer1 = east_send_buffer(c, grid.Nx, grid.Hx)
    send_buffer2 = west_send_buffer(c, grid.Nx, grid.Hx)

    send_tag1 = east_send_tag(east_bc)
    send_tag2 = west_send_tag(west_bc)

    @info "MPI.Isend: my_rank=$my_rank, rank_to_send_to=$rank_to_send_to1, send_tag=$send_tag1"
    @info "MPI.Isend: my_rank=$my_rank, rank_to_send_to=$rank_to_send_to2, send_tag=$send_tag2"

    send_req1 = MPI.Isend(send_buffer1, rank_to_send_to1, send_tag1, MPI.COMM_WORLD)
    send_req2 = MPI.Isend(send_buffer2, rank_to_send_to2, send_tag2, MPI.COMM_WORLD)

    rank_to_recv_from1 = east_bc.condition.to
    rank_to_recv_from2 = west_bc.condition.to

    recv_buffer1 = east_recv_buffer(grid)
    recv_buffer2 = west_recv_buffer(grid)

    recv_tag1 = east_recv_tag(east_bc)
    recv_tag2 = west_recv_tag(west_bc)

    @info "MPI.Recv!: my_rank=$my_rank, rank_to_recv_from=$rank_to_recv_from1, recv_tag=$recv_tag1"
    @info "MPI.Recv!: my_rank=$my_rank, rank_to_recv_from=$rank_to_recv_from2, recv_tag=$recv_tag2"

    MPI.Recv!(recv_buffer1, rank_to_recv_from1, recv_tag1, MPI.COMM_WORLD)
    MPI.Recv!(recv_buffer2, rank_to_recv_from2, recv_tag2, MPI.COMM_WORLD)

    @info "Communication done!"

    copy_recv_buffer_into_east_halo!(c, grid.Nx, grid.Hx, recv_buffer1)
    copy_recv_buffer_into_west_halo!(c, grid.Nx, grid.Hx, recv_buffer2)

    # @info "Sendrecv!: my_rank=$my_rank, rank_send_to=rank_recv_from=$rank_send_to, " *
    #       "send_tag=$send_tag, recv_tag=$recv_tag"
    #
    # MPI.Sendrecv!(send_buffer, rank_send_to,   send_tag,
    #               recv_buffer, rank_recv_from, recv_tag,
    #               MPI.COMM_WORLD)
    #
    # @info "Sendrecv!: my_rank=$my_rank done!"

    return nothing, nothing
end