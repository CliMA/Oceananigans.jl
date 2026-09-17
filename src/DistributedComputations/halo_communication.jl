using KernelAbstractions: @kernel, @index

using Oceananigans.Fields: instantiated_location

using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions:
    DistributedFillHalo,
    get_boundary_kernels

import Oceananigans.BoundaryConditions: fill_halo_event!, fill_halo_regions!

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

ID_DIGITS   = 2

# A Hashing function which returns a unique
# integer between 0 and 26 for a combination of
# 3 locations wither Center, Face, or Nothing
location_counter = 0
for LX in (:Face, :Center, :Nothing)
    for LY in (:Face, :Center, :Nothing)
        for LZ in (:Face, :Center, :Nothing)
            @eval loc_id(::$LX, ::$LY, ::$LZ) = $location_counter
            global location_counter += 1
        end
    end
end

# Functions that return unique send and recv MPI tags for each field, side, field location
# the MPI tag is an integer with:
#   digit 1-2: a unique integer for the field
#   digit 3-4: a unique identifier for the field's location that goes from 0 - 26 (see `loc_id`)
#   digit 5: the side we send / receive from

for side in sides
    side_str = string(side)
    send_tag_fn_name = Symbol("$(side)_send_tag")
    recv_tag_fn_name = Symbol("$(side)_recv_tag")
    @eval begin
        function $send_tag_fn_name(arch, grid, field_tag, location)
            field_id   = string(field_tag, pad=ID_DIGITS)
            loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
            side_digit = string(side_id[Symbol($side_str)])
            return parse(Int, field_id * loc_digit * side_digit)
        end

        function $recv_tag_fn_name(arch, grid, field_tag, location)
            field_id   = string(field_tag, pad=ID_DIGITS)
            loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
            side_digit = string(side_id[opposite_side[Symbol($side_str)]])
            return parse(Int, field_id * loc_digit * side_digit)
        end
    end
end

#####
##### Filling halos for halo communication boundary conditions
#####

function fill_halo_regions!(field::DistributedField, args...; kwargs...)
    fill_halo_regions!(field.data,
                       field.boundary_conditions,
                       field.indices,
                       instantiated_location(field),
                       field.grid,
                       field.communication_buffers,
                       args...;
                       kwargs...)
end

# Sometimes we want to fill halo using `adapted` arguments, where the grid has
# been stripped from the architecture. For this reason we pass it explicitly
maybe_distributed_fill_halo_regions!(arch, args...; kwargs...) = fill_halo_regions!(args...; kwargs...)
maybe_distributed_fill_halo_regions!(arch::Distributed, args...; kwargs...) = distributed_fill_halo_regions!(arch, args...; kwargs...)

# Otherwise we recover the architecture from the (still distributed) grid.
fill_halo_regions!(c::OffsetArray, boundary_conditions, indices, loc, grid::DistributedGrid, args...; kwargs...) =
    distributed_fill_halo_regions!(architecture(grid), c, boundary_conditions, indices, loc, grid, args...; kwargs...)

fill_halo_regions!(c::OffsetArray, ::Nothing, indices, loc, grid::DistributedGrid, args...; kwargs...) = nothing

function distributed_fill_halo_regions!(arch, c, boundary_conditions, indices, loc, grid, buffers, args...; kwargs...)
    kernels!, bcs = get_boundary_kernels(boundary_conditions, c, grid, loc, indices)

    distributed_fill_halo_events!(c, values(kernels!), values(bcs), loc, arch, grid, buffers, args...; kwargs...)

    fill_corners!(c, arch.connectivity, indices, loc, arch, grid, buffers, args...; kwargs...)

    return nothing
end

@inline distributed_fill_halo_events!(c, ::Tuple{}, ::Tuple{}, loc, arch, grid, args...; kwargs...) = nothing

@inline function distributed_fill_halo_events!(c, kernels!::Tuple, bcs::Tuple, loc, arch, grid, args...; kwargs...)
    distributed_fill_halo_event!(c, first(kernels!), first(bcs), loc, arch, grid, args...; kwargs...)
    distributed_fill_halo_events!(c, Base.tail(kernels!), Base.tail(bcs), loc, arch, grid, args...; kwargs...)
    return nothing
end

@inline function complete_comm!(c, arch, grid, buffers, requests, async, side)

    # if `isnothing(requests)`, `fill_halo!` did not involve MPI passing
    if isnothing(requests)
        return nothing
    end

    # Syncronous MPI fill_halo_event!
    cooperative_waitall!(requests)

    recv_from_buffers!(c, buffers, grid, side)

    return nothing
end

# corner passing routine
function fill_corners!(c, connectivity, indices, loc, arch, grid, buffers, args...;
                       async=false, only_local_halos=false, kw...)

    # No corner filling needed!
    only_local_halos && return nothing

    # Skip corners entirely if no corner neighbors exist (avoids unnecessary sync_device!)
    isnothing(connectivity.southwest) && isnothing(connectivity.southeast) &&
    isnothing(connectivity.northwest) && isnothing(connectivity.northeast) && return nothing

    # This has to be synchronized!
    fill_send_buffers!(c, buffers, grid, Val(:corners))

    if async && (arch isa AsynchronousDistributed)
      async_corner_halo_comms(c, connectivity, indices, loc, arch, grid, buffers, args...; kw...)
    else
      sync_corner_halo_comms(c, connectivity, indices, loc, arch, grid, buffers, args...; kw...)
    end

    return nothing
end

function sync_corner_halo_comms(c, connectivity, indices, loc, arch, grid, buffers, args...; kw...)
  sync_device!(arch)
  requests = MPI.Request[]

  reqsw = fill_southwest_halo!(c, connectivity.southwest, indices, loc, arch, grid, buffers, buffers.southwest, args...; kw...)
  reqse = fill_southeast_halo!(c, connectivity.southeast, indices, loc, arch, grid, buffers, buffers.southeast, args...; kw...)
  reqnw = fill_northwest_halo!(c, connectivity.northwest, indices, loc, arch, grid, buffers, buffers.northwest, args...; kw...)
  reqne = fill_northeast_halo!(c, connectivity.northeast, indices, loc, arch, grid, buffers, buffers.northeast, args...; kw...)

  !isnothing(reqsw) && push!(requests, reqsw...)
  !isnothing(reqse) && push!(requests, reqse...)
  !isnothing(reqnw) && push!(requests, reqnw...)
  !isnothing(reqne) && push!(requests, reqne...)

  complete_comm!(c, arch, grid, buffers, requests, false, Val(:corners))

end

function async_corner_halo_comms(c, connectivity, indices, loc, arch, grid, buffers, args...; kw...)
  fill_event = record_event(arch)
  add_fill_event!(buffers)

  Threads.@spawn begin
    # Need to lock the channel to show we are waiting on send buffers
    sync_event(fill_event)

    reqsw = fill_southwest_halo!(c, connectivity.southwest, indices, loc, arch, grid, buffers, buffers.southwest, args...; kw...)
    reqse = fill_southeast_halo!(c, connectivity.southeast, indices, loc, arch, grid, buffers, buffers.southeast, args...; kw...)
    reqnw = fill_northwest_halo!(c, connectivity.northwest, indices, loc, arch, grid, buffers, buffers.northwest, args...; kw...)
    reqne = fill_northeast_halo!(c, connectivity.northeast, indices, loc, arch, grid, buffers, buffers.northeast, args...; kw...)

    reqs = MPI.Request[]

    !isnothing(reqsw) && push!(reqs, reqsw...)
    !isnothing(reqse) && push!(reqs, reqse...)
    !isnothing(reqnw) && push!(reqs, reqnw...)
    !isnothing(reqne) && push!(reqs, reqne...)

    add_comm_requests!(buffers, reqs)
    complete_fill_event!(buffers)

  end

end

cooperative_wait(req::MPI.Request)            = MPI.Waitall(req)
cooperative_waitall!(req::MPI.Request)            = MPI.Waitall(req)
cooperative_waitall!(req::Array{MPI.Request}) = MPI.Waitall(req)
function cooperative_waitall!(request_channel::Channel)
  # If there are no requests, skip the waitall
  # For distributed fields, use wait_for_comms to ensure correct behaviour
  if !isempty(request_channel)
    for req in request_channel
      cooperative_waitall!(req)
    end
  end
end

# Fallback: for serial boundary conditions fall back to `fill_halo_event!` but prune out the additional `buffers`
# argument used only for distributed halo-filling boundary conditions
distributed_fill_halo_event!(c, kernel!, bcs, loc, arch, grid, buffers, args...; kwargs...) = fill_halo_event!(c, kernel!, bcs, loc, grid, args...; kwargs...)

# There are two additional keyword arguments (with respect to serial `fill_halo_event!`s) that take an effect on `DistributedGrids`:
# - only_local_halos: if true, only the local halos are filled, i.e. corresponding to non-communicating boundary conditions
# - async: if true, ansynchronous MPI communication is enabled
function distributed_fill_halo_event!(c, kernel!::DistributedFillHalo, bcs, loc, arch, grid, buffers, args...;
                                      async = false, only_local_halos = false, kwargs...)

    only_local_halos && return nothing # No need to do anything here

    buffer_side = kernel!.side

    fill_send_buffers!(c, buffers, grid, buffer_side)
    fill_event = record_event(arch)
    add_fill_event!(buffers)

    if async && (arch isa AsynchronousDistributed)
      Threads.@spawn perform_comms(fill_event, c, kernel!, bcs, loc, arch, grid, buffers, args...)
    else
      perform_comms(fill_event, c, kernel!, bcs, loc, arch, grid, buffers, args...)
      # Need to synchronize communications
      wait_for_comms!(buffers)
      recv_from_buffers!(c, buffers, grid, kernel!.side)
    end

    return nothing
end

function perform_comms(fill_event, c, kernel!::DistributedFillHalo, bcs, loc, arch, grid, buffers, args...)
        sync_event(fill_event)

        requests = kernel!(c, bcs..., loc, grid, arch, buffers)
        add_comm_requests!(buffers, requests)
        complete_fill_event!(buffers)
end
#####
##### fill_$corner_halo! where corner = [:southwest, :southeast, :northwest, :northeast]
#####

for side in [:southwest, :southeast, :northwest, :northeast]
    fill_corner_halo! = Symbol("fill_$(side)_halo!")
    send_side_halo  = Symbol("send_$(side)_halo")
    recv_side_halo! = Symbol("recv_$(side)_halo!")

    @eval begin
        $fill_corner_halo!(c, corner, indices, loc, arch, grid, buffers, ::Nothing, args...; kwargs...) = nothing

        function $fill_corner_halo!(c, corner, indices, loc, arch, grid, buffers, sd, args...; kwargs...)
            child_arch = child_architecture(arch)
            local_rank = arch.local_rank

            recv_req = $recv_side_halo!(c, grid, arch, loc, local_rank, corner, buffers)
            send_req = $send_side_halo(c, grid, arch, loc, local_rank, corner, buffers)

            return [send_req, recv_req]
        end
    end
end

#####
##### Double-sided Distributed fill_halo!s
#####

function (::DistributedFillHalo{<:WestAndEast})(c, west_bc, east_bc, loc, grid, arch, buffers)
    @assert west_bc.condition.from == east_bc.condition.from  # Extra protection in case of bugs
    local_rank = west_bc.condition.from

    recv_req1 = recv_west_halo!(c, grid, arch, loc, local_rank, west_bc.condition.to, buffers)
    recv_req2 = recv_east_halo!(c, grid, arch, loc, local_rank, east_bc.condition.to, buffers)

    send_req1 = send_west_halo(c, grid, arch, loc, local_rank, west_bc.condition.to, buffers)
    send_req2 = send_east_halo(c, grid, arch, loc, local_rank, east_bc.condition.to, buffers)

    return [send_req1, send_req2, recv_req1, recv_req2]
end

function (::DistributedFillHalo{<:SouthAndNorth})(c, south_bc, north_bc, loc, grid, arch, buffers)
    @assert south_bc.condition.from == north_bc.condition.from  # Extra protection in case of bugs
    local_rank = south_bc.condition.from

    recv_req1 = recv_south_halo!(c, grid, arch, loc, local_rank, south_bc.condition.to, buffers)
    recv_req2 = recv_north_halo!(c, grid, arch, loc, local_rank, north_bc.condition.to, buffers)

    send_req1 = send_south_halo(c, grid, arch, loc, local_rank, south_bc.condition.to, buffers)
    send_req2 = send_north_halo(c, grid, arch, loc, local_rank, north_bc.condition.to, buffers)

    return [send_req1, send_req2, recv_req1, recv_req2]
end

#####
##### Single-sided Distributed fill_halo!s
#####

function (::DistributedFillHalo{<:West})(c, bc, loc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_req = recv_west_halo!(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    send_req = send_west_halo(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    return [send_req, recv_req]
end

function (::DistributedFillHalo{<:East})(c, bc, loc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_req = recv_east_halo!(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    send_req = send_east_halo(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    return [send_req, recv_req]
end

function (::DistributedFillHalo{<:South})(c, bc, loc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_req = recv_south_halo!(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    send_req = send_south_halo(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    return [send_req, recv_req]
end

function (::DistributedFillHalo{<:North})(c, bc, loc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_req = recv_north_halo!(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    send_req = send_north_halo(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    return [send_req, recv_req]
end

#####
##### No communication in the vertical direction
#####

(::DistributedFillHalo{<:BottomAndTop})(args...) = nothing
(::DistributedFillHalo{<:Bottom})(args...) = nothing
(::DistributedFillHalo{<:Top})(args...) = nothing

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
        function $send_side_halo(c, grid, arch, location, local_rank, rank_to_send_to, buffers)
            send_buffer = $get_side_send_buffer(c, grid, buffers, arch)
            send_tag = $side_send_tag(arch, grid, get_comm_tag(buffers.state),  location)

            @debug "Sending " * $side_str * " halo: local_rank=$local_rank, rank_to_send_to=$rank_to_send_to, send_tag=$send_tag"
            send_req = MPI.Isend(send_buffer, rank_to_send_to, send_tag, arch.communicator)

            return send_req
        end

        @inline $get_side_send_buffer(c, grid, buffers, arch) = buffers.$side.send
    end
end

#####
##### Receiving and filling halos
#####

for side in sides
    side_str = string(side)
    recv_side_halo! = Symbol("recv_$(side)_halo!")
    underlying_side_halo = Symbol("underlying_$(side)_halo")
    side_recv_tag = Symbol("$(side)_recv_tag")
    get_side_recv_buffer = Symbol("get_$(side)_recv_buffer")

    @eval begin
        function $recv_side_halo!(c, grid, arch, location, local_rank, rank_to_recv_from, buffers)
            recv_buffer = $get_side_recv_buffer(c, grid, buffers, arch)
            recv_tag = $side_recv_tag(arch, grid, get_comm_tag(buffers.state), location)

            @debug "Receiving " * $side_str * " halo: local_rank=$local_rank, rank_to_recv_from=$rank_to_recv_from, recv_tag=$recv_tag"
            recv_req = MPI.Irecv!(recv_buffer, rank_to_recv_from, recv_tag, arch.communicator)

            return recv_req
        end

        @inline $get_side_recv_buffer(c, grid, buffers, arch) = buffers.$side.recv
    end
end
