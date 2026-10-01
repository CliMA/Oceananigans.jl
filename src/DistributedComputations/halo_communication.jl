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

const ID_DIGITS   = 2

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

fill_halo_regions!(field::DistributedField, args::Vararg{Any, N}; kwargs...) where N =
    fill_halo_regions!(field.data,
                       field.boundary_conditions,
                       field.indices,
                       instantiated_location(field),
                       field.grid,
                       field.communication_buffers,
                       args...;
                       kwargs...)

# Sometimes we want to fill halo using `adapted` arguments, where the grid has
# been stripped from the architecture. For this reason we pass it explicitly
maybe_distributed_fill_halo_regions!(arch, args...; kwargs...) = fill_halo_regions!(args...; kwargs...)
function maybe_distributed_fill_halo_regions!(arch::Distributed, c, boundary_conditions, indices, loc, grid, buffers, args::Vararg{Any, N}; kwargs...) where N
    return distributed_fill_halo_regions!(arch, c, boundary_conditions, indices, loc, grid, buffers, args; kwargs...)
end

# Otherwise we recover the architecture from the (still distributed) grid.
function fill_halo_regions!(c::OffsetArray, boundary_conditions, indices, loc, grid::DistributedGrid, buffers, args::Vararg{Any, N}; kwargs...) where N
    return distributed_fill_halo_regions!(architecture(grid), c, boundary_conditions, indices, loc, grid, buffers, args; kwargs...)
end

fill_halo_regions!(c::OffsetArray, ::Nothing, indices, loc, grid::DistributedGrid, args...; kwargs...) = nothing

function distributed_fill_halo_regions!(arch, c, boundary_conditions, indices, loc, grid, buffers, args...; kwargs...)
    kernels!, bcs = get_boundary_kernels(boundary_conditions, c, grid, loc, indices)
    distributed_fill_halo_events!(c, values(kernels!), values(bcs), loc, arch, grid, buffers, args; kwargs...)
    fill_corners!(c, arch.connectivity, indices, loc, arch, grid, buffers; kwargs...)
    return nothing
end

@inline distributed_fill_halo_events!(c, ::Tuple{}, ::Tuple{}, loc, arch, grid, buffers, args; kwargs...) = nothing

@inline function distributed_fill_halo_events!(c, kernels!::Tuple, bcs::Tuple, loc, arch, grid, buffers, args; kwargs...)
    distributed_fill_halo_event!(c, first(kernels!), first(bcs), loc, arch, grid, buffers, args; kwargs...)
    distributed_fill_halo_events!(c, Base.tail(kernels!), Base.tail(bcs), loc, arch, grid, buffers, args; kwargs...)
    return nothing
end


# corner passing routine
function fill_corners!(c, connectivity, indices, loc, arch, grid, buffers; async=false, only_local_halos=false, kw...)

    # No corner filling needed!
    only_local_halos && return nothing

    # Skip corners entirely if no corner neighbors exist (avoids unnecessary sync_device!)
    isnothing(connectivity.southwest) && isnothing(connectivity.southeast) &&
    isnothing(connectivity.northwest) && isnothing(connectivity.northeast) && return nothing

    # This has to be synchronized!
    fill_send_buffers!(c, buffers, grid, Val(:corners))

    if async && (arch isa AsynchronousDistributed)
      async_corner_halo_comms(c, connectivity, indices, loc, arch, grid, buffers)
    else
      sync_corner_halo_comms(c, connectivity, indices, loc, arch, grid, buffers)
    end

    return nothing
end

function sync_corner_halo_comms(c, connectivity, indices, loc, arch, grid, buffers)
    sync_device!(arch)
    waitall_comms!(post_corner_requests!(c, connectivity, indices, loc, arch, grid, buffers))
    recv_from_buffers!(c, buffers, grid, Val(:corners))
    return nothing
end

function async_corner_halo_comms(c, connectivity, indices, loc, arch, grid, buffers)
    fill_event = record_event(arch)

    async_comms!(fill_event, buffers) do
        post_corner_requests!(c, connectivity, indices, loc, arch, grid, buffers)
    end

    return nothing
end

function post_corner_requests!(c, connectivity, indices, loc, arch, grid, buffers)
    fill_southwest_halo!(c, connectivity.southwest, indices, loc, arch, grid, buffers, buffers.southwest)
    fill_southeast_halo!(c, connectivity.southeast, indices, loc, arch, grid, buffers, buffers.southeast)
    fill_northwest_halo!(c, connectivity.northwest, indices, loc, arch, grid, buffers, buffers.northwest)
    fill_northeast_halo!(c, connectivity.northeast, indices, loc, arch, grid, buffers, buffers.northeast)
    requests = buffers.state.requests
    return (requests.southwest, requests.southeast, requests.northwest, requests.northeast)
end

# Post the MPI requests of `post_requests()`, which returns the requests it used, once `fill_event` is done, without waiting for them
# to complete; `wait_for_comms!` completes them later. With more than one Julia thread the requests are posted and progressed by a separate task,
# while with a single thread a separate task would compete with main, so the requests are posted right away by the main thread and completed by
# `wait_for_comms!`.
function async_comms!(post_requests, fill_event, buffers)
    if Threads.nthreads() > 1
        add_fill_event!(buffers)
        errormonitor(Threads.@spawn begin
            sync_event(fill_event)
            requests = post_requests()
            progress_comms!(requests)
            complete_fill_event!(buffers)
        end)
    else
        sync_event(fill_event)
        post_requests()
    end
    return nothing
end

# Keep testing the requests instead of deferring all progress to `Waitall` (for MPI implementations without asynchronous progress)
progress_comms!(requests::Tuple) = foreach(progress_comms!, requests)
progress_comms!(::Nothing) = nothing

function progress_comms!(requests::MPI.UnsafeMultiRequest)
    while !MPI.Testall(requests)
        yield()
    end
    return nothing
end

waitall_comms!(requests::Tuple) = foreach(waitall_comms!, requests)
waitall_comms!(::Nothing) = nothing
waitall_comms!(requests::MPI.UnsafeMultiRequest) = MPI.Waitall(requests)

# Fallback: for serial boundary conditions fall back to `fill_halo_event!` but prune out the additional `buffers`
# argument used only for distributed halo-filling boundary conditions
distributed_fill_halo_event!(c, kernel!, bcs, loc, arch, grid, buffers, args; kwargs...) = fill_halo_event!(c, kernel!, bcs, loc, grid, args...; kwargs...)

# There are two additional keyword arguments (with respect to serial `fill_halo_event!`s) that take an effect on `DistributedGrids`:
# - only_local_halos: if true, only the local halos are filled, i.e. corresponding to non-communicating boundary conditions
# - async: if true, ansynchronous MPI communication is enabled
function distributed_fill_halo_event!(c, kernel!::DistributedFillHalo, bcs, loc, arch, grid, buffers, args;
                                      async = false, only_local_halos = false, kwargs...)

    only_local_halos && return nothing # No need to do anything here

    buffer_side = kernel!.side

    fill_send_buffers!(c, buffers, grid, buffer_side)
    fill_event = record_event(arch)

    if async && (arch isa AsynchronousDistributed)
        async_comms!(fill_event, buffers) do
            kernel!(c, bcs..., loc, grid, arch, buffers)
        end
    else
        sync_event(fill_event)
        requests = kernel!(c, bcs..., loc, grid, arch, buffers)
        waitall_comms!(requests)
        wait_for_comms!(buffers)
        recv_from_buffers!(c, buffers, grid, buffer_side)
    end

    return nothing
end

#####
##### fill_$corner_halo! where corner = [:southwest, :southeast, :northwest, :northeast]
#####

for side in [:southwest, :southeast, :northwest, :northeast]
    fill_corner_halo! = Symbol("fill_$(side)_halo!")
    send_side_halo  = Symbol("send_$(side)_halo")
    recv_side_halo! = Symbol("recv_$(side)_halo!")

    @eval begin
        $fill_corner_halo!(c, corner, indices, loc, arch, grid, buffers, ::Nothing) = nothing

        function $fill_corner_halo!(c, corner, indices, loc, arch, grid, buffers, sd)
            local_rank = arch.local_rank

            $recv_side_halo!(c, grid, arch, loc, local_rank, corner, buffers)
            $send_side_halo(c, grid, arch, loc, local_rank, corner, buffers)

            return nothing
        end
    end
end

#####
##### Double-sided Distributed fill_halo!s
#####

function (::DistributedFillHalo{<:WestAndEast})(c, west_bc, east_bc, loc, grid, arch, buffers)
    @assert west_bc.condition.from == east_bc.condition.from  # Extra protection in case of bugs
    local_rank = west_bc.condition.from

    recv_west_halo!(c, grid, arch, loc, local_rank, west_bc.condition.to, buffers)
    recv_east_halo!(c, grid, arch, loc, local_rank, east_bc.condition.to, buffers)

    send_west_halo(c, grid, arch, loc, local_rank, west_bc.condition.to, buffers)
    send_east_halo(c, grid, arch, loc, local_rank, east_bc.condition.to, buffers)

    return (buffers.state.requests.west, buffers.state.requests.east)
end

function (::DistributedFillHalo{<:SouthAndNorth})(c, south_bc, north_bc, loc, grid, arch, buffers)
    @assert south_bc.condition.from == north_bc.condition.from  # Extra protection in case of bugs
    local_rank = south_bc.condition.from

    recv_south_halo!(c, grid, arch, loc, local_rank, south_bc.condition.to, buffers)
    recv_north_halo!(c, grid, arch, loc, local_rank, north_bc.condition.to, buffers)

    send_south_halo(c, grid, arch, loc, local_rank, south_bc.condition.to, buffers)
    send_north_halo(c, grid, arch, loc, local_rank, north_bc.condition.to, buffers)

    return (buffers.state.requests.south, buffers.state.requests.north)
end

#####
##### Single-sided Distributed fill_halo!s
#####

function (::DistributedFillHalo{<:West})(c, bc, loc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_west_halo!(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    send_west_halo(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    return (buffers.state.requests.west,)
end

function (::DistributedFillHalo{<:East})(c, bc, loc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_east_halo!(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    send_east_halo(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    return (buffers.state.requests.east,)
end

function (::DistributedFillHalo{<:South})(c, bc, loc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_south_halo!(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    send_south_halo(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    return (buffers.state.requests.south,)
end

function (::DistributedFillHalo{<:North})(c, bc, loc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_north_halo!(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    send_north_halo(c, grid, arch, loc, local_rank, bc.condition.to, buffers)
    return (buffers.state.requests.north,)
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
            MPI.Isend(send_buffer, rank_to_send_to, send_tag, arch.communicator, buffers.state.requests.$side[1])

            return nothing
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
            MPI.Irecv!(recv_buffer, rank_to_recv_from, recv_tag, arch.communicator, buffers.state.requests.$side[2])

            return nothing
        end

        @inline $get_side_recv_buffer(c, grid, buffers, arch) = buffers.$side.recv
    end
end
