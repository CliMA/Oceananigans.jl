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

const sides = (:west, :east, :south, :north, :southwest, :southeast, :northwest, :northeast)
const side_id = Dict(side => n-1 for (n, side) in enumerate(sides))

const opposite_side = Dict(
    :west => :east,
    :east => :west,
    :south => :north,
    :north => :south,
    :southwest => :northeast,
    :southeast => :northwest,
    :northwest => :southeast,
    :northeast => :southwest,
)

# Each field gets `halo_tag_slots` consecutive MPI tags: one per side (0-7) and three for the tripolar fold (8-10)
const halo_tag_slots = 11

for side in sides
    send_tag_fn_name = Symbol("$(side)_send_tag")
    recv_tag_fn_name = Symbol("$(side)_recv_tag")
    send_slot = side_id[side]
    recv_slot = side_id[opposite_side[side]]
    @eval begin
        $send_tag_fn_name(arch, grid, field_tag) = Int(halo_tag_slots * field_tag + $send_slot)
        $recv_tag_fn_name(arch, grid, field_tag) = Int(halo_tag_slots * field_tag + $recv_slot)
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

function distributed_fill_halo_regions!(arch, c, boundary_conditions, indices, loc, grid, buffers, args; only_local_halos = false, kwargs...)
    # Complete an asynchronous fill of this field still in flight before its send buffers and requests are reused
    only_local_halos || wait_for_messages!(buffers)

    kernels!, bcs = get_boundary_kernels(boundary_conditions, c, grid, loc, indices)
    distributed_fill_halo_events!(c, values(kernels!), values(bcs), loc, arch, grid, buffers, args; only_local_halos, kwargs...)
    fill_corners!(c, arch.connectivity, arch, grid, buffers; only_local_halos, kwargs...)
    return nothing
end

@inline distributed_fill_halo_events!(c, ::Tuple{}, ::Tuple{}, loc, arch, grid, buffers, args; kwargs...) = nothing

@inline function distributed_fill_halo_events!(c, kernels!::Tuple, bcs::Tuple, loc, arch, grid, buffers, args; kwargs...)
    distributed_fill_halo_event!(c, first(kernels!), first(bcs), loc, arch, grid, buffers, args; kwargs...)
    distributed_fill_halo_events!(c, Base.tail(kernels!), Base.tail(bcs), loc, arch, grid, buffers, args; kwargs...)
    return nothing
end


# corner passing routine
function fill_corners!(c, connectivity, arch, grid, buffers; async=false, only_local_halos=false, kw...)

    # No corner filling needed!
    only_local_halos && return nothing

    # Skip corners entirely if no corner neighbors exist (avoids unnecessary sync_device!)
    isnothing(connectivity.southwest) && isnothing(connectivity.southeast) &&
    isnothing(connectivity.northwest) && isnothing(connectivity.northeast) && return nothing

    fill_send_buffers!(c, buffers, grid, Val(:corners))

    if async && (arch isa AsynchronousDistributed)
        async_corner_halo_comms(connectivity, arch, grid, buffers)
    else
        sync_corner_halo_comms(c, connectivity, arch, grid, buffers)
    end

    return nothing
end

function sync_corner_halo_comms(c, connectivity, arch, grid, buffers)
    sync_device!(arch)
    post_corner_messages!(connectivity, arch, grid, posted_by_main_thread(buffers))
    wait_for_messages!(buffers)
    recv_from_buffers!(c, buffers, grid, Val(:corners))
    return nothing
end

function async_corner_halo_comms(connectivity, arch, grid, buffers)
    record_event!(buffers.state.pack_event, arch)
    post_corner_messages!(connectivity, arch, grid, buffers)
    return nothing
end

function post_corner_messages!(connectivity, arch, grid, buffers)
    fill_southwest_halo!(connectivity.southwest, arch, grid, buffers, buffers.southwest)
    fill_southeast_halo!(connectivity.southeast, arch, grid, buffers, buffers.southeast)
    fill_northwest_halo!(connectivity.northwest, arch, grid, buffers, buffers.northwest)
    fill_northeast_halo!(connectivity.northeast, arch, grid, buffers, buffers.northeast)
    return nothing
end

waitall_requests!(requests::Tuple) = foreach(waitall_requests!, requests)
waitall_requests!(::Nothing) = nothing
waitall_requests!(requests::MPI.UnsafeMultiRequest) = MPI.Waitall(requests)

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
    record_event!(buffers.state.pack_event, arch)

    if async && (arch isa AsynchronousDistributed)
        kernel!(bcs..., grid, arch, buffers)
    else
        # The main thread waits for these messages anyway
        kernel!(bcs..., grid, arch, posted_by_main_thread(buffers))
        wait_for_messages!(buffers)
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
        $fill_corner_halo!(corner, arch, grid, buffers, ::Nothing) = nothing

        function $fill_corner_halo!(corner, arch, grid, buffers, _)
            local_rank = arch.local_rank

            $recv_side_halo!(grid, arch, local_rank, corner, buffers)
            $send_side_halo(grid, arch, local_rank, corner, buffers)

            return nothing
        end
    end
end

#####
##### Double-sided Distributed fill_halo!s
#####

function (::DistributedFillHalo{<:WestAndEast})(west_bc, east_bc, grid, arch, buffers)
    @assert west_bc.condition.from == east_bc.condition.from  # Extra protection in case of bugs
    local_rank = west_bc.condition.from

    recv_west_halo!(grid, arch, local_rank, west_bc.condition.to, buffers)
    recv_east_halo!(grid, arch, local_rank, east_bc.condition.to, buffers)

    send_west_halo(grid, arch, local_rank, west_bc.condition.to, buffers)
    send_east_halo(grid, arch, local_rank, east_bc.condition.to, buffers)

    return nothing
end

function (::DistributedFillHalo{<:SouthAndNorth})(south_bc, north_bc, grid, arch, buffers)
    @assert south_bc.condition.from == north_bc.condition.from  # Extra protection in case of bugs
    local_rank = south_bc.condition.from

    recv_south_halo!(grid, arch, local_rank, south_bc.condition.to, buffers)
    recv_north_halo!(grid, arch, local_rank, north_bc.condition.to, buffers)

    send_south_halo(grid, arch, local_rank, south_bc.condition.to, buffers)
    send_north_halo(grid, arch, local_rank, north_bc.condition.to, buffers)

    return nothing
end

#####
##### Single-sided Distributed fill_halo!s
#####

function (::DistributedFillHalo{<:West})(bc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_west_halo!(grid, arch, local_rank, bc.condition.to, buffers)
    send_west_halo(grid, arch, local_rank, bc.condition.to, buffers)
    return nothing
end

function (::DistributedFillHalo{<:East})(bc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_east_halo!(grid, arch, local_rank, bc.condition.to, buffers)
    send_east_halo(grid, arch, local_rank, bc.condition.to, buffers)
    return nothing
end

function (::DistributedFillHalo{<:South})(bc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_south_halo!(grid, arch, local_rank, bc.condition.to, buffers)
    send_south_halo(grid, arch, local_rank, bc.condition.to, buffers)
    return nothing
end

function (::DistributedFillHalo{<:North})(bc, grid, arch, buffers)
    local_rank = bc.condition.from
    recv_north_halo!(grid, arch, local_rank, bc.condition.to, buffers)
    send_north_halo(grid, arch, local_rank, bc.condition.to, buffers)
    return nothing
end

#####
##### No communication in the vertical direction
#####

(::DistributedFillHalo{<:BottomAndTop})(args...) = nothing
(::DistributedFillHalo{<:Bottom})(args...) = nothing
(::DistributedFillHalo{<:Top})(args...) = nothing

#####
##### Sending and receiving halos
#####

for side in sides
    side_str = string(side)
    send_side_halo = Symbol("send_$(side)_halo")
    recv_side_halo! = Symbol("recv_$(side)_halo!")
    side_send_tag = Symbol("$(side)_send_tag")
    side_recv_tag = Symbol("$(side)_recv_tag")

    @eval begin
        function $send_side_halo(grid, arch, local_rank, rank_to_send_to, buffers)
            send_buffer = buffers.$side.send
            send_tag = $side_send_tag(arch, grid, buffers.state.tag)

            @debug "Sending " * $side_str * " halo: local_rank=$local_rank, rank_to_send_to=$rank_to_send_to, send_tag=$send_tag"
            isend!(buffers.state, send_buffer, rank_to_send_to, send_tag, arch.communicator, buffers.state.requests.$side)

            return nothing
        end

        function $recv_side_halo!(grid, arch, local_rank, rank_to_recv_from, buffers)
            recv_buffer = buffers.$side.recv
            recv_tag = $side_recv_tag(arch, grid, buffers.state.tag)

            @debug "Receiving " * $side_str * " halo: local_rank=$local_rank, rank_to_recv_from=$rank_to_recv_from, recv_tag=$recv_tag"
            irecv!(buffers.state, recv_buffer, rank_to_recv_from, recv_tag, arch.communicator, buffers.state.requests.$side)

            return nothing
        end
    end
end
