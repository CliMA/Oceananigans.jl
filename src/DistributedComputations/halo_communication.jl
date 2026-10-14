using KernelAbstractions: @kernel, @index

using Oceananigans.Fields: instantiated_location

using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions:
    DistributedFillHalo,
    get_boundary_kernels

import Oceananigans.BoundaryConditions: fill_halo_event!, fill_halo_regions!

# Each field gets `halo_tag_slots` consecutive MPI tags, one per message exchanged with a given rank
const halo_tag_slots = 32

#####
##### Filling halos for halo communication BCs
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
    return nothing
end

@inline distributed_fill_halo_events!(c, ::Tuple{}, ::Tuple{}, loc, arch, grid, buffers, args; kwargs...) = nothing

@inline function distributed_fill_halo_events!(c, kernels!::Tuple, bcs::Tuple, loc, arch, grid, buffers, args; kwargs...)
    distributed_fill_halo_event!(c, first(kernels!), first(bcs), loc, arch, grid, buffers, args; kwargs...)
    distributed_fill_halo_events!(c, Base.tail(kernels!), Base.tail(bcs), loc, arch, grid, buffers, args; kwargs...)
    return nothing
end

# Fallback: for serial boundary conditions fall back to `fill_halo_event!` but prune out the additional `buffers`
# argument used only for distributed halo-filling boundary conditions
distributed_fill_halo_event!(c, kernel!, bcs, loc, arch, grid, buffers, args; kwargs...) = fill_halo_event!(c, kernel!, bcs, loc, grid, args...; kwargs...)

# There are two additional keyword arguments (with respect to serial `fill_halo_event!`s) that take an effect on `DistributedGrids`:
# - only_local_halos: if true, only the local halos are filled, i.e. corresponding to non-communicating boundary conditions
# - async: if true, ansynchronous MPI communication is enabled
function distributed_fill_halo_event!(c, kernel!::DistributedFillHalo, bcs, loc, arch, grid, buffers, args;
                                      async = false, only_local_halos = false, kwargs...)

    only_local_halos && return nothing # No need to do anything here

    side = kernel!.side

    fill_send_buffers!(c, buffers, side)
    record_event!(buffers.state.pack_event, arch)

    if async && (arch isa AsynchronousDistributed)
        post_messages!(arch, buffers, side)
    else
        # The main thread waits for these messages anyway
        post_messages!(arch, posted_by_main_thread(buffers), side)
        wait_for_messages!(buffers)
        recv_from_buffers!(c, buffers, side)
    end

    return nothing
end

function post_messages!(arch, buffers, side)
    for messages in side_messages(buffers, side)
        for segment in messages.receives
            irecv!(buffers.state, segment.buffer, segment.rank, segment.tag, arch.communicator, segment.requests)
        end
        for segment in messages.sends
            isend!(buffers.state, segment.buffer, segment.rank, segment.tag, arch.communicator, segment.requests)
        end
    end
    return nothing
end
