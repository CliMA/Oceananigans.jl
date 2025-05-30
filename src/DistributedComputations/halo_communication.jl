using KernelAbstractions: @kernel, @index

using Oceananigans.Fields: reduced_dimensions, 
                           instantiated_location,
                           fill_reduced_field_halos!

import Oceananigans.Fields: tupled_fill_halo_regions!

using Oceananigans.BoundaryConditions:
    fill_halo_size,
    fill_halo_offset,
    permute_boundary_conditions,
    fill_open_boundary_regions!,
    PBCT, DCBCT # tuples

import Oceananigans.BoundaryConditions:
    fill_halo_regions!, fill_first, fill_halo_event!,
    fill_west_halo!, fill_east_halo!, fill_south_halo!,
    fill_north_halo!, fill_bottom_halo!, fill_top_halo!,
    fill_west_and_east_halo!,
    fill_south_and_north_halo!,
    fill_bottom_and_top_halo!

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

# Functions that return unique send and recv MPI tags for each side, field location
# keeping into account the possibility of asynchronous communication.
# the MPI tag is an integer with:
#   digit 1-2: an counter which keeps track of how many communications are live. The counter is stored in `arch.mpi_tag`
#   digit 3-4: a unique identifier for the field's location that goes from 0 - 26 (see `loc_id`)
#   digit 5: the side we send / recieve from

for side in sides
    side_str = string(side)
    send_tag_fn_name = Symbol("$(side)_send_tag")
    recv_tag_fn_name = Symbol("$(side)_recv_tag")
    @eval begin
        function $send_tag_fn_name(arch, grid, location)
            field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
            loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
            side_digit = string(side_id[Symbol($side_str)])
            return parse(Int, field_id * loc_digit * side_digit)
        end

        function $recv_tag_fn_name(arch, grid, location)
            field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
            loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
            side_digit = string(side_id[opposite_side[Symbol($side_str)]])
            return parse(Int, field_id * loc_digit * side_digit)
        end
    end
end

#####
##### Filling halos for halo communication boundary conditions
#####

function tupled_fill_halo_regions!(fields, grid::DistributedGrid, args...; kwargs...)
    not_reduced_fields = fill_reduced_field_halos!(fields, args...; kwargs)

    for field in not_reduced_fields
        # Make sure we are filling a `Field` type.
        field isa Field && fill_halo_regions!(field, args...; kwargs...)
    end
end

function fill_halo_regions!(field::DistributedField, args...; kwargs...)
    reduced_dims = reduced_dimensions(field)

    return fill_halo_regions!(field.data,
                              field.boundary_conditions,
                              field.indices,
                              instantiated_location(field),
                              field.grid,
                              field.communication_buffers,
                              args...;
                              reduced_dimensions = reduced_dims,
                              kwargs...)
end

function fill_halo_regions!(c::OffsetArray, bcs, indices, loc, grid::DistributedGrid, buffers, args...;
                            fill_boundary_normal_velocities=true, kwargs...)

    if fill_boundary_normal_velocities
        fill_open_boundary_regions!(c, bcs, indices, loc, grid, buffers, args...; kwargs...)
    end

    arch = architecture(grid)
    fill_halos!, bcs = permute_boundary_conditions(bcs)
    number_of_tasks  = length(fill_halos!)
    outstanding_requests = length(arch.mpi_requests)

    for task = 1:number_of_tasks
        fill_halo_event!(c, fill_halos![task], bcs[task], indices, loc, arch, grid, buffers, args...; kwargs...)
    end

    fill_corners!(c, arch.connectivity, indices, loc, arch, grid, buffers, args...; kwargs...)

    # We increment the request counter only if we have actually initiated the MPI communication.
    # This is the case only if at least one of the boundary conditions is a distributed communication
    # boundary condition (DCBCT) _and_ the `only_local_halos` keyword argument is false.
    if length(arch.mpi_requests) > outstanding_requests
        arch.mpi_tag[] += 1
    end

    return nothing
end

@inline function pool_requests_or_complete_comm!(c, arch, grid, buffers, requests, async, side)

    # if `isnothing(requests)`, `fill_halo!` did not involve MPI passing
    if isnothing(requests)
        return nothing
    end

    # Overlapping communication and computation, store requests in a `MPI.Request`
    # pool to be waited upon later on when halos are required.
    if async && (arch isa AsynchronousDistributed)
        push!(arch.mpi_requests, requests...)
        return nothing
    end

    # Syncronous MPI fill_halo_event!
    cooperative_waitall!(requests)

    # Reset MPI tag
    arch.mpi_tag[] -= arch.mpi_tag[]
    recv_from_buffers!(c, buffers, grid, Val(side))

    return nothing
end

# corner passing routine
function fill_corners!(c, connectivity, indices, loc, arch, grid, buffers, args...;
                       async=false, only_local_halos=false, kw...)

    # No corner filling needed!
    only_local_halos && return nothing

    # This has to be synchronized!
    fill_send_buffers!(c, buffers, grid, Val(:corners))
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

    pool_requests_or_complete_comm!(c, arch, grid, buffers, requests, async, :corners)

    return nothing
end

@inline communication_side(::Val{fill_west_and_east_halo!})   = :west_and_east
@inline communication_side(::Val{fill_south_and_north_halo!}) = :south_and_north
@inline communication_side(::Val{fill_bottom_and_top_halo!})  = :bottom_and_top
@inline communication_side(::Val{fill_west_halo!})   = :west
@inline communication_side(::Val{fill_east_halo!})   = :east
@inline communication_side(::Val{fill_south_halo!})  = :south
@inline communication_side(::Val{fill_north_halo!})  = :north
@inline communication_side(::Val{fill_bottom_halo!}) = :bottom
@inline communication_side(::Val{fill_top_halo!})    = :top

cooperative_wait(req::MPI.Request)            = MPI.Waitall(req)
cooperative_waitall!(req::Array{MPI.Request}) = MPI.Waitall(req)

# There are two additional keyword arguments (with respect to serial `fill_halo_event!`s) that take an effect on `DistributedGrids`:
# - only_local_halos: if true, only the local halos are filled, i.e. corresponding to non-communicating boundary conditions
# - async: if true, ansynchronous MPI communication is enabled
function fill_halo_event!(c, fill_halos!, bcs, indices, loc, arch, grid::DistributedGrid, buffers, args...;
                          async = false, only_local_halos = false, kwargs...)

    buffer_side = communication_side(Val(fill_halos!))

    if !only_local_halos # Then we need to fill the `send` buffers
        fill_send_buffers!(c, buffers, grid, Val(buffer_side))
    end

    # Calculate size and offset of the fill_halo kernel
    # We assume that the kernel size is the same for west and east boundaries,
    # south and north boundaries and bottom and top boundaries
    size   = fill_halo_size(c, fill_halos!, indices, bcs[1], loc, grid)
    offset = fill_halo_offset(size, fill_halos!, indices)

    requests = fill_halos!(c, bcs..., size, offset, loc, arch, grid, buffers, args...; only_local_halos, kwargs...)
    pool_requests_or_complete_comm!(c, arch, grid, buffers, requests, async, buffer_side)

    return nothing
end

#####
##### fill_$corner_halo! where corner = [:southwest, :southeast, :northwest, :northeast]
#####

for side in [:southwest, :southeast, :northwest, :northeast]
    fill_corner_halo! = Symbol("fill_$(side)_halo!")
    send_side_halo  = Symbol("send_$(side)_halo")
    recv_and_fill_side_halo! = Symbol("recv_and_fill_$(side)_halo!")

    @eval begin
        $fill_corner_halo!(c, corner, indices, loc, arch, grid, buffers, ::Nothing, args...; kwargs...) = nothing

        function $fill_corner_halo!(c, corner, indices, loc, arch, grid, buffers, sd, args...; kwargs...)
            child_arch = child_architecture(arch)
            local_rank = arch.local_rank

            recv_req = $recv_and_fill_side_halo!(c, grid, arch, loc, local_rank, corner, buffers)
            send_req = $send_side_halo(c, grid, arch, loc, local_rank, corner, buffers)

            return [send_req, recv_req]
        end
    end
end

#####
##### Double-sided Distributed fill_halo!s
#####

for (side, opposite_side) in zip([:west, :south], [:east, :north])
    fill_both_halo! = Symbol("fill_$(side)_and_$(opposite_side)_halo!")
    send_side_halo  = Symbol("send_$(side)_halo")
    send_opposite_side_halo  = Symbol("send_$(opposite_side)_halo")
    recv_and_fill_side_halo! = Symbol("recv_and_fill_$(side)_halo!")
    recv_and_fill_opposite_side_halo! = Symbol("recv_and_fill_$(opposite_side)_halo!")

    @eval begin
        function $fill_both_halo!(c, bc_side::DCBCT, bc_opposite_side::DCBCT, size, offset, loc, arch::Distributed,
                                  grid::DistributedGrid, buffers, args...; only_local_halos = false, kwargs...)

            only_local_halos && return nothing

            sync_device!(arch)

            @assert bc_side.condition.from == bc_opposite_side.condition.from  # Extra protection in case of bugs
            local_rank = bc_side.condition.from

            recv_req1 = $recv_and_fill_side_halo!(c, grid, arch, loc, local_rank, bc_side.condition.to, buffers)
            recv_req2 = $recv_and_fill_opposite_side_halo!(c, grid, arch, loc, local_rank, bc_opposite_side.condition.to, buffers)

            send_req1 = $send_side_halo(c, grid, arch, loc, local_rank, bc_side.condition.to, buffers)
            send_req2 = $send_opposite_side_halo(c, grid, arch, loc, local_rank, bc_opposite_side.condition.to, buffers)

            return [send_req1, send_req2, recv_req1, recv_req2]
        end
    end
end

#####
##### Single-sided Distributed fill_halo!s
#####

for side in [:west, :east, :south, :north]
    fill_side_halo! = Symbol("fill_$(side)_halo!")
    send_side_halo  = Symbol("send_$(side)_halo")
    recv_and_fill_side_halo! = Symbol("recv_and_fill_$(side)_halo!")

    @eval begin
        function $fill_side_halo!(c, bc_side::DCBCT, size, offset, loc, arch::Distributed, grid::DistributedGrid,
                                 buffers, args...; only_local_halos = false, kwargs...)

            only_local_halos && return nothing

            sync_device!(arch)

            child_arch = child_architecture(arch)
            local_rank = bc_side.condition.from

            recv_req = $recv_and_fill_side_halo!(c, grid, arch, loc, local_rank, bc_side.condition.to, buffers)
            send_req = $send_side_halo(c, grid, arch, loc, local_rank, bc_side.condition.to, buffers)

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
        function $send_side_halo(c, grid, arch, location, local_rank, rank_to_send_to, buffers)
            send_buffer = $get_side_send_buffer(c, grid, buffers, arch)
            send_tag = $side_send_tag(arch, grid, location)

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
    recv_and_fill_side_halo! = Symbol("recv_and_fill_$(side)_halo!")
    underlying_side_halo = Symbol("underlying_$(side)_halo")
    side_recv_tag = Symbol("$(side)_recv_tag")
    get_side_recv_buffer = Symbol("get_$(side)_recv_buffer")

    @eval begin
        function $recv_and_fill_side_halo!(c, grid, arch, location, local_rank, rank_to_recv_from, buffers)
            recv_buffer = $get_side_recv_buffer(c, grid, buffers, arch)
            recv_tag = $side_recv_tag(arch, grid, location)

            @debug "Receiving " * $side_str * " halo: local_rank=$local_rank, rank_to_recv_from=$rank_to_recv_from, recv_tag=$recv_tag"
            recv_req = MPI.Irecv!(recv_buffer, rank_to_recv_from, recv_tag, arch.communicator)

            return recv_req
        end

        @inline $get_side_recv_buffer(c, grid, buffers, arch) = buffers.$side.recv
    end
end
