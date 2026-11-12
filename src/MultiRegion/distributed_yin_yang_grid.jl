using MPI: MPI
using Oceananigans.DistributedComputations: AsynchronousDistributed, CommunicationState, Distributed, halo_tag_slots,
    irecv!, isend!, new_event, next_field_tag!, posted_by_main_thread, progress_mailbox, record_event!, wait_for_messages!
using Oceananigans.Models.NonhydrostaticModels: NonhydrostaticModels
using Oceananigans.Utils: worksize

#####
##### A YinYangGrid on a `Distributed` architecture, whose `Partition` splits both panels or whose
##### `MultiRegionPartition(yin, yang)` splits each panel: each rank holds one piece of one panel, piece r on rank r - 1,
##### on the child architecture. Every rank builds the halo tables of every piece, so a
##### rank knows which of its values each other rank needs: the halos of a piece are pushed by their donors, one
##### message per donor and phase (copies, then interpolations, which may read the copied halos).
#####

# The ranks of a `Distributed` architecture partitioned by a `MultiRegionPartition`
Base.size(p::MultiRegionPartition) = (length(p), 1, 1)

function yin_yang_partition(arch::Distributed, partition, panel_size)
    p = yin_yang_partition(arch.partition, panel_size)
    Nr = MPI.Comm_size(arch.communicator)
    length(p) == Nr || throw(ArgumentError("A YinYangGrid with $(summary(p)) needs $(length(p)) MPI ranks; got $Nr."))
    return p
end

yin_yang_local_regions(arch::Distributed, layout) = (arch.local_rank + 1,)

# The halo pieces that this rank's piece `r` donates to the other pieces, as Vectors of `(receiver, table)`
struct YinYangExchange{S, V}
    scalars :: S # (; center_center, face_center, center_face, face_face) of (; copies, interpolations)
    vector :: V  # (; u_copies, v_copies, u, v)
end

function yin_yang_exchange(arch::Distributed, halos)
    r = arch.local_rank + 1
    donated(pieces) = eltype(pieces(first(halos)))[(q, table) for q in eachindex(halos) for (d, table) in pieces(halos[q]) if d == r]
    scalar(name) = (; copies = donated(h -> getfield(h.scalars, name).copies),
                      interpolations = donated(h -> getfield(h.scalars, name).interpolations))
    scalars = NamedTuple{(:center_center, :face_center, :center_face, :face_face)}(
        map(scalar, (:center_center, :face_center, :center_face, :face_face)))
    vector = NamedTuple{(:u_copies, :v_copies, :u, :v)}(
        map(name -> donated(h -> getfield(h.vector, name)), (:u_copies, :v_copies, :u, :v)))
    return YinYangExchange(scalars, vector)
end

Grids.on_architecture(arch, e::YinYangExchange) =
    YinYangExchange(map(s -> map(p -> on_architecture_pieces(arch, p), s), e.scalars),
                    map(p -> on_architecture_pieces(arch, p), e.vector))

@inline region_halo(s::NamedTuple, ::Center, ::Center) = s.center_center
@inline region_halo(s::NamedTuple, ::Face,   ::Center) = s.face_center
@inline region_halo(s::NamedTuple, ::Center, ::Face)   = s.center_face
@inline region_halo(s::NamedTuple, ::Face,   ::Face)   = s.face_face

const DistributedYinYangGrid = MultiRegionGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:SubregionPartition{<:YinYangPartition},
                                               <:Any, <:Any, <:Distributed}

const DistributedYinYangGridOfSomeKind = Union{DistributedYinYangGrid,
    ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:DistributedYinYangGrid, <:Any, <:Any, <:Any, <:Distributed}}

const DistributedYinYangField = Field{<:Any, <:Any, <:Any, <:Any, <:DistributedYinYangGridOfSomeKind}

# A multi-region grid on a `Distributed` architecture holds the regions of its rank and fills their halos itself,
# so its fields are built as on any multi-region grid
const DistributedMultiRegionGrids = Union{MultiRegionGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Distributed},
    ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:MultiRegionGrid, <:Any, <:Any, <:Any, <:Distributed}}

# The tendencies of a piece are computed on its serial region grid with the `Distributed` architecture of the model.
# Every side of a piece has halos filled by the exchange, so the cells that do not read them are `H` cells away from each side
const SerialGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractSerialArchitecture}

function Models.interior_tendency_kernel_parameters(::AsynchronousDistributed, grid::SerialGrid)
    Hx, Hy, _ = halo_size(grid)
    Wx, Wy, Wz = worksize(grid)
    return KernelParameters((Wx - 2Hx, Wy - 2Hy, Wz), (Hx, Hy, 0))
end

NonhydrostaticModels.buffer_parameters(parameters, ::SerialGrid, ::Distributed) = map(p -> KernelParameters(p...), parameters)

Fields.Field(loc::Tuple{<:LX, <:LY, <:LZ}, grid::DistributedMultiRegionGrids, data, bcs, indices::Tuple, op, status) where {LX, LY, LZ} =
    invoke(Field, Tuple{Tuple{LX, LY, LZ}, AbstractGrid, Any, Any, Any, Any, Any}, loc, grid, data, bcs, indices, op, status)

#####
##### Exchanging halo pieces
#####

first_table(tables::Tuple) = first(tables)
first_table(table) = table

# The halo messages of a field, with one [send, receive] pair of requests per donor or receiver piece, and the
# unpacking of the pieces it receives, left to `finish_pieces!`
struct YinYangCommunication{S}
    state :: S
    unpacks :: Vector{Any}
end

Adapt.adapt_structure(to, ::YinYangCommunication) = nothing
Grids.on_architecture(arch, communication::YinYangCommunication) = communication

piece_lists(halos) = (Iterators.flatmap(s -> (s.copies, s.interpolations), halos.scalars)...,
                      (getproperty(halos.vector, name) for name in (:u_copies, :v_copies, :u, :v))...)

function Fields.communication_buffers(grid::DistributedYinYangGridOfSomeKind, data, ::MultiRegionObject)
    arch = architecture(grid)
    connectivity = yin_yang_connectivity(grid)
    Nmessages = maximum(length, (piece_lists(getregion(connectivity.region_halos, 1))..., piece_lists(connectivity.exchange)...))
    requests = Tuple(MPI.UnsafeMultiRequest(2) for _ in 1:Nmessages)
    tag = UInt64(mod(next_field_tag!(arch), MPI.tag_ub() ÷ halo_tag_slots))
    pack_event = new_event(arch)
    state = CommunicationState(Threads.Atomic{UInt64}(0), requests, tag, pack_event, progress_mailbox(pack_event))
    return YinYangCommunication(state, Any[])
end

# Pack with `pack!(buffer, tables)` the halo pieces of `field` that this rank donates in `sends` and post their
# messages and those of the pieces listed in `receives`, which `finish_pieces!` unpacks with `unpack!(buffer, tables)`
function start_pieces!(pack!, unpack!, field, receives, sends, slot; async)
    data = getregion(field.data, 1)
    arch = architecture(field.grid)
    communication = field.communication_buffers
    state = communication.state
    tag = Int(halo_tag_slots * state.tag + slot)
    buffer(tables) = zeros(architecture(data), eltype(data), length(first_table(tables)), size(data, 3))

    sent = [(receiver, buffer(tables)) for (receiver, tables) in sends]
    for ((_, b), (_, tables)) in zip(sent, sends)
        pack!(b, tables)
    end
    record_event!(state.pack_event, arch)

    poster = async ? state : posted_by_main_thread(state)
    received = [(donor, buffer(tables)) for (donor, tables) in receives]
    for (requests, (donor, b)) in zip(state.requests, received)
        irecv!(poster, b, donor - 1, tag, arch.communicator, requests)
    end
    for (requests, (receiver, b)) in zip(state.requests, sent)
        isend!(poster, b, receiver - 1, tag, arch.communicator, requests)
    end

    # The unpacking holds on to the send buffers until `finish_pieces!` has completed their messages
    push!(communication.unpacks, () -> GC.@preserve sent foreach(((_, b), (_, tables)) -> unpack!(b, tables), received, receives))
    return nothing
end

function finish_pieces!(communication::YinYangCommunication)
    wait_for_messages!(communication.state)
    foreach(unpack! -> unpack!(), communication.unpacks)
    empty!(communication.unpacks)
    return nothing
end

finish_pieces!(::Nothing) = nothing

# The regions of a field share its communication, so a region synchronizes it too
const RegionOfDistributedField = Field{<:Any, <:Any, <:Any, <:Any, <:SerialGrid, <:Any, <:Any, <:Any, <:Any, <:Any, <:YinYangCommunication}

DistributedComputations.synchronize_communication!(field::Union{RegionOfDistributedField, Field{<:Any, <:Any, <:Any, <:Any, <:DistributedMultiRegionGrids}}) =
    finish_pieces!(field.communication_buffers)

function launch_pieces!(grid, data, tables, kernel!, args...)
    ks = axes(data, 3)
    kernel_parameters = KernelParameters((length(first_table(tables)), length(ks)), (0, first(ks) - 1))
    launch!(architecture(grid), grid, kernel_parameters, kernel!, args..., first(ks))
    return nothing
end

@kernel function _pack_copy!(buffer, c, copy, k₀)
    n, k = @index(Global, NTuple)
    @inbounds buffer[n, k - k₀ + 1] = c[copy.donors.i[n], copy.donors.j[n], k]
end

@kernel function _pack_scalar!(buffer, donor, donor_grid, loc, table, k₀)
    n, k = @index(Global, NTuple)
    @inbounds buffer[n, k - k₀ + 1] = interpolate_donor(donor, donor_grid, loc, table, n, k)
end

@kernel function _pack_vector_component!(buffer, uᵈ, vᵈ, (u_table, v_table), component, k₀)
    n, k = @index(Global, NTuple)
    @inbounds begin
        ũ = interpolate_donor(uᵈ, u_table, n, k)
        ṽ = interpolate_donor(vᵈ, v_table, n, k)
        buffer[n, k - k₀ + 1] = rotated_component(component, ũ, ṽ, u_table.rotation.cosψ[n], u_table.rotation.sinψ[n])
    end
end

# Receivers on immersed boundaries of `grid` are zeroed; `grid = nothing` leaves every receiver unmasked
@kernel function _unpack!(c, buffer, table, grid, loc, k₀)
    n, k = @index(Global, NTuple)
    @inbounds begin
        i, j = table.receivers.i[n], table.receivers.j[n]
        value = buffer[n, k - k₀ + 1]
        c[i, j, k] = ifelse(immersed_vector_receiver(i, j, k, grid, loc), zero(value), value)
    end
end

# The interpolations may read the copied halos, so the copies complete before they are packed
function exchange_copies!(field, receives, sends)
    grid, data = getregion(field.grid, 1), getregion(field.data, 1)
    pack!(b, copy) = launch_pieces!(grid, data, copy, _pack_copy!, b, data, copy)
    unpack!(b, copy) = launch_pieces!(grid, data, copy, _unpack!, data, b, copy, nothing, nothing)
    start_pieces!(pack!, unpack!, field, receives, sends, 0; async = false)
    return finish_pieces!(field.communication_buffers)
end

#####
##### Halo filling: an asynchronous fill leaves the interpolated halos to `synchronize_communication!`
#####

function BoundaryConditions.fill_halo_regions!(field::DistributedYinYangField, args::Vararg{Any, N}; async = false, kwargs...) where N
    without_boundary_conditions(field.boundary_conditions) && return nothing
    finish_pieces!(field.communication_buffers)
    fill_yin_yang_vertical_halos!(field, args...)
    seam_halos(field) || return nothing

    exchange = yin_yang_connectivity(field.grid).exchange
    loc = instantiated_location(field)
    receives = region_halo(getregion(yin_yang_region_halos(field.grid), 1), loc[1], loc[2])
    sends = region_halo(exchange.scalars, loc[1], loc[2])
    exchange_copies!(field, receives.copies, sends.copies)

    grid, data = getregion(field.grid, 1), getregion(field.data, 1)
    async &= architecture(field.grid) isa AsynchronousDistributed
    pack!(b, (table,)) = launch_pieces!(grid, data, table, _pack_scalar!, b, data, grid, loc, table)
    unpack!(b, (table,)) = launch_pieces!(grid, data, table, _unpack!, data, b, table, nothing, nothing)
    start_pieces!(pack!, unpack!, field, receives.interpolations, sends.interpolations, 1; async)
    async || finish_pieces!(field.communication_buffers)
    return nothing
end

const DistributedYinYangXComponent = DistributedYinYangField{<:Face, <:Center}
const DistributedYinYangYComponent = DistributedYinYangField{<:Center, <:Face}

function BoundaryConditions.fill_halo_regions!((u, v)::Tuple{DistributedYinYangXComponent, DistributedYinYangYComponent},
                                               args::Vararg{Any, N}; async = false, kwargs...) where N
    without_boundary_conditions(u.boundary_conditions) && return nothing
    finish_pieces!(u.communication_buffers)
    finish_pieces!(v.communication_buffers)
    fill_yin_yang_vertical_halos!(u, args...)
    fill_yin_yang_vertical_halos!(v, args...)
    seam_halos(u) || return nothing

    exchange = yin_yang_connectivity(u.grid).exchange
    receives = getregion(yin_yang_region_halos(u.grid), 1).vector
    sends = exchange.vector
    exchange_copies!(u, receives.u_copies, sends.u_copies)
    exchange_copies!(v, receives.v_copies, sends.v_copies)

    # Each component is interpolated from both donor components, and zeroed on the immersed receivers
    grid = getregion(u.grid, 1)
    async &= architecture(u.grid) isa AsynchronousDistributed
    uᵈ, vᵈ = getregion(u.data, 1), getregion(v.data, 1)
    for (c, receive, send, component) in ((u, receives.u, sends.u, Val(1)), (v, receives.v, sends.v, Val(2)))
        data = getregion(c.data, 1)
        loc = instantiated_location(c)
        pack!(b, tables) = launch_pieces!(grid, data, tables, _pack_vector_component!, b, uᵈ, vᵈ, tables, component)
        unpack!(b, tables) = launch_pieces!(grid, data, tables, _unpack!, data, b, first(tables), grid, loc)
        start_pieces!(pack!, unpack!, c, receive, send, 1; async)
    end
    async || foreach(c -> finish_pieces!(c.communication_buffers), (u, v))
    return nothing
end

#####
##### Reductions combine the result of every rank
#####

function all_reduce_regions!(combine, r, grid::DistributedYinYangGridOfSomeKind)
    values = on_architecture(CPU(), interior(r))
    MPI.Allreduce!(values, combine, architecture(grid).communicator)
    interior(r) .= on_architecture(architecture(r), values)
    return nothing
end
