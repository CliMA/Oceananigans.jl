using MPI: MPI
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Utils: sync_device!

#####
##### A YinYangGrid on a `Distributed` architecture with `partition = MultiRegionPartition(yin, yang)`: each rank
##### holds one piece of one panel, piece r on rank r - 1. Every rank builds the halo tables of every piece, so a
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
struct YinYangExchange{C, S, V}
    communicator :: C
    scalars :: S # (; center_center, face_center, center_face, face_face) of (; copies, interpolations)
    vector :: V  # (; u_copies, v_copies, u, v)
end

function yin_yang_exchange(arch::Distributed, halos)
    r = arch.local_rank + 1
    donated(pieces) = [(q, table) for q in eachindex(halos) for (d, table) in pieces(halos[q]) if d == r]
    scalar(name) = (; copies = donated(h -> getfield(h.scalars, name).copies),
                      interpolations = donated(h -> getfield(h.scalars, name).interpolations))
    scalars = NamedTuple{(:center_center, :face_center, :center_face, :face_face)}(
        map(scalar, (:center_center, :face_center, :center_face, :face_face)))
    vector = NamedTuple{(:u_copies, :v_copies, :u, :v)}(
        map(name -> donated(h -> getfield(h.vector, name)), (:u_copies, :v_copies, :u, :v)))
    return YinYangExchange(arch.communicator, scalars, vector)
end

Grids.on_architecture(arch, e::YinYangExchange) =
    YinYangExchange(e.communicator, map(s -> map(p -> on_architecture_pieces(arch, p), s), e.scalars),
                    map(p -> on_architecture_pieces(arch, p), e.vector))

@inline region_halo(s::NamedTuple, ::Center, ::Center) = s.center_center
@inline region_halo(s::NamedTuple, ::Face,   ::Center) = s.face_center
@inline region_halo(s::NamedTuple, ::Center, ::Face)   = s.center_face
@inline region_halo(s::NamedTuple, ::Face,   ::Face)   = s.face_face

const DistributedYinYangGrid = MultiRegionGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:SubregionPartition{<:YinYangPartition},
                                               <:YinYangConnectivity{<:Any, <:Any, <:Any, <:Any, <:YinYangExchange}}

const DistributedYinYangGridOfSomeKind = Union{DistributedYinYangGrid,
    ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:DistributedYinYangGrid, <:Any, <:Any, <:Any, <:AbstractSerialArchitecture}}

const DistributedYinYangField = Field{<:Any, <:Any, <:Any, <:Any, <:DistributedYinYangGridOfSomeKind}

#####
##### Exchanging halo pieces
#####

first_table(tables::Tuple) = first(tables)
first_table(table) = table

# Fill the halo pieces of `data` listed in `receives` from their donor ranks, after packing with `pack!(buffer, table)`
# the pieces this rank donates in `sends`; `unpack!(buffer, table)` writes a received buffer into `data`
function exchange_pieces!(pack!, unpack!, exchange, data, receives, sends, tag)
    arch = architecture(data)
    Nk = size(data, 3)
    buffer(tables) = zeros(arch, eltype(data), length(first_table(tables)), Nk)

    sent = [(receiver, buffer(tables)) for (receiver, tables) in sends]
    for ((_, b), (_, tables)) in zip(sent, sends)
        pack!(b, tables)
    end
    sync_device!(arch)

    received = [(donor, buffer(tables)) for (donor, tables) in receives]
    requests = [[MPI.Irecv!(b, exchange.communicator; source = donor - 1, tag) for (donor, b) in received];
                [MPI.Isend(b, exchange.communicator; dest = receiver - 1, tag) for (receiver, b) in sent]]
    MPI.Waitall(requests)

    for ((_, b), (_, tables)) in zip(received, receives)
        unpack!(b, tables)
    end
    return nothing
end

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

function exchange_copies!(field, exchange, receives, sends, tag)
    grid, data = getregion(field.grid, 1), getregion(field.data, 1)
    pack!(b, copy) = launch_pieces!(grid, data, copy, _pack_copy!, b, data, copy)
    unpack!(b, copy) = launch_pieces!(grid, data, copy, _unpack!, data, b, copy, nothing, nothing)
    return exchange_pieces!(pack!, unpack!, exchange, data, receives, sends, tag)
end

#####
##### Halo filling
#####

function BoundaryConditions.fill_halo_regions!(field::DistributedYinYangField, args::Vararg{Any, N}; kwargs...) where N
    without_boundary_conditions(field.boundary_conditions) && return nothing
    fill_yin_yang_vertical_halos!(field, args...)
    seam_halos(field) || return nothing

    exchange = yin_yang_connectivity(field.grid).exchange
    loc = instantiated_location(field)
    receives = region_halo(getregion(yin_yang_region_halos(field.grid), 1), loc[1], loc[2])
    sends = region_halo(exchange.scalars, loc[1], loc[2])
    exchange_copies!(field, exchange, receives.copies, sends.copies, 1)

    grid, data = getregion(field.grid, 1), getregion(field.data, 1)
    pack!(b, (table,)) = launch_pieces!(grid, data, table, _pack_scalar!, b, data, grid, loc, table)
    unpack!(b, (table,)) = launch_pieces!(grid, data, table, _unpack!, data, b, table, nothing, nothing)
    exchange_pieces!(pack!, unpack!, exchange, data, receives.interpolations, sends.interpolations, 2)
    return nothing
end

const DistributedYinYangXComponent = DistributedYinYangField{<:Face, <:Center}
const DistributedYinYangYComponent = DistributedYinYangField{<:Center, <:Face}

function BoundaryConditions.fill_halo_regions!((u, v)::Tuple{DistributedYinYangXComponent, DistributedYinYangYComponent},
                                               args::Vararg{Any, N}; kwargs...) where N
    without_boundary_conditions(u.boundary_conditions) && return nothing
    fill_yin_yang_vertical_halos!(u, args...)
    fill_yin_yang_vertical_halos!(v, args...)
    seam_halos(u) || return nothing

    exchange = yin_yang_connectivity(u.grid).exchange
    receives = getregion(yin_yang_region_halos(u.grid), 1).vector
    sends = exchange.vector
    exchange_copies!(u, exchange, receives.u_copies, sends.u_copies, 3)
    exchange_copies!(v, exchange, receives.v_copies, sends.v_copies, 4)

    # Each component is interpolated from both donor components, and zeroed on the immersed receivers
    grid = getregion(u.grid, 1)
    uᵈ, vᵈ = getregion(u.data, 1), getregion(v.data, 1)
    for (c, receive, send, component, tag) in ((u, receives.u, sends.u, Val(1), 5), (v, receives.v, sends.v, Val(2), 6))
        data = getregion(c.data, 1)
        loc = instantiated_location(c)
        pack!(b, tables) = launch_pieces!(grid, data, tables, _pack_vector_component!, b, uᵈ, vᵈ, tables, component)
        unpack!(b, tables) = launch_pieces!(grid, data, tables, _unpack!, data, b, first(tables), grid, loc)
        exchange_pieces!(pack!, unpack!, exchange, data, receive, send, tag)
    end
    return nothing
end

#####
##### Reductions combine the result of every rank
#####

function all_reduce_regions!(combine, r, grid::DistributedYinYangGridOfSomeKind)
    values = on_architecture(CPU(), interior(r))
    MPI.Allreduce!(values, combine, yin_yang_connectivity(grid).exchange.communicator)
    interior(r) .= on_architecture(architecture(r), values)
    return nothing
end
