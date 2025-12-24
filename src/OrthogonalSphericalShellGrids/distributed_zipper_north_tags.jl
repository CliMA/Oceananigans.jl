import Oceananigans.DistributedComputations: north_recv_tag,
                                             north_send_tag,
                                             northwest_recv_tag,
                                             northwest_send_tag,
                                             northeast_recv_tag,
                                             northeast_send_tag

ID_DIGITS = 2

sides  = (:west, :east, :south, :north, :southwest, :southeast, :northwest, :northeast)
side_id = Dict(side => n-1 for (n, side) in enumerate(sides))

# Change these and we are golden!
function north_recv_tag(arch, ::DistributedTripolarGridOfSomeKind, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? "8" : string(side_id[:south])
    return parse(Int, field_id * loc_digit * side_digit)
end

function north_send_tag(arch, ::DistributedTripolarGridOfSomeKind, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? "8" : string(side_id[:north])
    return parse(Int, field_id * loc_digit * side_digit)
end

function northwest_recv_tag(arch, ::DistributedTripolarGridOfSomeKind, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? "9" : string(side_id[:southeast])
    return parse(Int, field_id * loc_digit * side_digit)
end

function northwest_send_tag(arch, ::DistributedTripolarGridOfSomeKind, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? "9" : string(side_id[:northwest])
    return parse(Int, field_id * loc_digit * side_digit)
end

function northeast_recv_tag(arch, ::DistributedTripolarGridOfSomeKind, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? "10" : string(side_id[:southwest])
    return parse(Int, field_id * loc_digit * side_digit)
end

function northeast_send_tag(arch, ::DistributedTripolarGridOfSomeKind, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? "10" : string(side_id[:northeast])
    return parse(Int, field_id * loc_digit * side_digit)
end
