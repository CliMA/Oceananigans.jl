import Oceananigans.DistributedComputations: north_recv_tag,
                                             north_send_tag,
                                             northwest_recv_tag,
                                             northwest_send_tag,
                                             northeast_recv_tag,
                                             northeast_send_tag

# On the last rank the fold exchanges use the extra slots 8, 9 and 10
function north_recv_tag(arch, ::MPITripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 8 : side_id[:south]
    return Int(halo_tag_slots * field_id + slot)
end

function north_send_tag(arch, ::MPITripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 8 : side_id[:north]
    return Int(halo_tag_slots * field_id + slot)
end

function northwest_recv_tag(arch, ::MPITripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 9 : side_id[:southeast]
    return Int(halo_tag_slots * field_id + slot)
end

function northwest_send_tag(arch, ::MPITripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 9 : side_id[:northwest]
    return Int(halo_tag_slots * field_id + slot)
end

function northeast_recv_tag(arch, ::MPITripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 10 : side_id[:southwest]
    return Int(halo_tag_slots * field_id + slot)
end

function northeast_send_tag(arch, ::MPITripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 10 : side_id[:northeast]
    return Int(halo_tag_slots * field_id + slot)
end
