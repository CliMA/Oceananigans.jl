
# On the last rank the fold exchanges use the extra slots 8, 9 and 10
function DistributedComputations.north_recv_tag(arch, ::TripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 8 : side_id[:south]
    return Int(halo_tag_slots * field_id + slot)
end

function DistributedComputations.north_send_tag(arch, ::TripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 8 : side_id[:north]
    return Int(halo_tag_slots * field_id + slot)
end

function DistributedComputations.northwest_recv_tag(arch, ::TripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 9 : side_id[:southeast]
    return Int(halo_tag_slots * field_id + slot)
end

function DistributedComputations.northwest_send_tag(arch, ::TripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 9 : side_id[:northwest]
    return Int(halo_tag_slots * field_id + slot)
end

function DistributedComputations.northeast_recv_tag(arch, ::TripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 10 : side_id[:southwest]
    return Int(halo_tag_slots * field_id + slot)
end

function DistributedComputations.northeast_send_tag(arch, ::TripolarGridOfSomeKind, field_id)
    last_rank = arch.local_index[2] == ranks(arch)[2]
    slot = last_rank ? 10 : side_id[:northeast]
    return Int(halo_tag_slots * field_id + slot)
end
