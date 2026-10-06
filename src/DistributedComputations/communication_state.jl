using Adapt: Adapt

# `fill_events` counts the halo messages handed to the progress worker and not yet complete
struct CommState{R, E, W}
    fill_events :: Threads.Atomic{UInt64}
    requests :: R # one [send, recv] pair per active side, `nothing` for the inactive ones
    tag :: UInt64
    event :: E # re-recorded after every pack of the send buffers
    progress_worker :: W # channel to the progress worker, `nothing` without it
end

# CommState only lives on host, so never needs to be converted
Adapt.adapt_structure(to, cs::CommState) = nothing
on_architecture(arch, cs::CommState) = cs

communication_state(arch, sides...) = nothing

# The tag is the number of distributed fields created before this one, so all ranks must create their distributed fields
# in the same order: a field created on some ranks only shifts the tags of all later fields, and their halo messages never match.
function communication_state(arch::Distributed, west, east, south, north, southwest, southeast, northwest, northeast)
    sides = (; west, east, south, north, southwest, southeast, northwest, northeast)
    tag = UInt64(mod(get_new_tag(arch), MPI.tag_ub() ÷ halo_tag_slots))
    event = new_event(arch)
    return CommState(Threads.Atomic{UInt64}(0), map(side_requests, sides), tag, event, progress_worker(event))
end

side_requests(::Nothing) = nothing
side_requests(buffer) = MPI.UnsafeMultiRequest(2)

add_fill_event!(f) = nothing
add_fill_event!(f::Field) = add_fill_event!(f.communication_buffers)
add_fill_event!(cs::CommState) = Threads.atomic_add!(cs.fill_events, UInt64(1))

wait_for_comms!(_) = nothing
wait_for_comms!(f::Field) = wait_for_comms!(f.communication_buffers)

# Wait for the messages handed to the progress worker, then complete those posted by the main thread
function wait_for_comms!(cs::CommState)
    while cs.fill_events[] != 0
        check_progress_worker()
        yield()
    end
    waitall_comms!(values(cs.requests))
    return nothing
end

# The same state, but its messages are posted right away by the main thread
without_progress_worker(cs::CommState) = CommState(cs.fill_events, cs.requests, cs.tag, cs.event, nothing)

get_comm_tag(cs::CommState) = cs.tag
