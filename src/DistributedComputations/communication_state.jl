using Adapt: Adapt

struct CommunicationState{R, E, M}
    pending_messages :: Threads.Atomic{UInt64}
    requests :: R # one [send, recv] pair per active side, `nothing` for the inactive ones
    tag :: UInt64
    pack_event :: E
    mailbox :: M # to the progress worker, or `nothing`
    unpack_pending :: Base.RefValue{Bool} # an asynchronous fill was posted but its halos are not yet unpacked
end

# CommunicationState only lives on host, so never needs to be converted
Adapt.adapt_structure(to, cs::CommunicationState) = nothing
on_architecture(arch, cs::CommunicationState) = cs

communication_state(arch, sides...) = nothing

function communication_state(arch::Distributed, west, east, south, north, southwest, southeast, northwest, northeast)
    sides = (; west, east, south, north, southwest, southeast, northwest, northeast)
    tag = UInt64(mod(next_field_tag!(arch), MPI.tag_ub() ÷ halo_tag_slots))
    pack_event = new_event(arch)
    return CommunicationState(Threads.Atomic{UInt64}(0), map(side_requests, sides), tag, pack_event, progress_mailbox(pack_event), Ref(false))
end

side_requests(::Nothing) = nothing
side_requests(buffer) = MPI.UnsafeMultiRequest(2)

wait_for_messages!(_) = nothing
wait_for_messages!(f::Field) = wait_for_messages!(f.communication_buffers)

# The requests completed by the progress worker are already null, so `waitall_requests!` only completes those posted by the main thread
function wait_for_messages!(cs::CommunicationState)
    while cs.pending_messages[] != 0
        rethrow_progress_worker_failure()
        yield()
    end
    waitall_requests!(values(cs.requests))
    return nothing
end

posted_by_main_thread(cs::CommunicationState) = CommunicationState(cs.pending_messages, cs.requests, cs.tag, cs.pack_event, nothing, cs.unpack_pending)

mark_unpack_pending!(cs::CommunicationState) = (cs.unpack_pending[] = true; nothing)
mark_unpack_pending!(_) = nothing

# Unpacking twice would overwrite halos modified since the first unpack with the stale received values
function take_unpack_pending!(cs::CommunicationState)
    pending = cs.unpack_pending[]
    cs.unpack_pending[] = false
    return pending
end

take_unpack_pending!(_) = true
