using Adapt: Adapt

struct CommunicationState{E, M}
    pending_messages :: Threads.Atomic{UInt64}
    pack_event :: E
    mailbox :: M # to the progress worker, or `nothing`
end

# CommunicationState only lives on host, so never needs to be converted
Adapt.adapt_structure(to, cs::CommunicationState) = nothing
on_architecture(arch, cs::CommunicationState) = cs

function communication_state(arch::Distributed)
    pack_event = new_event(arch)
    return CommunicationState(Threads.Atomic{UInt64}(0), pack_event, progress_mailbox(pack_event))
end

wait_for_messages!(_) = nothing
wait_for_messages!(f::Field) = wait_for_messages!(f.communication_buffers)

posted_by_main_thread(cs::CommunicationState) = CommunicationState(cs.pending_messages, cs.pack_event, nothing)
