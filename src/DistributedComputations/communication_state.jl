using Adapt: Adapt

# State of the halo communication of a distributed field.
# - `fill_events` counts the halo exchanges still in flight on communicating tasks (multi-threaded runs): it is
#   incremented by the main thread when a task is spawned and decremented by the task once its MPI requests have
#   completed, so that the main thread knows when the receive buffers are ready to be unpacked.
# - `pending_requests` holds the requests posted by the main thread itself (single-threaded runs), which are
#   completed in `wait_for_comms!`. Only the main thread touches it.
struct CommState
    fill_events :: Threads.Atomic{UInt64}
    pending_requests :: Vector{MPI.Request}
    tag :: UInt64
end

# CommState only lives on host, so never needs to be converted
Adapt.adapt_structure(to, cs::CommState) = nothing
on_architecture(arch, cs::CommState) = cs

communication_state(arch) = nothing
communication_state(arch::Distributed) = CommState(Threads.Atomic{UInt64}(0), MPI.Request[], mod(get_new_tag(arch), 10^ID_DIGITS))

add_fill_event!(f) = nothing
add_fill_event!(f::Field) = add_fill_event!(f.communication_buffers.state)
add_fill_event!(cs::CommState) = Threads.atomic_add!(cs.fill_events, UInt64(1))

complete_fill_event!(f) = nothing
complete_fill_event!(f::Field) = complete_fill_event!(f.communication_buffers.state)
complete_fill_event!(cs::CommState) = Threads.atomic_sub!(cs.fill_events, UInt64(1))

add_pending_requests!(cs::CommState, ::Nothing) = nothing
add_pending_requests!(cs::CommState, requests) = append!(cs.pending_requests, requests)

wait_for_comms!(_) = nothing
wait_for_comms!(f::Field) = wait_for_comms!(f.communication_buffers.state)

function wait_for_comms!(cs::CommState)
    while cs.fill_events[] != 0
        yield()
    end

    if !isempty(cs.pending_requests)
        MPI.Waitall(cs.pending_requests)
        empty!(cs.pending_requests)
    end

    return nothing
end

get_comm_tag(cs::CommState) = cs.tag
