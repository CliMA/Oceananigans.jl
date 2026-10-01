using Adapt: Adapt

# `fill_events` counts the halo exchanges still in flight on communicating tasks (multi-threaded runs)
struct CommState{R}
    fill_events :: Threads.Atomic{UInt64}
    requests :: R # one [send, recv] pair per active side, `nothing` for the inactive ones
    tag :: UInt64
end

# CommState only lives on host, so never needs to be converted
Adapt.adapt_structure(to, cs::CommState) = nothing
on_architecture(arch, cs::CommState) = cs

communication_state(arch, sides...) = nothing

function communication_state(arch::Distributed, west, east, south, north, southwest, southeast, northwest, northeast)
    sides = (; west, east, south, north, southwest, southeast, northwest, northeast)
    return CommState(Threads.Atomic{UInt64}(0), map(side_requests, sides), UInt64(mod(get_new_tag(arch), 10^ID_DIGITS)))
end

side_requests(::Nothing) = nothing
side_requests(buffer) = MPI.UnsafeMultiRequest(2)

add_fill_event!(f) = nothing
add_fill_event!(f::Field) = add_fill_event!(f.communication_buffers.state)
add_fill_event!(cs::CommState) = Threads.atomic_add!(cs.fill_events, UInt64(1))

complete_fill_event!(f) = nothing
complete_fill_event!(f::Field) = complete_fill_event!(f.communication_buffers.state)
complete_fill_event!(cs::CommState) = Threads.atomic_sub!(cs.fill_events, UInt64(1))

wait_for_comms!(_) = nothing
wait_for_comms!(f::Field) = wait_for_comms!(f.communication_buffers.state)

# Wait for the communicating tasks; with a single thread, complete the requests posted by the main thread
function wait_for_comms!(cs::CommState)
    while cs.fill_events[] != 0
        yield()
    end
    Threads.nthreads() == 1 && waitall_comms!(values(cs.requests))
    return nothing
end

get_comm_tag(cs::CommState) = cs.tag
