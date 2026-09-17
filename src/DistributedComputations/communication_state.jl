using Adapt: Adapt
# State of communications for MPI
# The requests Channel exists for threads to add MPI requests to, in a thread safe way,
# for the main thread to then wait on.
# The fill_events value keeps track of how many events to fill the send buffer
# are being waited on, to gate skipping past communication sync when the channel is empty
# because the requests have not been added yet. The counter should be incremented by the main
# thread, and decremented by the thread that is spawned for that event _after_ the MPI comms
# are performed
struct CommState
  comm_requests::Channel
  fill_events::Threads.Atomic{UInt64}
  tag::UInt64
end

# CommState only lives on host, so never needs to be converted
Adapt.adapt_structure(to, cs::CommState) = nothing
on_architecture(arch, cs::CommState) = nothing

communication_state(arch) = nothing
communication_state(arch::Distributed) = CommState(Channel(Inf), Threads.Atomic{UInt64}(0), get_new_tag(arch))

add_fill_event!(f) = nothing
add_fill_event!(f::Field) = add_fill_event!(f.communication_buffers.state)
function add_fill_event!(cs::CommState)
  Threads.atomic_add!(cs.fill_events, UInt64(1))
end

complete_fill_event!(f) = nothing
complete_fill_event!(f::Field) = complete_fill_event!(f.communication_buffers.state)

function complete_fill_event!(cs::CommState)
  Threads.atomic_sub!(cs.fill_events, UInt64(1))
end

add_comm_requests!(_, _) = nothing
add_comm_requests!(f::Field, reqs) = add_comm_requests!(f.communication_buffers.state, reqs)

add_comm_requests!(cs::CommState, reqs::Nothing) = nothing

function add_comm_requests!(cs::CommState, reqs)
  put!(cs.comm_requests, reqs)
end

wait_for_comms!(_) = nothing
wait_for_comms!(f::Field) = wait_for_comms!(f.communication_buffers.state)

function wait_for_comms!(cs::CommState)
  # Wait for fill_events == 0
  fill_finished = false
  while !fill_finished
    fill_finished = (cs.fill_events[] == 0)
  end
  # Wait for MPI comms to complete
  cooperative_waitall!(cs.comm_requests)
end

get_comm_tag(cs::CommState) = cs.tag
