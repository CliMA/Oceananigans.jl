using Adapt: Adapt

# Each field keeps one send and one receive request per side in preallocated slots, laid out so that every
# exchange (west + east, south + north, corners) occupies a contiguous range of slots
const REQUEST_SLOTS = (west = 1:2, east = 3:4, south = 5:6, north = 7:8,
                       southwest = 9:10, southeast = 11:12, northwest = 13:14, northeast = 15:16)

const NUMBER_OF_REQUEST_SLOTS = 16

@inline send_slot(side) = first(REQUEST_SLOTS[side])
@inline recv_slot(side) = last(REQUEST_SLOTS[side])

# Index of the exchange (1: west + east, 2: south + north, 3: corners) that owns `slots`
@inline exchange_index(slots) = (first(slots) - 1) ÷ 4 + 1

# State of the halo communication of a distributed field.
# - `fill_events` counts the halo exchanges still in flight on communicating tasks (multi-threaded runs): it is
#   incremented by the main thread when a task is spawned and decremented by the task once its requests have
#   completed, so that the main thread knows when the receive buffers are ready to be unpacked.
# - `requests` holds the MPI requests of all sides (see `REQUEST_SLOTS`). A slot is reused only after its
#   request has completed, so posting and testing requests does not allocate.
# - `test_flags` holds one completion flag per exchange, so that tasks testing different exchanges do not share it.
struct CommState
    fill_events :: Threads.Atomic{UInt64}
    requests :: MPI.UnsafeMultiRequest
    test_flags :: Vector{Cint}
    tag :: UInt64
end

# CommState only lives on host, so never needs to be converted
Adapt.adapt_structure(to, cs::CommState) = nothing
on_architecture(arch, cs::CommState) = cs

communication_state(arch) = nothing
communication_state(arch::Distributed) = CommState(Threads.Atomic{UInt64}(0),
                                                   MPI.UnsafeMultiRequest(NUMBER_OF_REQUEST_SLOTS),
                                                   zeros(Cint, 3),
                                                   mod(get_new_tag(arch), 10^ID_DIGITS))

add_fill_event!(f) = nothing
add_fill_event!(f::Field) = add_fill_event!(f.communication_buffers.state)
add_fill_event!(cs::CommState) = Threads.atomic_add!(cs.fill_events, UInt64(1))

complete_fill_event!(f) = nothing
complete_fill_event!(f::Field) = complete_fill_event!(f.communication_buffers.state)
complete_fill_event!(cs::CommState) = Threads.atomic_sub!(cs.fill_events, UInt64(1))

# Keep testing the requests in `slots` instead of deferring all progress to `Waitall` (for MPI implementations without asynchronous progress).
# MPI_Testall is called directly on the preallocated handles and flag, so that polling does not allocate.
function progress_comms!(cs::CommState, slots::UnitRange{Int})
    GC.@preserve cs begin
        handles  = pointer(cs.requests.vals, first(slots))
        complete = pointer(cs.test_flags, exchange_index(slots))
        while true
            MPI.API.MPI_Testall(length(slots), handles, complete, MPI.API.MPI_STATUSES_IGNORE[])
            unsafe_load(complete) == 0 || break
            yield()
        end
    end
    return nothing
end

progress_comms!(cs, ::Nothing) = nothing

function waitall_comms!(cs::CommState, slots::UnitRange{Int})
    GC.@preserve cs begin
        MPI.API.MPI_Waitall(length(slots), pointer(cs.requests.vals, first(slots)), MPI.API.MPI_STATUSES_IGNORE[])
    end
    return nothing
end

waitall_comms!(cs, ::Nothing) = nothing

wait_for_comms!(_) = nothing
wait_for_comms!(f::Field) = wait_for_comms!(f.communication_buffers.state)

# Wait for the communicating tasks, then complete the requests posted by the main thread (single-threaded runs);
# slots without a pending request are null and ignored by `Waitall`.
function wait_for_comms!(cs::CommState)
    while cs.fill_events[] != 0
        yield()
    end

    waitall_comms!(cs, 1:NUMBER_OF_REQUEST_SLOTS)

    return nothing
end

get_comm_tag(cs::CommState) = cs.tag
