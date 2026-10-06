# One long-lived task per process (and type of event) posts and progresses all asynchronous halo messages,
# so that MPI is polled by a single thread and no task is spawned per message.

# One `MPI.Isend` or `MPI.Irecv!`, posted once the send buffers are packed (`event`).
# The buffer is a raw pointer, so that the worker only handles concrete types and never touches (GPU) arrays.
struct HaloMessage{E}
    send :: Bool
    buffer :: MPI.Buffer{MPI.MPIPtr}
    root :: Any # keeps the array behind `buffer` alive until the message is complete; never read
    rank :: Int
    tag :: Int
    communicator :: MPI.Comm
    requests :: MPI.UnsafeMultiRequest
    index :: Int
    event :: E
    fill_events :: Threads.Atomic{UInt64}
end

function HaloMessage(state, send, array, rank, tag, communicator, requests, index)
    buffer = MPI.Buffer(array)
    root = Base.cconvert(MPI.MPIPtr, buffer.data)
    pointer = Base.unsafe_convert(MPI.MPIPtr, root)
    buffer = MPI.Buffer(pointer, buffer.count, buffer.datatype)
    return HaloMessage(send, buffer, root, rank, tag, mpi_communicator(communicator), requests, index, state.event, state.fill_events)
end

# Communicators that wrap an MPI communicator (e.g. `NCCLCommunicator`) extend this to unwrap it
mpi_communicator(communicator::MPI.Comm) = communicator

# Each side of a communication state has one [send, recv] pair of requests
isend!(state, array, rank, tag, communicator, requests) = submit!(state, HaloMessage(state, true, array, rank, tag, communicator, requests, 1))
irecv!(state, array, rank, tag, communicator, requests) = submit!(state, HaloMessage(state, false, array, rank, tag, communicator, requests, 2))

# Without the progress worker, the main thread posts right away
function submit!(state::CommState{<:Any, <:Any, Nothing}, message)
    sync_event(state.event)
    post!(message)
    return nothing
end

function submit!(state, message)
    add_fill_event!(state)
    put!(state.progress_worker, message)
    return nothing
end

const progress_workers = Dict{DataType, Channel}()
const progress_failure = Ref{Any}(nothing)
const progress_worker_enabled = Ref{Union{Nothing, Bool}}(nothing)

# The worker needs a spare thread in either pool, and calls MPI concurrently with the main thread (`MPI_THREAD_MULTIPLE`)
function use_progress_worker()
    if isnothing(progress_worker_enabled[])
        spare_thread = any(>(1), Threads.nthreads.((:default, :interactive)))
        progress_worker_enabled[] = spare_thread && MPI.Query_thread() == MPI.THREAD_MULTIPLE
    end
    return progress_worker_enabled[]::Bool
end

# A single interactive thread is the main thread itself, so the worker goes to the default pool
progress_worker_threadpool() = Threads.nthreads(:interactive) > 1 ? :interactive : :default

# The channel to the worker, or `nothing` without it. There is one worker per type of event
# (e.g. CPU and GPU fields in the same run), so that each worker handles a single concrete type of message.
function progress_worker(event::E) where E
    use_progress_worker() || return nothing
    worker = get!(() -> start_progress_worker(HaloMessage{E}), progress_workers, E)
    return worker::Channel{HaloMessage{E}}
end

function start_progress_worker(M)
    messages = Channel{M}(Inf)
    pool = progress_worker_threadpool()
    errormonitor(Threads.@spawn pool progress_messages!(messages))
    MPI.add_finalize_hook!(() -> close(messages))
    return messages
end

function post!(message::HaloMessage)
    request = message.requests[message.index]
    if message.send
        MPI.Isend(message.buffer, message.rank, message.tag, message.communicator, request)
    else
        MPI.Irecv!(message.buffer, message.rank, message.tag, message.communicator, request)
    end
    return nothing
end

function complete!(message::HaloMessage)
    MPI.Test(message.requests[message.index]) || return false
    Threads.atomic_sub!(message.fill_events, UInt64(1))
    return true
end

# Unlike `filter!(!complete!, posted)`, which shrinks `posted` so that the next `push!` reallocates
function remove_complete!(posted)
    n = 0
    for message in posted
        complete!(message) || (posted[n += 1] = message)
    end
    resize!(posted, n)
    return nothing
end

function progress_messages!(messages::Channel{M}) where M
    packing = M[] # waiting for their send buffers to be packed, in stream order
    posted = M[]  # waiting for MPI to complete them
    try
        while true
            # block on new messages only when none is in flight
            while isready(messages) || (isempty(packing) && isempty(posted))
                push!(packing, take!(messages))
            end
            # the worker resumes on any thread, which MPI needs bound to the device
            bind_thread!(isempty(packing) ? first(posted).event : first(packing).event)
            while !isempty(packing) && event_done(first(packing).event)
                post!(first(packing))
                push!(posted, popfirst!(packing))
            end
            remove_complete!(posted)
            yield()
        end
    catch error
        error isa InvalidStateException && !isopen(messages) && return nothing # closed by `MPI.Finalize`
        progress_failure[] = CapturedException(error, catch_backtrace())
        rethrow()
    end
end

check_progress_worker() = isnothing(progress_failure[]) || throw(progress_failure[])
