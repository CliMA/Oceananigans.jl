# One long-lived task per process (and type of event) posts and progresses all asynchronous halo messages,
# so that MPI is polled by a single thread and no task is spawned per message.

# The buffer is a raw pointer, so that the worker only handles concrete types and never touches (GPU) arrays
struct HaloMessage{E}
    send :: Bool
    buffer :: MPI.Buffer{MPI.MPIPtr}
    keep_alive :: Any # the array behind `buffer`, until the message is complete
    rank :: Int
    tag :: Int
    communicator :: MPI.Comm
    request :: MPI.MultiRequestItem{MPI.UnsafeMultiRequest}
    pack_event :: E
    pending_messages :: Threads.Atomic{UInt64}
end

# Each side of a communication state has one [send, recv] pair of requests
function HaloMessage(state, send, array, rank, tag, communicator, requests)
    buffer = MPI.Buffer(array)
    keep_alive = Base.cconvert(MPI.MPIPtr, buffer.data)
    pointer = Base.unsafe_convert(MPI.MPIPtr, keep_alive)
    buffer = MPI.Buffer(pointer, buffer.count, buffer.datatype)
    request = requests[send ? 1 : 2]
    return HaloMessage(send, buffer, keep_alive, rank, tag, mpi_communicator(communicator), request, state.pack_event, state.pending_messages)
end

# Communicators that wrap an MPI communicator (e.g. `NCCLCommunicator`) extend this to unwrap it
mpi_communicator(communicator::MPI.Comm) = communicator

isend!(state, array, rank, tag, communicator, requests) = submit!(state, HaloMessage(state, true, array, rank, tag, communicator, requests))
irecv!(state, array, rank, tag, communicator, requests) = submit!(state, HaloMessage(state, false, array, rank, tag, communicator, requests))

function submit!(state::CommunicationState{<:Any, <:Any, Nothing}, message)
    wait_for_event(state.pack_event)
    post!(message)
    return nothing
end

function submit!(state, message)
    Threads.atomic_add!(state.pending_messages, UInt64(1))
    put!(state.mailbox, message)
    return nothing
end

const progress_mailboxes = Dict{DataType, Channel}()
const progress_worker_failure = Ref{Any}(nothing)
const progress_worker_enabled = Ref{Union{Nothing, Bool}}(nothing)

# The worker calls MPI concurrently with the main thread (`MPI_THREAD_MULTIPLE`)
has_spare_thread() = any(>(1), Threads.nthreads.((:default, :interactive)))

function use_progress_worker()
    if isnothing(progress_worker_enabled[])
        progress_worker_enabled[] = has_spare_thread() && MPI.Query_thread() == MPI.THREAD_MULTIPLE
    end
    return progress_worker_enabled[]::Bool
end

# One worker per type of event (e.g. CPU and GPU fields in the same run), so that each handles a single concrete type of message
function progress_mailbox(pack_event::E) where E
    use_progress_worker() || return nothing
    mailbox = get!(() -> start_progress_worker(HaloMessage{E}), progress_mailboxes, E)
    return mailbox::Channel{HaloMessage{E}}
end

function start_progress_worker(M)
    mailbox = Channel{M}(Inf)
    # A single interactive thread is the main thread itself, so the worker goes to the default pool
    pool = Threads.nthreads(:interactive) > 1 ? :interactive : :default
    errormonitor(Threads.@spawn pool run_progress_worker!(mailbox))
    MPI.add_finalize_hook!(() -> close(mailbox))
    return mailbox
end

function post!(message::HaloMessage)
    if message.send
        MPI.Isend(message.buffer, message.rank, message.tag, message.communicator, message.request)
    else
        MPI.Irecv!(message.buffer, message.rank, message.tag, message.communicator, message.request)
    end
    return nothing
end

# Compacts `in_flight` in place: `filter!` would shrink it, so that the next `push!` reallocates
function remove_completed!(in_flight)
    n = 0
    for message in in_flight
        if MPI.Test(message.request)
            Threads.atomic_sub!(message.pending_messages, UInt64(1))
        else
            in_flight[n += 1] = message
        end
    end
    resize!(in_flight, n)
    return nothing
end

function run_progress_worker!(mailbox::Channel{M}) where M
    awaiting_pack = M[] # in stream order
    in_flight = M[]
    try
        while isopen(mailbox) # closed by `MPI.Finalize`
            while isready(mailbox) || (isempty(awaiting_pack) && isempty(in_flight))
                push!(awaiting_pack, take!(mailbox))
            end
            bind_thread_to_device!(isempty(awaiting_pack) ? first(in_flight).pack_event : first(awaiting_pack).pack_event)
            while !isempty(awaiting_pack) && event_done(first(awaiting_pack).pack_event)
                post!(first(awaiting_pack))
                push!(in_flight, popfirst!(awaiting_pack))
            end
            remove_completed!(in_flight)
            yield()
        end
    catch error
        error isa InvalidStateException && !isopen(mailbox) && return nothing # closed by `MPI.Finalize`
        progress_worker_failure[] = CapturedException(error, catch_backtrace())
        rethrow()
    end
end

rethrow_progress_worker_failure() = isnothing(progress_worker_failure[]) || throw(progress_worker_failure[])
