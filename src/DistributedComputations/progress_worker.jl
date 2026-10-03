# One long-lived task per process posts and progresses all asynchronous halo exchanges,
# so that MPI is polled by a single thread and no task is spawned per exchange.

struct HaloExchange{P, E, B, A}
    post_requests! :: P
    fill_event :: E
    buffers :: B
    args :: A
end

struct PostedExchange{R, B}
    requests :: R
    buffers :: B
end

const progress_queue = Ref{Union{Nothing, Channel{Any}}}(nothing)
const progress_failure = Ref{Any}(nothing)
const progress_worker_enabled = Ref{Union{Nothing, Bool}}(nothing)

# The worker needs a thread besides the main one, and calls MPI concurrently with it (`MPI_THREAD_MULTIPLE`)
function use_progress_worker()
    if isnothing(progress_worker_enabled[])
        spare_thread = Threads.nthreads(:default) + Threads.nthreads(:interactive) > 1
        progress_worker_enabled[] = spare_thread && MPI.Query_thread() == MPI.THREAD_MULTIPLE
    end
    return progress_worker_enabled[]::Bool
end

# A single interactive thread is the main thread itself, so the worker goes to the default pool
progress_worker_threadpool() = Threads.nthreads(:interactive) > 1 ? :interactive : :default

function submit_exchange!(exchange::HaloExchange)
    queue = progress_queue[]

    if isnothing(queue)
        queue = Channel{Any}(Inf)
        pool = progress_worker_threadpool()
        errormonitor(Threads.@spawn pool progress_exchanges!(queue))
        MPI.add_finalize_hook!(() -> put!(queue, nothing))
        progress_queue[] = queue
    end

    put!(queue, exchange)
    return nothing
end

function post!(exchange::HaloExchange)
    sync_event(exchange.fill_event)
    requests = exchange.post_requests!(exchange.args..., exchange.buffers)
    return PostedExchange(requests, exchange.buffers)
end

requests_complete(requests::Tuple) = all(requests_complete, requests)
requests_complete(::Nothing) = true
requests_complete(requests::MPI.UnsafeMultiRequest) = MPI.Testall(requests)

function complete!(exchange::PostedExchange)
    requests_complete(exchange.requests) || return false
    complete_fill_event!(exchange.buffers)
    return true
end

function progress_exchanges!(queue)
    in_flight = PostedExchange[]
    try
        while true
            # block on the queue only when no exchange is in flight
            while isready(queue) || isempty(in_flight)
                exchange = take!(queue)
                isnothing(exchange) && return nothing
                push!(in_flight, post!(exchange))
            end
            filter!(!complete!, in_flight)
            yield()
        end
    catch error
        progress_failure[] = CapturedException(error, catch_backtrace())
        rethrow()
    end
end

check_progress_worker() = isnothing(progress_failure[]) || throw(progress_failure[])
