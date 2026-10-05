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

const pending_exchanges = Ref{Union{Nothing, Channel{Any}}}(nothing)
const progress_failure = Ref{Any}(nothing)
const progress_worker_enabled = Ref{Union{Nothing, Bool}}(nothing)

# The worker needs more than one default thread, and calls MPI concurrently with the main thread (`MPI_THREAD_MULTIPLE`)
function use_progress_worker()
    if isnothing(progress_worker_enabled[])
        spare_thread = any(>(1), Threads.nthreads.((:default, :interactive)))
        progress_worker_enabled[] = spare_thread && MPI.Query_thread() == MPI.THREAD_MULTIPLE
    end
    return progress_worker_enabled[]::Bool
end

# A single interactive thread is the main thread itself, so the worker goes to the default pool
progress_worker_threadpool() = Threads.nthreads(:interactive) > 1 ? :interactive : :default

function submit_exchange!(exchange::HaloExchange)
    isnothing(pending_exchanges[]) && (pending_exchanges[] = start_progress_worker())
    put!(pending_exchanges[], exchange)
    return nothing
end

function start_progress_worker()
    exchanges = Channel{Any}(Inf)
    pool = progress_worker_threadpool()
    errormonitor(Threads.@spawn pool progress_exchanges!(exchanges))
    MPI.add_finalize_hook!(() -> put!(exchanges, nothing))
    return exchanges
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

function progress_exchanges!(exchanges)
    in_flight = PostedExchange[]
    try
        while true
            # block on new exchanges only when none is in flight
            while isready(exchanges) || isempty(in_flight)
                exchange = take!(exchanges)
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
