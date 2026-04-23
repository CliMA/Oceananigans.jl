#####
##### Async wrapper around any PartlyInMemory backend.
##### Reads the next sliding window into `buffer_fts.data` on a `Threads.@spawn` task
##### so reload I/O is hidden behind compute. The hot/cold logic lives in
##### `update_field_time_series!(::PrefetchingFTS, ...)` below.
#####
##### Race invariant: between the spawn at the end of one update and the wait at the
##### start of the next, the worker mutates `buffer_fts.data`. No code outside
##### `update_field_time_series!(::PrefetchingFTS, ...)` may touch `buffer_fts` in that
##### window — `getproperty` warns on raw access and `Adapt` strips the wrapper for GPU
##### descent so kernels never see the Task or the buffer.
#####
##### File-handle assumption: `Prefetched` assumes the underlying backend is the only
##### reader of its data source. Reading from the same file via a separate FTS on the
##### main thread while a prefetch task is in flight has unspecified behaviour
##### (we observed JLD2 in particular not surviving it).
#####

mutable struct Prefetched{B<:AbstractInMemoryBackend{Int}, F} <: AbstractInMemoryBackend{Int, false}
    base_backend :: B
    pending      :: Union{Task, Nothing}
    buffer_fts   :: F
    next_start   :: Int
end

function Base.getproperty(p::Prefetched, name::Symbol)
    name in (:base_backend, :pending, :next_start) && return getfield(p, name)
    if name === :buffer_fts
        @warn "`buffer_fts` is touched by a worker task; mutating it is undefined behaviour."
        return getfield(p, name)
    end
    return getproperty(getfield(p, :base_backend), name)
end

Base.length(p::Prefetched)  = length(p.base_backend)
Base.summary(p::Prefetched) = string("Prefetched(", summary(p.base_backend), "; pending=", !isnothing(p.pending), ")")

function new_backend(p::Prefetched, start, length) 
    p.base_backend = new_backend(p.base_backend, start, length)
    return p
end

# GPU descent: kernels never see the Task / buffer FTS — strip to base backend.
Adapt.adapt_structure(to, p::Prefetched) = Adapt.adapt(to, p.base_backend)

# Architecture transfer: cold-restart prefetch on the new architecture.
function on_architecture(to, p::Prefetched)
    isnothing(p.pending) || try; wait(p.pending); catch; end
    new_base_backend = on_architecture(to, p.base_backend)
    new_buffer       = on_architecture(to, getfield(p, :buffer_fts))
    return Prefetched(new_base_backend, nothing, new_buffer, 0)
end

const PrefetchingFTS = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:Prefetched}

#####
##### Construction handoff
#####

function build_runtime_backend(backend::AbstractInMemoryBackend{Int, true}, grid, loc, indices, times, path, name, time_indexing, boundary_conditions, reader_kw)
    if Threads.nthreads() == 1
        @warn "InMemory(N; prefetch=true) requires JULIA_NUM_THREADS ≥ 2 to overlap I/O with compute; " *
              "disabling prefetch. Re-launch Julia with `-t 2` (or higher) to enable."
        return backend
    end
    LX, LY, LZ = typeof.(loc)
    buffer_data = new_data(eltype(grid), grid, loc, indices, length(backend))
    buffer_fts = FieldTimeSeries{LX, LY, LZ}(buffer_data, grid, backend, boundary_conditions, indices,
                                             times, path, name, time_indexing, reader_kw)
    return Prefetched(backend, nothing, buffer_fts, 0)
end

#####
##### Runtime: hot/cold update path
#####

function update_field_time_series!(fts::PrefetchingFTS, n₁::Int, n₂=n₁)
    in_time_range(fts, fts.time_indexing, n₁, n₂) && return nothing

    backend    = fts.backend
    Nm         = length(backend)
    Nt         = length(fts.times)
    needed     = n₁
    pending    = backend.pending
    buffer_fts = getfield(backend, :buffer_fts)  # bypass tamper-guard warning

    backend.pending = nothing # so a failed wait can't poison later reloads

    hot = !isnothing(pending) && backend.next_start == needed
    if hot
        try
            wait(pending)
        catch
            @warn "prefetch failed; falling back to synchronous load."
            hot = false
        end
    elseif !isnothing(pending)
        try; wait(pending); catch; end # drain stale prefetch silently
    end

    if !hot
        buffer_fts.backend = new_backend(buffer_fts.backend, needed, Nm)
        set!(buffer_fts)
    end

    new_backend(backend, needed, Nm) # advance user-visible window via in-place mutation
    copyto!(parent(fts.data), parent(buffer_fts.data))
    fill_halo_regions!(fts)

    new_next = time_index(buffer_fts.backend, fts.time_indexing, Nt, Nm)

    if new_next == needed # Linear/Clamp at end-of-data: no further window
        backend.next_start = 0
        return nothing
    end

    buffer_fts.backend = new_backend(buffer_fts.backend, new_next, Nm)
    backend.next_start = new_next
    backend.pending = Threads.@spawn try
        set!(buffer_fts)
    catch e
        @error "prefetch task failed" exception=(e, catch_backtrace())
        rethrow()
    end

    return nothing
end
