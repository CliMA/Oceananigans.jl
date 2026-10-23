"""
    sanitize_environ!()

Remove malformed entries (those without an `=`) from the process environment and
return the list of removed entries.

This works around a Cray MPICH bug seen on HPE Cray EX systems (e.g. NERSC's
Perlmutter): on multi-node jobs launched with `srun`, a malformed environment entry
is inserted after `MPI_Init`. CUDA.jl is sensitive to this entry, and on multi-node
GPU runs it causes hangs/errors at the first device operation (single-node runs are
unaffected, which makes this easy to miss).

Call once immediately after `MPI.Init()` (i.e. after constructing a `Distributed`
architecture, which initializes MPI):

```julia
using MPI, CUDA, Oceananigans
arch = Distributed(GPU())
Oceananigans.DistributedComputations.sanitize_environ!()
```

See CliMA/Oceananigans.jl discussion #5513 ("Using NERSC's Perlmutter HPC").
"""
function sanitize_environ!()
    envp = unsafe_load(cglobal(:environ, Ptr{Ptr{UInt8}}))

    valid = String[]
    removed = String[]
    i = 0
    while true
        ptr = unsafe_load(envp, i + 1)
        ptr == C_NULL && break
        entry = unsafe_string(Base.Cstring(ptr))
        push!(occursin('=', entry) ? valid : removed, entry)
        i += 1
    end

    ccall(:clearenv, Cint, ())
    for entry in valid
        key, value = split(entry, '='; limit = 2)
        rc = ccall(:setenv, Cint, (Cstring, Cstring, Cint), key, value, 1)
        rc == 0 || error("setenv failed for $key (rc=$rc)")
    end

    return removed
end
