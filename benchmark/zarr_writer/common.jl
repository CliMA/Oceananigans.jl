using Oceananigans
using JLD2, NCDatasets, Zarr
using Printf, Statistics

const FIELD_SHAPE = (128, 128, 64)
const NSTEPS      = 50

function make_model(; nfields=5)
    grid = RectilinearGrid(CPU(), size=FIELD_SHAPE, extent=(1, 1, 1),
                           topology=(Periodic, Periodic, Periodic))
    tracer_names = Tuple(Symbol("c$i") for i in 1:nfields)
    model = NonhydrostaticModel(grid; tracers=tracer_names)
    for n in tracer_names
        set!(model, n => (x, y, z) -> rand())
    end
    set!(model, u=(x, y, z) -> 1.0, v=(x, y, z) -> -1.0, w=(x, y, z) -> 0.0)
    return model, tracer_names
end

# Collect (u, v, w) + tracers as one NamedTuple of length nfields_total.
make_outputs(model, tracer_names) =
    merge((; u=model.velocities.u, v=model.velocities.v, w=model.velocities.w),
          NamedTuple{tracer_names}(Tuple(model.tracers[t] for t in tracer_names)))

cleanup(paths...) = for p in paths
    isdir(p) && rm(p; recursive=true, force=true)
    isfile(p) && rm(p; force=true)
end

# Approximate on-disk size of any path (file or directory tree).
function path_size(p)
    isfile(p) && return filesize(p)
    isdir(p)  || return 0
    total = 0
    for (root, _, files) in walkdir(p), f in files
        total += filesize(joinpath(root, f))
    end
    return total
end

bytes_to_mib(b) = b / (1024 * 1024)

# Build a writer, time its run!, and return (run_seconds, per_step_seconds, bytes).
function bench_writer(writer_kind::Symbol, model, outputs, path; compressor=nothing)
    cleanup(path)
    sim = Simulation(model, Δt=1.0, stop_iteration=NSTEPS)
    writer = if writer_kind === :JLD2
        JLD2Writer(model, outputs;
                   filename=path, dir=".",
                   schedule=IterationInterval(1),
                   overwrite_existing=true, with_halos=false,
                   array_type=Array{Float32})
    elseif writer_kind === :NetCDF
        NetCDFWriter(model, outputs;
                     filename=path, dir=".",
                     schedule=IterationInterval(1),
                     overwrite_existing=true, with_halos=false,
                     array_type=Array{Float32})
    elseif writer_kind === :Zarr
        ZarrWriter(model, outputs;
                   filename=path, dir=".",
                   schedule=IterationInterval(1),
                   overwrite_existing=true, with_halos=false,
                   array_type=Array{Float32},
                   compressor=compressor)
    else
        error("Unknown writer kind: $writer_kind")
    end
    sim.output_writers[:bench] = writer
    t0 = time_ns()
    run!(sim)
    t1 = time_ns()
    run_s    = (t1 - t0) / 1e9
    per_step = run_s / NSTEPS
    bytes    = path_size(path)
    return (run_s, per_step, bytes)
end

# Read back the full timeseries of `name`. Returns seconds.
function bench_read(writer_kind::Symbol, path, name::String)
    t0 = time_ns()
    if writer_kind === :Zarr
        # The .zarr suffix is required for the path dispatch.
        fts = FieldTimeSeries(path, name)
    else
        fts = FieldTimeSeries(path, name)
    end
    t1 = time_ns()
    return (t1 - t0) / 1e9
end

# Print a markdown row.
print_md_row(io, kind, run_s, per_step, bytes, read_s) =
    @printf(io, "| %-6s | %7.3f | %9.3f | %7.2f | %7.3f |\n",
            kind, run_s, per_step * 1000, bytes_to_mib(bytes), read_s)

function print_md_table(io, results)
    println(io, "| Writer | Run (s) | Per step (ms) | Size (MiB) | Read (s) |")
    println(io, "|--------|---------|---------------|------------|----------|")
    for (kind, run_s, per_step, bytes, read_s) in results
        print_md_row(io, kind, run_s, per_step, bytes, read_s)
    end
end
