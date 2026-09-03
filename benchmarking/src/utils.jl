
#####
##### Benchmark utilities
#####

"""
    path_size(p)

On-disk size of a file or directory tree, in bytes. Returns `0` if the path doesn't exist.
Required for directory-store outputs (e.g. Zarr's `DirectoryStore`).
"""
function path_size(p)
    isfile(p) && return filesize(p)
    isdir(p)  || return 0
    total = 0
    for (root, _, files) in walkdir(p)
        for f in files
            total += filesize(joinpath(root, f))
        end
    end
    return total
end

"""
    zarr_chunk_shape(zarr_path::AbstractString, variable_name::AbstractString)

Read the actual spatial chunk shape used for `variable_name` from a written Zarr store
(excludes the time axis, which the ZarrWriter always chunks at 1). Returns `nothing` if
the store or variable cannot be opened.
"""
function zarr_chunk_shape(zarr_path::AbstractString, variable_name::AbstractString)
    try
        group = Zarr.zopen(zarr_path)
        arr = group[variable_name]
        chunks = collect(Int, arr.metadata.chunks)
        return chunks[1:end-1]
    catch
        return nothing
    end
end


"""
    benchmark_time_stepping(model;
                            time_steps = 100,
                            Δt = 60,
                            warmup_steps = 10,
                            name = "benchmark",
                            verbose = true)

Run a benchmark by executing `time_steps` time steps of the given model.
Uses `many_time_steps!` to avoid Simulation overhead.

Returns a `BenchmarkResult` containing timing information and system metadata.
"""
function benchmark_time_stepping(model;
                                 time_steps = 100,
                                 Δt = 60,
                                 warmup_steps = 10,
                                 name = "benchmark",
                                 group = "",
                                 verbose = true)

    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    total_points = Nx * Ny * Nz

    if verbose
        @info "Benchmark: $name"
        @info "  Architecture: $arch"
        @info "  Float type: $FT"
        @info "  Grid size: $Nx × $Ny × $Nz ($total_points points)"
        @info "  Time step: $Δt s"
        @info "  Warmup steps: $warmup_steps"
        @info "  Benchmark steps: $time_steps"
    end

    # Warmup phase
    if verbose
        @info "  Running warmup..."
    end
    many_time_steps!(model, Δt, warmup_steps)

    # Synchronize device before timing
    sync_device!(arch)

    # Benchmark phase
    if verbose
        @info "  Running benchmark..."
    end
    start_time = time_ns()
    many_time_steps!(model, Δt, time_steps)
    sync_device!(arch)
    end_time = time_ns()

    total_time_seconds = (end_time - start_time) / 1e9
    time_per_step_seconds = total_time_seconds / time_steps
    steps_per_second = time_steps / total_time_seconds
    grid_points_per_second = total_points / time_per_step_seconds

    gpu_memory_used = arch isa GPU ? CUDACore.MemoryInfo().pool_used_bytes : 0
    metadata = BenchmarkMetadata(arch)

    result = BenchmarkResult(
        name,
        group,
        string(FT),
        (Nx, Ny, Nz),
        time_steps,
        Δt,
        total_time_seconds,
        time_per_step_seconds,
        steps_per_second,
        grid_points_per_second,
        gpu_memory_used,
        metadata,
    )

    if verbose
        @info "  Results:"
        @info "    Total time: $(@sprintf("%.3f", total_time_seconds)) s"
        @info "    Time per step: $(@sprintf("%.6f", time_per_step_seconds)) s"
        @info "    Grid points/s: $(@sprintf("%.2e", grid_points_per_second))"
        if arch isa GPU
            @info "    GPU memory usage: $(Base.format_bytes(gpu_memory_used))"
        end
    end

    if arch isa GPU
        CUDA.reclaim()
    end

    return result
end

#####
##### Full simulation with output (for validation and longer runs)
#####

"""
    run_benchmark_simulation(model;
                             stop_time = 24hours,
                             Δt = 60,
                             output_interval = 1hour,
                             output_dir = ".",
                             name = "benchmark_simulation",
                             output_fields = (:u, :v, :T, :S),
                             verbose = true)

Run a full simulation with output writers for validation and longer benchmarks.

Returns a `SimulationResult` containing timing information and the output file path.
"""
function run_benchmark_simulation(model;
                                  stop_time = 24hours,
                                  Δt = 60,
                                  output_interval = 1hour,
                                  output_dir = ".",
                                  name = "benchmark_simulation",
                                  group = "",
                                  output_fields = (:u, :v, :T, :S),
                                  verbose = true)

    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    total_points = Nx * Ny * Nz

    # Build output filename
    timestamp = Dates.format(now(UTC), "yyyy-mm-dd_HHMMSS")
    output_filename = joinpath(output_dir, "$(name)_$(timestamp).jld2")
    final_filename = replace(output_filename, ".jld2" => "_final.jld2")

    if verbose
        @info "Benchmark Simulation: $name"
        @info "  Architecture: $arch"
        @info "  Float type: $FT"
        @info "  Grid size: $Nx × $Ny × $Nz ($total_points points)"
        @info "  Time step: $Δt s"
        @info "  Stop time: $(stop_time) s ($(stop_time / 3600) hours)"
        @info "  Output interval: $(output_interval) s ($(output_interval / 60) minutes)"
        @info "  Surface output: $output_filename"
        @info "  Final 3D snapshot: $final_filename"
    end

    simulation = Simulation(model; Δt, stop_time)

    if verbose
        wall_time_ref = Ref(time_ns())
        function progress(sim)
            elapsed = (time_ns() - wall_time_ref[]) / 1e9
            wall_time_ref[] = time_ns()
            u_max = maximum(abs, sim.model.velocities.u)
            @info @sprintf("Time: %.1f/%.1f hours, Δt: %.2f s, max|u|: %.2f m/s, wall: %.1f s",
                           time(sim) / 3600, stop_time / 3600, sim.Δt, u_max, elapsed)
        end
        simulation.callbacks[:progress] = Callback(progress, TimeInterval(output_interval))
    end

    # Build outputs dictionary from field names
    outputs = Dict{Symbol, Any}()
    for field_name in output_fields
        if haskey(model.velocities, field_name)
            outputs[field_name] = model.velocities[field_name]
        elseif hasproperty(model, :tracers) && haskey(model.tracers, field_name)
            outputs[field_name] = model.tracers[field_name]
        end
    end

    # Periodic output: surface slices
    simulation.output_writers[:surface] = JLD2Writer(model, outputs;
        filename = output_filename,
        indices = (:, :, Nz),
        schedule = TimeInterval(output_interval),
        overwrite_existing = true
    )

    # Final snapshot: full 3D fields
    simulation.output_writers[:final_3d] = JLD2Writer(model, outputs;
        filename = final_filename,
        schedule = IterationInterval(typemax(Int)),
        overwrite_existing = true
    )

    function save_final_snapshot(sim)
        @info "  Saving final 3D snapshot to: $final_filename"
        Oceananigans.OutputWriters.write_output!(sim.output_writers[:final_3d], sim)
    end
    simulation.callbacks[:final_snapshot] = Callback(save_final_snapshot, SpecifiedTimes(stop_time))

    sync_device!(arch)

    if verbose
        @info "  Starting simulation..."
    end

    start_time = time_ns()
    run!(simulation)
    sync_device!(arch)
    end_time = time_ns()

    wall_time_seconds = (end_time - start_time) / 1e9
    time_steps = iteration(simulation)
    time_per_step_seconds = wall_time_seconds / time_steps
    steps_per_second = time_steps / wall_time_seconds
    grid_points_per_second = total_points / time_per_step_seconds

    gpu_memory_used = arch isa GPU ? CUDACore.MemoryInfo().pool_used_bytes : 0
    metadata = BenchmarkMetadata(arch)

    result = SimulationResult(
        name,
        group,
        string(FT),
        (Nx, Ny, Nz),
        Float64(stop_time),
        time_steps,
        Float64(Δt),
        wall_time_seconds,
        time_per_step_seconds,
        steps_per_second,
        grid_points_per_second,
        output_filename,
        gpu_memory_used,
        metadata,
    )

    if verbose
        @info "  Simulation complete!"
        @info "    Wall time: $(@sprintf("%.1f", wall_time_seconds)) s ($(@sprintf("%.2f", wall_time_seconds / 3600)) hours)"
        @info "    Time steps: $time_steps"
        @info "    Time per step: $(@sprintf("%.6f", time_per_step_seconds)) s"
        @info "    Grid points/s: $(@sprintf("%.2e", grid_points_per_second))"
        @info "    Surface timeseries: $output_filename"
        @info "    Final 3D snapshot: $final_filename"
        if arch isa GPU
            @info "    GPU memory usage: $(Base.format_bytes(gpu_memory_used))"
        end
    end

    if arch isa GPU
        CUDA.reclaim()
    end

    return result
end

#####
##### IO benchmark (measuring performance with heavy 3D output)
#####

"""
    run_io_benchmark(model;
                     time_steps = 1440,
                     Δt = 60,
                     warmup_steps = 10,
                     output_iteration_interval = 1,
                     output_format = "jld2",
                     output_dir = ".",
                     name = "io_benchmark",
                     zarr_chunks = nothing,
                     verbose = true)

Run a benchmark that measures the performance impact of writing heavy 3D output.
Outputs 5 full 3D fields (u, v, w, T, S) at the specified `output_iteration_interval`.

The `output_format` can be `"jld2"`, `"netcdf"`, or `"zarr"`. When `output_format = "zarr"`,
`zarr_chunks` may be a tuple of spatial chunk sizes (matching the grid rank) or `nothing`
to let `ZarrWriter` pick the default.

Returns an `IOBenchmarkResult` containing timing information, output file path, total
output size, and (for Zarr) the resolved spatial chunk shape.
"""
function run_io_benchmark(model;
                          time_steps = 1440,
                          Δt = 60,
                          warmup_steps = 10,
                          output_iteration_interval = 1,
                          output_format = "jld2",
                          output_dir = ".",
                          name = "io_benchmark",
                          group = "",
                          zarr_chunks = nothing,
                          verbose = true)

    output_format in ("jld2", "netcdf", "zarr") ||
        error("Unknown output_format: $output_format. Use \"jld2\", \"netcdf\", or \"zarr\".")

    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    total_points = Nx * Ny * Nz

    extension = output_format == "jld2"   ? "jld2" :
                output_format == "netcdf" ? "nc"   :
                "zarr"
    timestamp = Dates.format(now(UTC), "yyyy-mm-dd_HHMMSS")
    output_filename = joinpath(output_dir, "$(name)_$(timestamp).$extension")

    if verbose
        @info "IO Benchmark: $name"
        @info "  Architecture: $arch"
        @info "  Float type: $FT"
        @info "  Grid size: $Nx × $Ny × $Nz ($total_points points)"
        @info "  Time step: $Δt s"
        @info "  Time steps: $time_steps"
        @info "  Warmup steps: $warmup_steps"
        @info "  Output format: $output_format"
        @info "  Output iteration interval: $output_iteration_interval"
        @info "  Output fields: u, v, w, T, S (full 3D)"
        @info "  Output file: $output_filename"
        if output_format == "zarr"
            @info "  Zarr chunks: $(isnothing(zarr_chunks) ? "auto" : string(zarr_chunks))"
        end
    end

    if verbose
        @info "  Running warmup..."
    end
    many_time_steps!(model, Δt, warmup_steps)
    sync_device!(arch)

    model.clock.iteration = 0
    model.clock.time = 0

    simulation = Simulation(model; Δt, stop_iteration=time_steps)

    output_field_names = (:u, :v, :w, :T, :S)
    outputs = Dict{Symbol, Any}()
    for field_name in output_field_names
        if haskey(model.velocities, field_name)
            outputs[field_name] = model.velocities[field_name]
        elseif hasproperty(model, :tracers) && haskey(model.tracers, field_name)
            outputs[field_name] = model.tracers[field_name]
        end
    end

    if output_format == "jld2"
        simulation.output_writers[:fields_3d] = JLD2Writer(model, outputs;
            filename = output_filename,
            schedule = IterationInterval(output_iteration_interval),
            overwrite_existing = true,
        )
    elseif output_format == "netcdf"
        simulation.output_writers[:fields_3d] = NetCDFWriter(model, outputs;
            filename = output_filename,
            schedule = IterationInterval(output_iteration_interval),
            overwrite_existing = true,
        )
    else  # zarr
        simulation.output_writers[:fields_3d] = ZarrWriter(model, outputs;
            filename = output_filename,
            schedule = IterationInterval(output_iteration_interval),
            overwrite_existing = true,
            chunks = zarr_chunks,
        )
    end

    if verbose
        wall_time_ref = Ref(time_ns())
        function progress(sim)
            elapsed = (time_ns() - wall_time_ref[]) / 1e9
            wall_time_ref[] = time_ns()
            @info @sprintf("  Step %d/%d, wall: %.1f s",
                           iteration(sim), time_steps, elapsed)
        end
        progress_interval = max(1, time_steps ÷ 10)
        simulation.callbacks[:progress] = Callback(progress, IterationInterval(progress_interval))
    end

    if verbose
        @info "  Starting IO benchmark..."
    end

    start_time = time_ns()
    run!(simulation)
    sync_device!(arch)
    end_time = time_ns()

    total_time_seconds = (end_time - start_time) / 1e9
    time_per_step_seconds = total_time_seconds / time_steps
    steps_per_second = time_steps / total_time_seconds
    grid_points_per_second = total_points / time_per_step_seconds

    total_output_size_bytes = path_size(output_filename)

    chunk_shape = output_format == "zarr" ?
                  zarr_chunk_shape(output_filename, "T") :
                  nothing

    gpu_memory_used = arch isa GPU ? CUDACore.MemoryInfo().pool_used_bytes : 0
    metadata = BenchmarkMetadata(arch)

    result = IOBenchmarkResult(
        name,
        group,
        string(FT),
        (Nx, Ny, Nz),
        time_steps,
        Float64(Δt),
        output_format,
        output_iteration_interval,
        total_time_seconds,
        time_per_step_seconds,
        steps_per_second,
        grid_points_per_second,
        output_filename,
        Int64(total_output_size_bytes),
        chunk_shape,
        gpu_memory_used,
        metadata,
    )

    if verbose
        @info "  IO Benchmark complete!"
        @info "    Total time: $(@sprintf("%.3f", total_time_seconds)) s"
        @info "    Time per step: $(@sprintf("%.6f", time_per_step_seconds)) s"
        @info "    Grid points/s: $(@sprintf("%.2e", grid_points_per_second))"
        @info "    Output file: $output_filename"
        @info "    Output size: $(Base.format_bytes(total_output_size_bytes))"
        if !isnothing(chunk_shape)
            @info "    Chunk shape: $(Tuple(chunk_shape))"
        end
        if arch isa GPU
            @info "    GPU memory usage: $(Base.format_bytes(gpu_memory_used))"
        end
    end

    if arch isa GPU
        CUDA.reclaim()
    end

    return result
end

#####
##### Read benchmark (measuring read performance of a previously-written store)
#####

"""
    run_read_benchmark(model;
                       time_steps = 100,
                       Δt = 60,
                       warmup_steps = 5,
                       output_iteration_interval = 1,
                       format = "zarr",
                       output_dir = ".",
                       name = "read_benchmark",
                       zarr_chunks = nothing,
                       read_variable = "T",
                       verbose = true)

Write a small dataset with the requested `format` (`"jld2"`, `"netcdf"`, or `"zarr"`),
then benchmark reading it back. Times the bulk `FieldTimeSeries(path, name)` construction
and a per-snapshot iteration over `fts[1..Nt]`. The `model` is used only for the setup
write; reads are timed independently.

Returns a `ReadBenchmarkResult`.
"""
function run_read_benchmark(model;
                            time_steps = 100,
                            Δt = 60,
                            warmup_steps = 5,
                            output_iteration_interval = 1,
                            format = "zarr",
                            output_dir = ".",
                            name = "read_benchmark",
                            group = "",
                            zarr_chunks = nothing,
                            read_variable = "T",
                            verbose = true)

    format in ("jld2", "netcdf", "zarr") ||
        error("Unknown format: $format. Use \"jld2\", \"netcdf\", or \"zarr\".")

    grid = model.grid
    arch = architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    total_points = Nx * Ny * Nz

    if verbose
        @info "Read Benchmark: $name"
        @info "  Architecture: $arch"
        @info "  Float type: $FT"
        @info "  Grid size: $Nx × $Ny × $Nz ($total_points points)"
        @info "  Time steps to write: $time_steps"
        @info "  Output iteration interval: $output_iteration_interval"
        @info "  Format: $format"
        @info "  Variable to read: $read_variable"
        @info "  Setting up: writing source dataset..."
    end

    io_result = run_io_benchmark(model;
        time_steps,
        Δt,
        warmup_steps,
        output_iteration_interval,
        output_format = format,
        output_dir,
        name = "$(name)_source",
        group,
        zarr_chunks,
        verbose = false,
    )

    output_filename = io_result.output_file
    file_size_bytes = io_result.total_output_size_bytes
    chunk_shape = io_result.chunk_shape

    if verbose
        @info "  Source dataset ready: $output_filename"
        @info "  Source dataset size: $(Base.format_bytes(file_size_bytes))"
        @info "  Starting read benchmark..."
    end

    sync_device!(arch)

    bulk_start = time_ns()
    field_time_series = FieldTimeSeries(output_filename, read_variable)
    bulk_end = time_ns()

    snapshots = length(field_time_series.times)

    iteration_start = time_ns()
    for n in 1:snapshots
        @inbounds field_time_series[n]
    end
    iteration_end = time_ns()

    bulk_read_seconds = (bulk_end - bulk_start) / 1e9
    iteration_seconds = (iteration_end - iteration_start) / 1e9
    time_per_snapshot_seconds = iteration_seconds / max(snapshots, 1)
    snapshots_per_second = snapshots / max(iteration_seconds, eps())
    grid_points_per_second = total_points / max(time_per_snapshot_seconds, eps())

    metadata = BenchmarkMetadata(arch)

    result = ReadBenchmarkResult(
        name,
        group,
        format,
        string(FT),
        (Nx, Ny, Nz),
        snapshots,
        output_filename,
        Int64(file_size_bytes),
        chunk_shape,
        bulk_read_seconds,
        iteration_seconds,
        time_per_snapshot_seconds,
        snapshots_per_second,
        grid_points_per_second,
        metadata,
    )

    if verbose
        @info "  Read Benchmark complete!"
        @info "    Snapshots: $snapshots"
        @info "    Bulk read: $(@sprintf("%.6f", bulk_read_seconds)) s"
        @info "    Iteration: $(@sprintf("%.6f", iteration_seconds)) s"
        @info "    Per-snapshot: $(@sprintf("%.6f", time_per_snapshot_seconds)) s"
        @info "    Snapshots/s: $(@sprintf("%.2f", snapshots_per_second))"
        @info "    Grid points/s: $(@sprintf("%.2e", grid_points_per_second))"
    end

    return result
end
