
#####
##### Benchmark utilities
#####

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
                                 verbose = true)

    grid = model.grid
    arch = Oceananigans.Architectures.architecture(grid)
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
    synchronize_device(arch)

    # Benchmark phase
    if verbose
        @info "  Running benchmark..."
    end
    start_time = time_ns()
    many_time_steps!(model, Δt, time_steps)
    synchronize_device(arch)
    end_time = time_ns()

    total_time_seconds = (end_time - start_time) / 1e9
    time_per_step_seconds = total_time_seconds / time_steps
    steps_per_second = time_steps / total_time_seconds
    grid_points_per_second = total_points / time_per_step_seconds

    gpu_memory_used = arch isa GPU ? CUDA.MemoryInfo().pool_used_bytes : 0
    metadata = BenchmarkMetadata(arch)

    result = BenchmarkResult(
        name,
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
                                  output_fields = (:u, :v, :T, :S),
                                  verbose = true)

    grid = model.grid
    arch = Oceananigans.Architectures.architecture(grid)
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

    synchronize_device(arch)

    if verbose
        @info "  Starting simulation..."
    end

    start_time = time_ns()
    run!(simulation)
    synchronize_device(arch)
    end_time = time_ns()

    wall_time_seconds = (end_time - start_time) / 1e9
    time_steps = iteration(simulation)
    time_per_step_seconds = wall_time_seconds / time_steps
    steps_per_second = time_steps / wall_time_seconds
    grid_points_per_second = total_points / time_per_step_seconds

    gpu_memory_used = arch isa GPU ? CUDA.MemoryInfo().pool_used_bytes : 0
    metadata = BenchmarkMetadata(arch)

    result = SimulationResult(
        name,
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
##### Device synchronization
#####

synchronize_device(::Oceananigans.Architectures.CPU) = nothing

function synchronize_device(::Oceananigans.Architectures.GPU)
    CUDA.synchronize()
    return nothing
end
