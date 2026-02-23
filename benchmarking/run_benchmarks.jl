#####
##### Oceananigans Benchmark Script
#####
##### This script runs benchmark cases with configurable parameters
##### via command-line arguments. Default device is GPU.
#####
##### Modes:
#####   - benchmark: Quick performance benchmarks (default)
#####   - simulate: Full runs with output for validation
#####   - io: IO-heavy benchmarks measuring 3D output performance
#####
##### Usage (benchmark mode):
#####   julia --project run_benchmarks.jl                                     # Default: 360x180x50, GPU, Float32
#####   julia --project run_benchmarks.jl --size=180x90x25                    # Smaller grid on GPU
#####   julia --project run_benchmarks.jl --size="180x90x25, 360x180x50"     # Multiple sizes
#####   julia --project run_benchmarks.jl --device=CPU --size=90x45x10        # CPU benchmark
#####
##### Usage (simulate mode):
#####   julia --project run_benchmarks.jl --mode=simulate --size=360x180x50 --stop_time=24.0
#####
##### Usage (io mode):
#####   julia --project run_benchmarks.jl --mode=io --size=360x180x50 --output_iteration_interval=1
#####   julia --project run_benchmarks.jl --mode=io --device=CPU --size=90x45x10 --time_steps=10 --output_iteration_interval=2
#####

using ArgParse: @add_arg_table!, ArgParseSettings, parse_args
using OceananigsBenchmarks: earth_ocean, benchmark_time_stepping, run_benchmark_simulation, run_io_benchmark
using JSON: JSON
using Oceananigans
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, SmagorinskyLilly,
    IsopycnalSkewSymmetricDiffusivity, HorizontalScalarBiharmonicDiffusivity

using Printf: @printf
using Dates: DateTime, now, UTC

#####
##### Argument parsing
#####

function parse_commandline()
    s = ArgParseSettings(
        description = "Run Oceananigans benchmarks with configurable parameters.",
        version = "0.1.0",
        add_version = true
    )

    @add_arg_table! s begin
        "--mode"
            help = "Mode: 'benchmark' for quick performance tests, 'simulate' for full runs with output, 'io' for IO-heavy benchmarks"
            arg_type = String
            default = "benchmark"

        "--size"
            help = "Grid size as NxxNyxNz (e.g., 360x180x50). " *
                   "Multiple sizes can be specified as comma-separated list."
            arg_type = String
            default = "360x180x50"

        "--device"
            help = "Device to run on: CPU or GPU"
            arg_type = String
            default = "GPU"

        "--case"
            help = "Benchmark case: earth_ocean"
            arg_type = String
            default = "earth_ocean"

        "--grid_type"
            help = "Grid type: tripolar, lat_lon (with bathymetry), or lat_lon_flat (no bathymetry)"
            arg_type = String
            default = "tripolar"

        "--float_type"
            help = "Floating point type: Float32 or Float64. " *
                   "Multiple types can be specified as comma-separated list."
            arg_type = String
            default = "Float32"

        "--momentum_advection"
            help = "Momentum advection scheme: WENOVectorInvariant5, WENOVectorInvariant9. " *
                   "Multiple schemes can be specified as comma-separated list."
            arg_type = String
            default = "WENOVectorInvariant9"

        "--tracer_advection"
            help = "Tracer advection scheme: WENO5, WENO7, WENO9, Centered2. " *
                   "Multiple schemes can be specified as comma-separated list."
            arg_type = String
            default = "WENO7"

        "--closure"
            help = "Turbulence closure: nothing, CATKE, SmagorinskyLilly, " *
                   "CATKE+Biharmonic, CATKE+GM+Biharmonic. " *
                   "Multiple closures can be specified as comma-separated list " *
                   "(use semicolons to separate multiple compound closures)."
            arg_type = String
            default = "CATKE"

        "--timestepper"
            help = "Time stepping scheme: SplitRungeKutta3, QuasiAdamsBashforth2"
            arg_type = String
            default = "SplitRungeKutta3"

        "--time_steps"
            help = "Number of time steps (benchmark mode only)"
            arg_type = Int
            default = 100

        "--warmup_steps"
            help = "Number of warmup time steps (benchmark mode only)"
            arg_type = Int
            default = 10

        "--dt"
            help = "Time step size in seconds"
            arg_type = Float64
            default = 60.0

        "--stop_time"
            help = "Simulation stop time in hours (simulate mode only)"
            arg_type = Float64
            default = 24.0

        "--output_interval"
            help = "Output interval in hours (simulate mode only)"
            arg_type = Float64
            default = 1.0

        "--output"
            help = "Output JSON filename for benchmark results"
            arg_type = String
            default = "benchmark_results.json"

        "--output_dir"
            help = "Directory for simulation output files (simulate and io modes)"
            arg_type = String
            default = "."

        "--output_iteration_interval"
            help = "Output iteration interval for IO benchmark mode (e.g., 1, 2, 6, 144)"
            arg_type = Int
            default = 1

        "--output_format"
            help = "Output file format for IO benchmark mode: jld2 or netcdf"
            arg_type = String
            default = "jld2"

        "--tracers"
            help = "Tracer names as comma-separated list (e.g., T,S or T,S,C1,C2,C3)"
            arg_type = String
            default = "T,S"

        "--clear"
            help = "Clear existing results file before writing"
            action = :store_true
    end

    return parse_args(s)
end

#####
##### Parsing utilities for comma-separated lists
#####

"""
    parse_list(str)

Parse a comma-separated string into a vector of trimmed strings.
"""
function parse_list(str::AbstractString)
    return [strip(s) for s in split(str, ",")]
end

"""
    parse_size(size_str)

Parse a size string into a tuple (Nx, Ny, Nz).
Format: "NxxNyxNz" (e.g., "360x180x50").
"""
function parse_size(size_str)
    parts = split(size_str, "x")
    length(parts) == 3 || error("Invalid size: $size_str. Use NxxNyxNz (e.g., 360x180x50).")
    return Tuple(parse(Int, p) for p in parts)
end

#####
##### Factory functions to create schemes from names
#####

make_architecture(name) = (@eval $(Symbol(name)))()
make_float_type(name) = @eval $(Symbol(name))

function make_momentum_advection(name, FT)
    name == "nothing" && return nothing

    m = match(r"^WENOVectorInvariant(\d+)$", name)
    if !isnothing(m)
        order = parse(Int, m[1])
        return WENOVectorInvariant(FT; order)
    end

    error("Unknown momentum advection: $name. Use WENOVectorInvariant5, WENOVectorInvariant9.")
end

function make_tracer_advection(name, FT)
    name == "nothing" && return nothing

    m = match(r"^([A-Za-z]+)(\d+)$", name)
    isnothing(m) && error("Unknown tracer advection: $name. Use WENO5, WENO7, WENO9, Centered2.")
    scheme = @eval $(Symbol(m[1]))
    order = parse(Int, m[2])
    return scheme(FT; order)
end

function make_closure(name, FT)
    name == "nothing" && return nothing
    name == "CATKE" && return CATKEVerticalDiffusivity()
    name == "SmagorinskyLilly" && return SmagorinskyLilly(FT)
    name == "CATKE+Biharmonic" && return (CATKEVerticalDiffusivity(),
                                          HorizontalScalarBiharmonicDiffusivity(ν=1e12))
    name == "CATKE+GM+Biharmonic" && return (CATKEVerticalDiffusivity(),
                                              IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3),
                                              HorizontalScalarBiharmonicDiffusivity(ν=1e12))
    error("Unknown closure: $name. Use nothing, CATKE, SmagorinskyLilly, CATKE+Biharmonic, CATKE+GM+Biharmonic.")
end

make_timestepper(name) = Symbol(name)

#####
##### Main benchmarking logic
#####

function run_benchmarks(args)
    mode = args["mode"]
    arch = make_architecture(args["device"])
    case = args["case"]
    grid_type = args["grid_type"]

    # Parse lists from arguments
    sizes = [parse_size(s) for s in parse_list(args["size"])]
    float_types = [make_float_type(s) for s in parse_list(args["float_type"])]
    momentum_advections = parse_list(args["momentum_advection"])
    tracer_advections = parse_list(args["tracer_advection"])
    closures = parse_list(args["closure"])
    timestepper = make_timestepper(args["timestepper"])
    tracers = Tuple(Symbol(strip(s)) for s in split(args["tracers"], ","))

    # Mode-specific parameters
    Δt = args["dt"]
    time_steps = args["time_steps"]
    warmup_steps = args["warmup_steps"]
    stop_time = args["stop_time"] * 3600  # Convert hours to seconds
    output_interval = args["output_interval"] * 3600  # Convert hours to seconds
    output_dir = args["output_dir"]
    output_iteration_interval = args["output_iteration_interval"]
    output_format = args["output_format"]

    # Default to 1440 time steps for IO mode when the user hasn't explicitly set it
    if mode == "io" && time_steps == 100
        time_steps = 1440
    end

    results = []

    println("=" ^ 95)
    println("Oceananigans Benchmark Suite")
    println("=" ^ 95)
    println("Date: ", now(UTC))
    println("Mode: ", mode)
    println("Case: ", case)
    println("Grid type: ", grid_type)
    println("Architecture: ", arch)
    println("Sizes: ", sizes)
    println("Float types: ", float_types)
    println("Momentum advection: ", momentum_advections)
    println("Tracer advection: ", tracer_advections)
    println("Closures: ", closures)
    println("Tracers: ", tracers)
    println("Timestepper: ", timestepper)
    if mode == "benchmark"
        println("Time steps: ", time_steps, " (warmup: ", warmup_steps, ")")
    elseif mode == "io"
        println("Time steps: ", time_steps, " (warmup: ", warmup_steps, ")")
        println("Output format: ", output_format)
        println("Output iteration interval: ", output_iteration_interval)
        println("Output fields: u, v, w, T, S (full 3D)")
    else
        println("Stop time: ", args["stop_time"], " hours")
        println("Output interval: ", args["output_interval"], " hours")
    end
    println("Δt: ", Δt, " s")
    println("=" ^ 95)
    println()

    # Loop over all combinations using Iterators.product
    for ((Nx, Ny, Nz), FT, mom_adv_name, trc_adv_name, cls_name) in
            Iterators.product(sizes, float_types, momentum_advections, tracer_advections, closures)

        # Build benchmark name
        size_str = "$(Nx)x$(Ny)x$(Nz)"
        ft_str = FT == Float32 ? "F32" : "F64"
        name = "EarthOcean_$(size_str)_$(ft_str)_$(mom_adv_name)_$(trc_adv_name)_$(cls_name)"

        println("\n", "-" ^ 70)
        println("Running: $name")
        println("-" ^ 70)

        # Create schemes
        momentum_advection = make_momentum_advection(mom_adv_name, FT)
        tracer_advection = make_tracer_advection(trc_adv_name, FT)
        closure = make_closure(cls_name, FT)

        # Create model
        if case == "earth_ocean"
            model = earth_ocean(arch;
                Nx, Ny, Nz,
                grid_type,
                float_type = FT,
                momentum_advection,
                tracer_advection,
                closure,
                tracers,
                timestepper
            )
        else
            error("Unknown case: $case")
        end

        # Run based on mode
        result = if mode == "benchmark"
            benchmark_time_stepping(model; time_steps, Δt, warmup_steps, name, verbose=true)
        elseif mode == "simulate"
            run_benchmark_simulation(model;
                stop_time, Δt, output_interval, output_dir, name, verbose=true)
        elseif mode == "io"
            run_io_benchmark(model;
                time_steps, Δt, warmup_steps, output_iteration_interval, output_format, output_dir, name, verbose=true)
        else
            error("Unknown mode: $mode. Use 'benchmark', 'simulate', or 'io'.")
        end
        push!(results, result)
    end

    return results
end

#####
##### Main entry point
#####

function main()
    args = parse_commandline()
    results = run_benchmarks(args)

    #####
    ##### Summary table
    #####

    println("\n", "=" ^ 105)
    println("BENCHMARK SUMMARY")
    println("=" ^ 105)
    println()

    @printf("%-55s %8s %12s %12s %10s %15s\n", "Benchmark", "Float", "Grid", "Time/Step", "Steps/s", "Points/s")
    println("-" ^ 105)

    for r in results
        grid_str = "$(r.grid_size[1])×$(r.grid_size[2])×$(r.grid_size[3])"
        @printf("%-55s %8s %12s %10.4f ms %10.2f %15.2e\n",
            r.name,
            r.float_type,
            grid_str,
            r.time_per_step_seconds * 1000,
            r.steps_per_second,
            r.grid_points_per_second
        )
    end

    println("=" ^ 105)

    #####
    ##### Save results to JSON
    #####

    if !isempty(results)
        output_file = args["output"]
        clear_file = args["clear"]

        # Load existing results or start fresh
        all_entries = if clear_file || !isfile(output_file)
            if clear_file && isfile(output_file)
                println("\nCleared existing results file: $output_file")
            end
            results
        else
            existing_data = JSON.parse(read(output_file))
            println("\nAppending to existing results file: $output_file")
            vcat(existing_data, results)
        end

        # Write all results to JSON
        open(output_file, "w") do io
            JSON.json(io, all_entries; pretty=true)
        end

        println("Results saved to: $output_file ($(length(results)) new, $(length(all_entries)) total)")

        # Generate markdown report from the full JSON data
        md_file = replace(output_file, ".json" => ".md")
        generate_markdown_report(md_file, JSON.parse(read(output_file)))
        println("Markdown report saved to: $md_file")
    end

    println("Benchmarks completed at ", now(UTC), "Z")
end

"""
Generate a markdown report from benchmark results.
"""
function generate_markdown_report(filename, entries)
    open(filename, "w") do io
        println(io, "# Oceananigans Benchmark Results")
        println(io)

        if !isempty(entries)
            metadata = entries[end]["metadata"]

            println(io, "## System Information")
            println(io)
            println(io, "| Property | Value |")
            println(io, "|----------|-------|")
            println(io, "| Julia | ", metadata["julia_version"], " |")
            println(io, "| Oceananigans | ", metadata["oceananigans_version"], " |")
            println(io, "| NumericalEarth | ", metadata["numericalearth_version"], " |")
            println(io, "| Architecture | ", metadata["architecture"], " |")
            println(io, "| CPU | ", metadata["cpu_model"], " |")
            println(io, "| Threads | ", metadata["num_threads"], " |")
            if !isnothing(metadata["gpu_name"])
                println(io, "| GPU | ", metadata["gpu_name"], " |")
                println(io, "| CUDA | ", metadata["cuda_version"], " |")
            end
            println(io, "| Hostname | ", metadata["hostname"], " |")
            println(io)
        end

        println(io, "## Results")
        println(io)
        println(io, "| Benchmark | Float | Grid | Time/Step (ms) | Steps/s | Points/s | Timestamp |")
        println(io, "|-----------|-------|------|----------------|---------|----------|-----------|")

        for entry in entries
            grid = entry["grid_size"]
            grid_str = "$(grid[1])×$(grid[2])×$(grid[3])"
            timestamp = entry["metadata"]["timestamp"]

            @printf(io, "| `%s` | %s | %s | %.2f | %.2f | %.2e | %s |\n",
                    entry["name"],
                    entry["float_type"],
                    grid_str,
                    entry["time_per_step_seconds"] * 1000,
                    entry["steps_per_second"],
                    entry["grid_points_per_second"],
                    timestamp)
        end
    end
end

# Run when invoked as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
