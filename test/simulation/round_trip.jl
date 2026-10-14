include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: topology
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid, ConformalCubedSpherePanelGrid

# Load only the writer extension packages needed for the selected writers, so a
# subset run (e.g. FTS_ROUND_TRIP_WRITERS=jld2) needs neither Zarr nor NCDatasets.
const FTS_ROUND_TRIP_WRITER_NAMES = split(get(ENV, "FTS_ROUND_TRIP_WRITERS", "jld2,netcdf,zarr"), ",")
"netcdf" in FTS_ROUND_TRIP_WRITER_NAMES && @eval using NCDatasets
"zarr"   in FTS_ROUND_TRIP_WRITER_NAMES && @eval using Zarr

#####
##### Round-trip harness: write output, reconstruct a FieldTimeSeries, compare
#####
##### One axis is the grid (the four categories that lacked coverage: stretched
##### rectilinear, immersed, stretched lat-lon, orthogonal spherical shell); the
##### other axis is the output writer (JLD2, NetCDF, Zarr). Every pair is exercised
##### with a plain `@test`; pairs the writer/reader cannot yet round-trip simply
##### fail, which makes the remaining gaps explicit.
#####

cᵢ(x, y, z) = exp(-(z / 10)^2) * cos(deg2rad(x)) * cos(deg2rad(y))

# Grids span the four reconstruction-sensitive categories. Each is built small to
# keep CI cheap; tripolar uses sizes known to be well-behaved over a couple steps.
function round_trip_grids(arch)
    stretched_z = [-1.0, -0.7, -0.4, -0.1, 0.0]

    stretched_rectilinear = RectilinearGrid(arch, size=(4, 4, 4),
                                            x=(0, 1), y=(0, 1), z=stretched_z,
                                            topology=(Periodic, Periodic, Bounded))

    underlying_rectilinear = RectilinearGrid(arch, size=(4, 4, 4),
                                             x=(0, 1), y=(0, 1), z=(-1, 0),
                                             topology=(Periodic, Periodic, Bounded))
    immersed_rectilinear = ImmersedBoundaryGrid(underlying_rectilinear, GridFittedBottom((x, y) -> -0.5))

    stretched_z_deep = [-1000.0, -600.0, -300.0, -100.0, 0.0]
    stretched_latlon = LatitudeLongitudeGrid(arch, size=(4, 4, 4),
                                             longitude=(0, 60), latitude=(-10, 10), z=stretched_z_deep,
                                             topology=(Bounded, Bounded, Bounded))

    tripolar = TripolarGrid(arch, size=(12, 10, 3), z=(-100, 0))
    immersed_tripolar = ImmersedBoundaryGrid(TripolarGrid(arch, size=(12, 10, 3), z=(-100, 0)),
                                             GridFittedBottom((λ, φ) -> -50))

    cubed_sphere_panel = ConformalCubedSpherePanelGrid(arch, size=(8, 8, 3), z=(-100, 0), halo=(2, 2, 2))

    return [("stretched_rectilinear", stretched_rectilinear),
            ("immersed_rectilinear",  immersed_rectilinear),
            ("stretched_latlon",      stretched_latlon),
            ("tripolar",              tripolar),
            ("immersed_tripolar",     immersed_tripolar),
            ("cubed_sphere_panel",    cubed_sphere_panel)]
end

# Each writer needs its own filename convention, and the warnings its reader emits.
# Set ENV["FTS_ROUND_TRIP_WRITERS"] (e.g. "jld2" or "jld2,netcdf") to run a subset.
netcdf_reader_logs = ((:warn, "Reading boundary conditions from NetCDF files is not supported for FieldTimeSeries. " *
                              "Using default FieldBoundaryConditions for `grid` and `location`."),)
zarr_reader_logs = ((:warn, "Reading boundary conditions from Zarr stores is not supported. " *
                            "Using default FieldBoundaryConditions for `grid` and `location`."),)

function round_trip_writers()
    writers = [(name="jld2",   Writer=JLD2Writer,   filename=base -> base,         reader_logs=()),
               (name="netcdf", Writer=NetCDFWriter, filename=base -> base * ".nc", reader_logs=netcdf_reader_logs),
               (name="zarr",   Writer=ZarrWriter,   filename=base -> base,         reader_logs=zarr_reader_logs)]
    return filter(w -> w.name in FTS_ROUND_TRIP_WRITER_NAMES, writers)
end

# The FFT-based pressure solver is only approximate on immersed boundary grids, and warns about it
pressure_solver_logs(grid) = grid isa ImmersedBoundaryGrid ?
    ((:warn, r"^The FFT-based pressure_solver for NonhydrostaticModels on ImmersedBoundaryGrid"),) : ()

function round_trip_model(grid)
    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    if underlying isa RectilinearGrid
        return @test_logs pressure_solver_logs(grid)... NonhydrostaticModel(grid; tracers=:c)
    else
        free_surface = SplitExplicitFreeSurface(grid; substeps=10)
        return HydrostaticFreeSurfaceModel(grid; free_surface, tracers=:c)
    end
end

function grids_match(reconstructed, original)
    size(reconstructed) == size(original) || return false
    topology(reconstructed) == topology(original) || return false
    typeof(reconstructed).name === typeof(original).name || return false
    if original isa ImmersedBoundaryGrid
        typeof(reconstructed.underlying_grid).name === typeof(original.underlying_grid).name || return false
    end
    return true
end

# Masked cells come back as NaN on immersed grids, so compare NaN-to-NaN as equal.
function values_match(fts, snapshots; atol=1e-4)
    length(fts.times) == length(snapshots) || return false
    for k in eachindex(snapshots)
        reconstructed = Array(interior(fts[k]))
        reference = snapshots[k]
        size(reconstructed) == size(reference) || return false
        for (a, b) in zip(reconstructed, reference)
            ((isnan(a) && isnan(b)) || isapprox(a, b; atol)) || return false
        end
    end
    return true
end

function field_time_series_round_trips(grid_name, grid, writer_spec, arch)
    base = "fts_round_trip_$(grid_name)_$(writer_spec.name)_$(typeof(arch))"
    dir = mktempdir()

    model = round_trip_model(grid)
    set!(model, c=cᵢ)

    snapshots = Array{eltype(grid), 3}[]
    sim = Simulation(model; Δt=1, stop_iteration=2, verbose=false)
    sim.callbacks[:save] = Callback(s -> push!(snapshots, Array(interior(s.model.tracers.c))), IterationInterval(1))
    sim.output_writers[:writer] = writer_spec.Writer(model, (; c=model.tracers.c);
                                                      dir,
                                                      filename=writer_spec.filename(base),
                                                      schedule=IterationInterval(1),
                                                      overwrite_files=true)
    run!(sim)

    path = sim.output_writers[:writer].filepath
    fts = @test_logs writer_spec.reader_logs... FieldTimeSeries(path, "c"; architecture=arch)
    matches = grids_match(fts.grid, grid) && values_match(fts, snapshots)

    rm(dir; recursive=true)

    return matches
end

@testset "FieldTimeSeries round-trip across writers and grids" begin
    for arch in archs
        @testset "$grid_name × $(writer_spec.name) [$(typeof(arch))]" for (grid_name, grid) in round_trip_grids(arch), writer_spec in round_trip_writers()
            @test field_time_series_round_trips(grid_name, grid, writer_spec, arch)
        end
    end
end
