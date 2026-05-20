include("dependencies_for_runtests.jl")

using Zarr

#####
##### ZarrWriter smoke tests (Phase 1: construction + show, no I/O)
#####

@testset "ZarrWriter [skeleton]" begin
    @info "  Testing ZarrWriter construction and kwarg surface..."

    for arch in archs
        grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))

        # NamedTuple of fields
        writer_nt = ZarrWriter(model, model.velocities;
                               filename = "test_zarr_nt",
                               schedule = TimeInterval(1),
                               dir = ".",
                               overwrite_existing = true)
        @test writer_nt isa ZarrWriter
        @test length(writer_nt.outputs) == 3

        # Dict of fields
        outputs_dict = Dict("u" => model.velocities.u, "T" => model.tracers.T)
        writer_dict = ZarrWriter(model, outputs_dict;
                                 filename = "test_zarr_dict",
                                 schedule = IterationInterval(1),
                                 dir = ".",
                                 overwrite_existing = true)
        @test writer_dict isa ZarrWriter
        @test length(writer_dict.outputs) == 2

        # Full kwarg surface
        writer_full = ZarrWriter(model, (; u=model.velocities.u);
                                 filename = "test_zarr_full",
                                 schedule = TimeInterval(1),
                                 dir = ".",
                                 indices = (:, :, 1),
                                 with_halos = false,
                                 array_type = Array{Float64},
                                 overwrite_existing = true,
                                 verbose = true,
                                 part = 1,
                                 chunks = (4, 4, 1, 1),
                                 compressor = Zarr.BloscCompressor(clevel=3))
        @test writer_full isa ZarrWriter
        @test writer_full.with_halos == false
        @test writer_full.array_type == Array{Float64}
        @test writer_full.chunks == (4, 4, 1, 1)
        @test writer_full.compressor isa Zarr.BloscCompressor

        # Zarr-specific: user-supplied store (DictStore)
        dict_store = Zarr.DictStore()
        writer_dict_store = ZarrWriter(model, (; u=model.velocities.u);
                                       store = dict_store,
                                       schedule = TimeInterval(1),
                                       overwrite_existing = true)
        @test writer_dict_store isa ZarrWriter
        @test writer_dict_store.store === dict_store

        # ZipStore rejection
        # ZipStore needs a byte vector — make an empty one
        @test_throws ArgumentError ZarrWriter(model, (; u=model.velocities.u);
                                              store = Zarr.ZipStore(UInt8[]),
                                              schedule = TimeInterval(1))

        # Missing filename + missing store
        @test_throws ArgumentError ZarrWriter(model, (; u=model.velocities.u);
                                              schedule = TimeInterval(1))

        # show should not error
        io = IOBuffer()
        show(io, writer_full)
        @test occursin("ZarrWriter", String(take!(io)))

        # summary should not error
        @test occursin("ZarrWriter", summary(writer_nt))
    end
end

#####
##### Phase 2 — Time-axis writing + raw round-trip
#####

@testset "ZarrWriter [round-trip]" begin
    @info "  Testing ZarrWriter round-trip via raw Zarr.zopen..."

    for arch in archs
        grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        model = NonhydrostaticModel(grid; tracers=:c)

        # Seed fields with known values. Use fully-Periodic topology so boundary
        # conditions don't override the set value at any cell.
        set!(model, u=(x, y, z) -> 1.0,
                    v=(x, y, z) -> 2.0,
                    w=(x, y, z) -> 3.0,
                    c=(x, y, z) -> 4.0)

        zarrpath = abspath(joinpath(".", "test_zarr_roundtrip.zarr"))
        isdir(zarrpath) && rm(zarrpath; recursive=true, force=true)

        simulation = Simulation(model, Δt=1.0, stop_iteration=2)
        simulation.output_writers[:fields] = ZarrWriter(model, merge(model.velocities, model.tracers);
                                                       filename = "test_zarr_roundtrip",
                                                       dir = ".",
                                                       schedule = IterationInterval(1),
                                                       overwrite_existing = true,
                                                       with_halos = false)
        run!(simulation)

        @test isdir(zarrpath)

        # Read back via raw Zarr.zopen
        g = Zarr.zopen(zarrpath)

        # time array
        @test "time" in keys(g.arrays)
        times = g["time"][:]
        @test length(times) == 3                       # initial + 2 iterations
        @test times ≈ [0.0, 1.0, 2.0]

        # Each velocity component is a 4D Zarr array (Nx, Ny, Nz, Nt)
        for (name, expected_val) in (("u", 1.0), ("v", 2.0), ("w", 3.0), ("c", 4.0))
            @test name in keys(g.arrays)
            arr = g[name]
            @test ndims(arr) == 4
            @test size(arr, 4) == 3
            # Spatial size matches grid (no halos)
            data = arr[:, :, :, 1]
            @test all(data .≈ Float32(expected_val))

            # _ARRAY_DIMENSIONS attribute is set and reversed (C-order)
            dims_attr = arr.attrs["_ARRAY_DIMENSIONS"]
            @test dims_attr[1] == "time"    # first (slowest-varying) dim in C order
            @test length(dims_attr) == 4
        end

        rm(zarrpath; recursive=true, force=true)
    end
end

#####
##### Phase 3 — Operations, reductions, functions, WindowedTimeAverage
#####

@testset "ZarrWriter [operations, reductions, functions, WindowedTimeAverage]" begin
    @info "  Testing ZarrWriter with non-Field outputs..."

    for arch in archs
        grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        model = NonhydrostaticModel(grid; tracers=:c)

        set!(model, u=(x, y, z) -> 1.0,
                    v=(x, y, z) -> 2.0,
                    c=(x, y, z) -> 4.0)

        # Reduction: column-mean of c
        c_avg = Field(Average(model.tracers.c, dims=(1, 2)))

        # AbstractOperation: u + v
        u_plus_v = model.velocities.u + model.velocities.v

        # Function: scalar
        f_scalar(model) = model.clock.time^2

        # Function: profile (Nz-vector)
        zC = znodes(grid, Center())
        f_profile(model) = collect(model.clock.time .* exp.(zC))

        # WindowedTimeAverage over a Field
        outputs = (c=model.tracers.c,
                   c_avg=c_avg,
                   u_plus_v=u_plus_v,
                   scalar_f=f_scalar,
                   profile_f=f_profile)

        zarrpath = abspath(joinpath(".", "test_zarr_ops.zarr"))
        isdir(zarrpath) && rm(zarrpath; recursive=true, force=true)

        # WindowedTimeAverage requires AveragedTimeInterval schedule
        wta_outputs = (c_wta=model.tracers.c,)
        wta_path = abspath(joinpath(".", "test_zarr_wta.zarr"))
        isdir(wta_path) && rm(wta_path; recursive=true, force=true)

        simulation = Simulation(model, Δt=1.0, stop_iteration=2)
        simulation.output_writers[:ops] =
            ZarrWriter(model, outputs;
                       filename = "test_zarr_ops",
                       dir = ".",
                       schedule = IterationInterval(1),
                       overwrite_existing = true,
                       with_halos = false,
                       dimensions = Dict("scalar_f" => (), "profile_f" => ("z_aac",)))

        simulation.output_writers[:wta] =
            ZarrWriter(model, wta_outputs;
                       filename = "test_zarr_wta",
                       dir = ".",
                       schedule = AveragedTimeInterval(1.0, window=1.0),
                       overwrite_existing = true,
                       with_halos = false)

        run!(simulation)

        # --- Verify ops store ---
        g = Zarr.zopen(zarrpath)

        # Reduction (Average over (1, 2)) → shape (1, 1, Nz, Nt)
        @test "c_avg" in keys(g.arrays)
        c_avg_arr = g["c_avg"]
        @test size(c_avg_arr) == (1, 1, 4, 3)
        # c was set to 4.0 everywhere, so the column-average is 4.0
        @test all(c_avg_arr[:, :, :, 1] .≈ Float32(4.0))

        # AbstractOperation u+v → shape matches the operand grid
        @test "u_plus_v" in keys(g.arrays)
        upv_arr = g["u_plus_v"]
        @test ndims(upv_arr) == 4
        @test size(upv_arr, 4) == 3
        @test all(upv_arr[:, :, :, 1] .≈ Float32(3.0))   # 1 + 2

        # Scalar function output
        @test "scalar_f" in keys(g.arrays)
        scalar_arr = g["scalar_f"]
        @test size(scalar_arr) == (3,)        # only time
        @test scalar_arr[1] ≈ 0.0             # t=0 → 0^2
        @test scalar_arr[3] ≈ 4.0             # t=2 → 2^2

        # Profile function output
        @test "profile_f" in keys(g.arrays)
        profile_arr = g["profile_f"]
        @test size(profile_arr) == (4, 3)     # (Nz, Nt)
        @test all(profile_arr[:, 1] .≈ 0.0)   # t=0 → 0 * exp(zC)
        @test profile_arr[:, 3] ≈ Float32.(2.0 .* exp.(zC))

        # _ARRAY_DIMENSIONS contains user-supplied dim name + time (reversed)
        @test profile_arr.attrs["_ARRAY_DIMENSIONS"] == ["time", "z_aac"]
        @test scalar_arr.attrs["_ARRAY_DIMENSIONS"] == ["time"]

        # --- WTA store ---
        g_wta = Zarr.zopen(wta_path)
        @test "c_wta" in keys(g_wta.arrays)
        @test size(g_wta["c_wta"], 4) >= 1
        @test all(g_wta["c_wta"][:, :, :, 1] .≈ Float32(4.0))

        # Missing dimensions for function output → error
        bad_outputs = (h=f_scalar,)
        bad_path = abspath(joinpath(".", "test_zarr_bad.zarr"))
        isdir(bad_path) && rm(bad_path; recursive=true, force=true)
        simulation2 = Simulation(NonhydrostaticModel(grid; tracers=:c), Δt=1.0, stop_iteration=1)
        simulation2.output_writers[:bad] =
            ZarrWriter(simulation2.model, bad_outputs;
                       filename = "test_zarr_bad",
                       dir = ".",
                       schedule = IterationInterval(1),
                       overwrite_existing = true)
        @test_throws ArgumentError run!(simulation2)

        rm(zarrpath; recursive=true, force=true)
        rm(wta_path; recursive=true, force=true)
        isdir(bad_path) && rm(bad_path; recursive=true, force=true)
    end
end

#####
##### Phase 4 — Grid reconstruction + multi-grid
#####

using Oceananigans.OutputWriters: ZarrWriter
using Oceananigans.Fields: Field

@testset "ZarrWriter [grid reconstruction + multi-grid]" begin
    @info "  Testing grid reconstruction and multi-grid support..."

    ZarrExt = Base.get_extension(Oceananigans, :OceananigansZarrExt)
    reconstruct_zarr_grid = ZarrExt.reconstruct_zarr_grid

    for arch in archs
        # --- Single grid: writes to grid/.zattrs ---
        grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 2, 3),
                               topology=(Periodic, Periodic, Bounded))
        model = NonhydrostaticModel(grid; tracers=:c)

        zarrpath = abspath(joinpath(".", "test_zarr_grid.zarr"))
        isdir(zarrpath) && rm(zarrpath; recursive=true, force=true)

        simulation = Simulation(model, Δt=1.0, stop_iteration=1)
        simulation.output_writers[:fields] = ZarrWriter(model, (; u=model.velocities.u);
                                                       filename = "test_zarr_grid",
                                                       dir = ".",
                                                       schedule = IterationInterval(1),
                                                       overwrite_existing = true)
        run!(simulation)

        # `grid/` subgroup exists for single-grid writer (no suffix)
        g = Zarr.zopen(zarrpath)
        @test "grid" in keys(g.groups)
        attrs = g.groups["grid"].attrs
        @test haskey(attrs, "underlying_grid_reconstruction_args")
        @test haskey(attrs, "underlying_grid_reconstruction_kwargs")
        @test haskey(attrs, "grid_reconstruction_metadata")
        @test attrs["grid_reconstruction_metadata"]["underlying_grid_type"] == "RectilinearGrid"

        # Single-grid → no grid_index attribute on outputs
        @test !haskey(g["u"].attrs, "grid_index")

        # Round-trip: rebuild the grid
        reconstructed = reconstruct_zarr_grid(g; architecture=arch)
        @test reconstructed isa RectilinearGrid
        @test size(reconstructed) == size(grid)
        @test topology(reconstructed) == topology(grid)

        rm(zarrpath; recursive=true, force=true)

        # --- Multi-grid: outputs on two grids end up under grid_1/, grid_2/ ---
        coarse_grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 1, 1),
                                      topology=(Periodic, Periodic, Bounded))
        coarse_u    = Field{Face, Center, Center}(coarse_grid)
        set!(coarse_u, (x, y, z) -> 5.0)

        multi_outputs = (u_fine=model.velocities.u, u_coarse=coarse_u)
        multi_path = abspath(joinpath(".", "test_zarr_multigrid.zarr"))
        isdir(multi_path) && rm(multi_path; recursive=true, force=true)

        # Reset model clock for fresh sim
        model.clock.iteration = 0
        model.clock.time = 0.0
        simulation2 = Simulation(model, Δt=1.0, stop_iteration=1)
        simulation2.output_writers[:multi] = ZarrWriter(model, multi_outputs;
                                                       filename = "test_zarr_multigrid",
                                                       dir = ".",
                                                       schedule = IterationInterval(1),
                                                       overwrite_existing = true)
        run!(simulation2)

        gm = Zarr.zopen(multi_path)
        @test "grid_1" in keys(gm.groups)
        @test "grid_2" in keys(gm.groups)
        @test !("grid" in keys(gm.groups))

        # Each output is tagged with grid_index
        @test gm["u_fine"].attrs["grid_index"] in (1, 2)
        @test gm["u_coarse"].attrs["grid_index"] in (1, 2)
        @test gm["u_fine"].attrs["grid_index"] != gm["u_coarse"].attrs["grid_index"]

        # Reconstruct both grids
        idx_fine   = gm["u_fine"].attrs["grid_index"]
        idx_coarse = gm["u_coarse"].attrs["grid_index"]
        rg_fine   = reconstruct_zarr_grid(gm; grid_index=idx_fine, architecture=arch)
        rg_coarse = reconstruct_zarr_grid(gm; grid_index=idx_coarse, architecture=arch)
        @test size(rg_fine)   == (4, 4, 4)
        @test size(rg_coarse) == (2, 2, 2)

        rm(multi_path; recursive=true, force=true)
    end
end

#####
##### Phase 5 — FieldTimeSeries Zarr reader
#####

@testset "ZarrWriter [FieldTimeSeries reader]" begin
    @info "  Testing FieldTimeSeries(path, name) for Zarr stores..."

    for arch in archs
        grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 2, 3),
                               topology=(Periodic, Periodic, Periodic))
        model = NonhydrostaticModel(grid; tracers=:c)

        set!(model, u=(x, y, z) -> 1.5,
                    v=(x, y, z) -> -2.5,
                    c=(x, y, z) -> 3.7)

        zarrpath = abspath(joinpath(".", "test_zarr_fts.zarr"))
        isdir(zarrpath) && rm(zarrpath; recursive=true, force=true)

        simulation = Simulation(model, Δt=0.5, stop_iteration=3)
        simulation.output_writers[:fields] = ZarrWriter(model, (u=model.velocities.u,
                                                                v=model.velocities.v,
                                                                c=model.tracers.c);
                                                       filename = "test_zarr_fts",
                                                       dir = ".",
                                                       schedule = IterationInterval(1),
                                                       overwrite_existing = true,
                                                       with_halos = false)
        run!(simulation)

        # Read back u via FieldTimeSeries
        u_fts = FieldTimeSeries(zarrpath, "u")
        @test u_fts isa FieldTimeSeries
        @test length(u_fts.times) == 4
        @test u_fts.times ≈ [0.0, 0.5, 1.0, 1.5]
        @test size(u_fts.grid) == size(grid)

        # First step values match the seed
        u0 = Array(interior(u_fts[1]))
        @test all(u0 .≈ Float32(1.5))

        # v has location (Center, Face, Center)
        v_fts = FieldTimeSeries(zarrpath, "v")
        @test location(v_fts) == (Center, Face, Center)
        v0 = Array(interior(v_fts[1]))
        @test all(v0 .≈ Float32(-2.5))

        # c (tracer at Center, Center, Center)
        c_fts = FieldTimeSeries(zarrpath, "c")
        @test location(c_fts) == (Center, Center, Center)
        c0 = Array(interior(c_fts[1]))
        @test all(c0 .≈ Float32(3.7))

        # Pass explicit architecture
        u_fts_arch = FieldTimeSeries(zarrpath, "u"; architecture=arch)
        @test u_fts_arch isa FieldTimeSeries

        rm(zarrpath; recursive=true, force=true)
    end
end

#####
##### Phase 6 — File splitting, append, checkpoint+restart
#####

@testset "ZarrWriter [append + restart]" begin
    @info "  Testing ZarrWriter append-on-existing-store semantics..."

    for arch in archs
        grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        model = NonhydrostaticModel(grid; tracers=:c)
        set!(model, u=(x, y, z) -> 0.5, c=(x, y, z) -> 9.0)

        zarrpath = abspath(joinpath(".", "test_zarr_append.zarr"))
        isdir(zarrpath) && rm(zarrpath; recursive=true, force=true)

        # --- Run 1: 3 steps with a fresh store ---
        sim1 = Simulation(model, Δt=1.0, stop_iteration=3)
        sim1.output_writers[:fields] = ZarrWriter(model, (; u=model.velocities.u, c=model.tracers.c);
                                                  filename = "test_zarr_append",
                                                  dir = ".",
                                                  schedule = IterationInterval(1),
                                                  overwrite_existing = true,
                                                  with_halos = false)
        run!(sim1)
        @test isdir(zarrpath)
        @test length(Zarr.zopen(zarrpath)["time"][:]) == 4   # initial + 3 iterations

        # --- Run 2: model.clock not reset → new writer with overwrite=false appends ---
        # (Simulates a continued simulation: same in-memory model, new writer pointing at
        # the same path, the previous writer was dropped.)
        sim2 = Simulation(model, Δt=1.0, stop_iteration=6)
        sim2.output_writers[:fields] = ZarrWriter(model, (; u=model.velocities.u, c=model.tracers.c);
                                                  filename = "test_zarr_append",
                                                  dir = ".",
                                                  schedule = IterationInterval(1),
                                                  overwrite_existing = false,        # APPEND
                                                  with_halos = false)
        run!(sim2)

        g = Zarr.zopen(zarrpath)
        times = g["time"][:]
        @test length(times) == 7
        @test issorted(times)
        @test allunique(times)
        @test times ≈ [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        # FieldTimeSeries reads the full 7-step series
        u_fts = FieldTimeSeries(zarrpath, "u")
        @test length(u_fts.times) == 7

        # --- Dtype validation on restart ---
        bad_model = NonhydrostaticModel(grid; tracers=:c)
        bad_model.clock.iteration = 7   # pretend we're continuing
        bad_writer = ZarrWriter(bad_model, (; u=bad_model.velocities.u);
                                filename = "test_zarr_append",
                                dir = ".",
                                schedule = IterationInterval(1),
                                overwrite_existing = false,
                                array_type = Array{Float64})
        @test_throws ArgumentError Oceananigans.initialize!(bad_writer, bad_model)

        rm(zarrpath; recursive=true, force=true)
    end
end

#####
##### Phase 7 — Alternative stores: DictStore (in-memory) + ZipStore (read)
#####

@testset "ZarrWriter [alternative stores]" begin
    @info "  Testing ZarrWriter with DictStore (memory) + ZipStore (read after finalize)..."

    for arch in archs
        grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        model = NonhydrostaticModel(grid; tracers=:c)
        set!(model, u=(x, y, z) -> 0.5)

        # --- DictStore: writer runs entirely in memory ---
        dict_store = Zarr.DictStore()
        sim = Simulation(model, Δt=1.0, stop_iteration=2)
        sim.output_writers[:fields] = ZarrWriter(model, (; u=model.velocities.u);
                                                 store = dict_store,
                                                 schedule = IterationInterval(1),
                                                 overwrite_existing = false)
        run!(sim)
        g_dict = Zarr.zopen(dict_store)
        @test "time" in keys(g_dict.arrays)
        @test "u" in keys(g_dict.arrays)
        @test length(g_dict["time"][:]) == 3
        @test all(g_dict["u"][:, :, :, 1] .≈ Float32(0.5))

        # --- ZipStore: write to a DirectoryStore, finalize to .zip, read back ---
        zarrpath = abspath(joinpath(".", "test_zarr_zip.zarr"))
        zippath  = abspath(joinpath(".", "test_zarr_zip.zip"))
        isdir(zarrpath) && rm(zarrpath; recursive=true, force=true)
        isfile(zippath) && rm(zippath; force=true)

        model.clock.iteration = 0
        model.clock.time = 0.0
        sim2 = Simulation(model, Δt=1.0, stop_iteration=2)
        sim2.output_writers[:fields] = ZarrWriter(model, (; u=model.velocities.u);
                                                  filename = "test_zarr_zip",
                                                  dir = ".",
                                                  schedule = IterationInterval(1),
                                                  overwrite_existing = true)
        run!(sim2)
        open(zippath, "w") do io
            Zarr.writezip(io, sim2.output_writers[:fields].store)
        end
        @test isfile(zippath)

        # FieldTimeSeries reads a zip-finalized store via the .zip extension dispatch.
        u_fts = FieldTimeSeries(zippath, "u")
        @test u_fts isa FieldTimeSeries
        @test length(u_fts.times) == 3
        @test all(Array(interior(u_fts[1])) .≈ Float32(0.5))

        # Writer rejects ZipStore at construction.
        @test_throws ArgumentError ZarrWriter(model, (; u=model.velocities.u);
                                              store = Zarr.ZipStore(read(zippath)),
                                              schedule = IterationInterval(1))

        rm(zarrpath; recursive=true, force=true)
        rm(zippath; force=true)
    end
end

#####
##### Phase 8 — Grid-type sweep: LatLon, OSSG (Tripolar, RotatedLatLon), ImmersedBoundaryGrid
#####
##### Single-rank round-trip per grid type. Asserts both data write and grid
##### serialization round-trip via `FieldTimeSeries(path, name)`.
#####
##### Known gaps (see PR #5605 follow-ups, intentionally not fixed in this PR):
#####   * OSSG / TripolarGrid / RotatedLatitudeLongitudeGrid have no
#####     `constructor_arguments` method → grid serialization throws at write time.
#####   * ImmersedBoundaryGrid reconstruction explicitly throws at read time
#####     (`Immersed-boundary reconstruction not yet implemented for Zarr.`).
##### These cases are marked `@test_broken` so CI stays green while the gap is
##### visible.
#####

function zarr_round_trip(grid; tag::String)
    return mktempdir() do tmp
        filename = "grid_sweep_$(tag)"
        path = abspath(joinpath(tmp, filename * ".zarr"))
        model = HydrostaticFreeSurfaceModel(grid; tracers = (:T,))
        sim = Simulation(model, Δt = 1.0, stop_iteration = 1)
        sim.output_writers[:zarr] =
            ZarrWriter(model, (; T = model.tracers.T);
                       filename,
                       dir = tmp,
                       schedule = IterationInterval(1),
                       overwrite_existing = true,
                       with_halos = false)
        try
            run!(sim)
            field_time_series = FieldTimeSeries(path, "T")
            size_ok = size(field_time_series)[1:3] == size(grid)
            time_ok = length(field_time_series.times) == 2
            grid_ok = field_time_series.grid == grid
            return size_ok && time_ok && grid_ok
        catch err
            @info "  zarr_round_trip[$tag] caught $(typeof(err)): $(sprint(showerror, err))"
            return false
        end
    end
end

@testset "ZarrWriter [grid-type sweep]" begin
    @info "  Testing ZarrWriter round-trip across grid types..."

    for arch in archs
        # (tag, is_known_to_work, factory)
        grid_factories = [
            ("latitude_longitude_regular", true,
             arch -> LatitudeLongitudeGrid(arch;
                                           size = (4, 4, 2),
                                           longitude = (0, 360),
                                           latitude  = (-60, 60),
                                           z = (-100, 0),
                                           topology = (Periodic, Bounded, Bounded))),

            ("latitude_longitude_stretched", true,
             arch -> LatitudeLongitudeGrid(arch;
                                           size = (4, 4, 2),
                                           longitude = (0, 360),
                                           latitude  = collect(range(-60, 60, length = 5)),
                                           z = [-100.0, -50.0, 0.0],
                                           topology = (Periodic, Bounded, Bounded))),

            ("tripolar", false,
             arch -> TripolarGrid(arch;
                                  size = (4, 5, 1),
                                  z = (-100, 0),
                                  first_pole_longitude = 75,
                                  north_poles_latitude = 35,
                                  southernmost_latitude = -35)),

            ("rotated_latitude_longitude", false,
             arch -> RotatedLatitudeLongitudeGrid(arch;
                                                  size = (4, 4, 1),
                                                  latitude  = (-60, 60),
                                                  longitude = (-60, 60),
                                                  z = (-100, 0),
                                                  north_pole = (0, 0),
                                                  topology = (Bounded, Bounded, Bounded))),

            ("immersed_boundary_rectilinear", false,
             arch -> ImmersedBoundaryGrid(
                 RectilinearGrid(arch;
                                 size = (4, 4, 4),
                                 extent = (1, 1, 1),
                                 topology = (Periodic, Periodic, Bounded)),
                 GridFittedBottom(on_architecture(arch, fill(-0.5, 4, 4))))),

            ("immersed_boundary_latitude_longitude", false,
             arch -> ImmersedBoundaryGrid(
                 LatitudeLongitudeGrid(arch;
                                       size = (4, 4, 2),
                                       longitude = (0, 360),
                                       latitude  = (-60, 60),
                                       z = (-100, 0),
                                       topology = (Periodic, Bounded, Bounded)),
                 GridFittedBottom(on_architecture(arch, fill(-50.0, 4, 4))))),

            ("immersed_boundary_tripolar", false,
             arch -> ImmersedBoundaryGrid(
                 TripolarGrid(arch;
                              size = (4, 5, 1),
                              z = (-100, 0),
                              first_pole_longitude = 75,
                              north_poles_latitude = 35,
                              southernmost_latitude = -35),
                 GridFittedBottom(on_architecture(arch, fill(-50.0, 4, 5))))),
        ]

        for (tag, is_known_to_work, factory) in grid_factories
            @testset "$tag" begin
                grid = factory(arch)
                if is_known_to_work
                    @test zarr_round_trip(grid; tag)
                else
                    @test_broken zarr_round_trip(grid; tag)
                end
            end
        end
    end
end
