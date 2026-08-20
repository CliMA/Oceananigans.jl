include("dependencies_for_runtests.jl")

using Zarr
using Dates: value

#####
##### ZarrWriter construction and display
#####

@testset "ZarrWriter [construction]" begin
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

    materialize = Oceananigans.OutputWriters.materialize_serialized_output
    @test materialize("Oceananigans.Grids.Periodic") === Periodic
    @test materialize("(Colon(), 1:4, 3)") == (:, 1:4, 3)
    @test materialize("-Inf") === -Inf
    @test materialize("+Inf") === Inf
    @test_throws ArgumentError materialize("run(\`touch unsafe_output_metadata\`)")
end

#####
##### Time-axis writing and raw round-trip
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
                                                       with_halos = false,
                                                       global_attributes = Dict("title" => "CF interoperability test"),
                                                       output_attributes = Dict("c" => Dict("standard_name" => "test_tracer")))
        run!(simulation)

        @test isdir(zarrpath)

        # Read back via raw Zarr.zopen
        g = Zarr.zopen(zarrpath)

        # time array
        @test "time" in keys(g.arrays)
        times = g["time"][:]
        @test length(times) == 3                       # initial + 2 iterations
        @test times ≈ [0.0, 1.0, 2.0]
        @test g.attrs["Conventions"] == "CF-1.13"
        @test g.attrs["title"] == "CF interoperability test"
        @test isfile(joinpath(zarrpath, ".zmetadata"))

        # Scientific coordinates are root-level coordinate arrays, where CF-aware
        # analysis tools can discover them. Grid reconstruction metadata remains private.
        for coordinate_name in ("x_caa", "y_aca", "z_aac")
            @test coordinate_name in keys(g.arrays)
            coordinate = g[coordinate_name]
            @test coordinate.attrs["_ARRAY_DIMENSIONS"] == [coordinate_name]
        end

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
        @test g["c"].attrs["standard_name"] == "test_tracer"

        rm(zarrpath; recursive=true, force=true)
    end
end

#####
##### DateTime coordinate
#####

@testset "ZarrWriter [DateTime coordinate]" begin
    for arch in archs
        grid = RectilinearGrid(arch; size=(1, 1, 1), extent=(1, 1, 1))
        clock = Clock(time=DateTime(2021, 1, 1))
        model = NonhydrostaticModel(grid; clock, timestepper=:QuasiAdamsBashforth2, tracers=:c)

        filepath = abspath("test_zarr_datetime.zarr")
        isdir(filepath) && rm(filepath; recursive=true, force=true)

        stop_time = DateTime(2021, 1, 1, 0, 0, 1)
        simulation = Simulation(model; Δt=1second, stop_time)
        simulation.output_writers[:zarr] =
            ZarrWriter(model, (; c=model.tracers.c);
                       filename=filepath,
                       schedule=IterationInterval(1),
                       overwrite_existing=true,
                       include_grid_metrics=false)
        run!(simulation)

        group = Zarr.zopen(filepath)
        epoch = DateTime(2000, 1, 1)
        expected_initial_time = value(DateTime(2021, 1, 1) - epoch) / 1000
        @test group["time"].attrs["units"] == "seconds since 2000-01-01 00:00:00"
        @test group["time"][1] == expected_initial_time

        rm(filepath; recursive=true, force=true)
    end
end

#####
##### Operations, reductions, functions, and WindowedTimeAverage
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

        # Reduction (Average over (1, 2)) omits both reduced axes.
        @test "c_avg" in keys(g.arrays)
        c_avg_arr = g["c_avg"]
        @test size(c_avg_arr) == (4, 3)
        @test c_avg_arr.attrs["_ARRAY_DIMENSIONS"] == ["time", "z_aac"]
        @test !any(isempty, c_avg_arr.attrs["_ARRAY_DIMENSIONS"])
        # c was set to 4.0 everywhere, so the column-average is 4.0
        @test all(c_avg_arr[:, 1] .≈ Float32(4.0))

        c_avg_fts = FieldTimeSeries(zarrpath, "c_avg")
        @test size(c_avg_fts[1]) == (1, 1, 4)
        @test all(interior(c_avg_fts[1]) .≈ 4)

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
##### Grid reconstruction and multiple grids
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
##### FieldTimeSeries Zarr reader
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
##### File splitting, append, and checkpoint/restart
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
##### Alternative stores: DictStore and ZipStore
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
##### Grid-type sweep
#####
##### Single-rank round-trip per grid type. Asserts both data write and grid
##### serialization round-trip via `FieldTimeSeries(path, name)`.
#####
matching_grid_structure(original, reconstructed) =
    matching_grid_structure_base(original, reconstructed)

function matching_grid_structure(original::OrthogonalSphericalShellGrid,
                                 reconstructed::OrthogonalSphericalShellGrid)
    return matching_grid_structure_base(original, reconstructed) &&
           typeof(original.conformal_mapping).name.wrapper ===
           typeof(reconstructed.conformal_mapping).name.wrapper
end

matching_grid_structure_base(original, reconstructed) =
    typeof(original).name.wrapper === typeof(reconstructed).name.wrapper &&
    typeof(architecture(original)) === typeof(architecture(reconstructed)) &&
    size(original) == size(reconstructed) &&
    halo_size(original) == halo_size(reconstructed) &&
    topology(original) == topology(reconstructed) &&
    eltype(original) == eltype(reconstructed)

function matching_grid_structure(original::ImmersedBoundaryGrid,
                                 reconstructed::ImmersedBoundaryGrid)
    return matching_grid_structure(original.underlying_grid, reconstructed.underlying_grid) &&
           typeof(original.immersed_boundary).name.wrapper ===
           typeof(reconstructed.immersed_boundary).name.wrapper
end

function zarr_round_trip(grid; tag::String, with_halos = false)
    return mktempdir() do tmp
        filename = "grid_sweep_$(tag)"
        path = abspath(joinpath(tmp, filename * ".zarr"))
        free_surface = SplitExplicitFreeSurface(grid; substeps=5)
        model = HydrostaticFreeSurfaceModel(grid; tracers=(:T,), free_surface)
        simulation = Simulation(model, Δt=1, stop_iteration=1)
        simulation.output_writers[:zarr] =
            ZarrWriter(model, (; T=model.tracers.T);
                       filename,
                       dir=tmp,
                       schedule=IterationInterval(1),
                       overwrite_existing=true,
                       with_halos)
        run!(simulation)
        field_time_series = FieldTimeSeries(path, "T"; architecture=architecture(grid))
        size_ok = size(field_time_series)[1:3] == size(grid)
        time_ok = length(field_time_series.times) == 2
        grid_ok = matching_grid_structure(grid, field_time_series.grid)
        return size_ok && time_ok && grid_ok
    end
end

@testset "ZarrWriter [grid-type sweep]" begin
    @info "  Testing ZarrWriter round-trip across grid types..."

    for arch in archs
        grid_factories = [
            ("latitude_longitude_regular",
             arch -> LatitudeLongitudeGrid(arch;
                                           size = (4, 4, 2),
                                           longitude = (0, 360),
                                           latitude  = (-60, 60),
                                           z = (-100, 0),
                                           topology = (Periodic, Bounded, Bounded))),

            ("latitude_longitude_stretched",
             arch -> LatitudeLongitudeGrid(arch;
                                           size = (4, 4, 2),
                                           longitude = (0, 360),
                                           latitude  = collect(range(-60, 60, length = 5)),
                                           z = [-100.0, -50.0, 0.0],
                                           topology = (Periodic, Bounded, Bounded))),

            ("tripolar",
             arch -> TripolarGrid(arch;
                                  size = (4, 5, 1),
                                  z = (-100, 0),
                                  first_pole_longitude = 75,
                                  north_poles_latitude = 35,
                                  southernmost_latitude = -35)),

            ("rotated_latitude_longitude",
             arch -> RotatedLatitudeLongitudeGrid(arch;
                                                  size = (4, 4, 1),
                                                  latitude  = (-60, 60),
                                                  longitude = (-60, 60),
                                                  z = (-100, 0),
                                                  north_pole = (0, 0),
                                                  topology = (Bounded, Bounded, Bounded))),

            ("immersed_boundary_rectilinear",
             arch -> ImmersedBoundaryGrid(
                 RectilinearGrid(arch;
                                 size = (4, 4, 4),
                                 extent = (1, 1, 1),
                                 topology = (Periodic, Periodic, Bounded)),
                 GridFittedBottom(on_architecture(arch, fill(-0.5, 4, 4))))),

            ("immersed_boundary_latitude_longitude",
             arch -> ImmersedBoundaryGrid(
                 LatitudeLongitudeGrid(arch;
                                       size = (4, 4, 2),
                                       longitude = (0, 360),
                                       latitude  = (-60, 60),
                                       z = (-100, 0),
                                       topology = (Periodic, Bounded, Bounded)),
                 GridFittedBottom(on_architecture(arch, fill(-50.0, 4, 4))))),

            ("immersed_boundary_tripolar",
             arch -> ImmersedBoundaryGrid(
                 TripolarGrid(arch;
                              size = (4, 5, 1),
                              z = (-100, 0),
                              first_pole_longitude = 75,
                              north_poles_latitude = 35,
                              southernmost_latitude = -35),
                 GridFittedBottom(on_architecture(arch, fill(-50.0, 4, 5))))),
        ]

        for (tag, factory) in grid_factories
            @testset "$tag" begin
                grid = factory(arch)
                @test zarr_round_trip(grid; tag)

                if arch isa GPU && tag in ("latitude_longitude_stretched", "tripolar")
                    halo_tag = tag * "_with_halos"
                    @test zarr_round_trip(grid; tag = halo_tag, with_halos = true)
                end
            end
        end
    end
end

#####
##### TripolarGrid round-trip
#####

using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

@testset "ZarrWriter [TripolarGrid round-trip]" begin
    @info "  Testing ZarrWriter with TripolarGrid..."

    for arch in archs
        grid = TripolarGrid(arch; size=(20, 16, 4), z=(-100, 0))
        fs   = SplitExplicitFreeSurface(grid; substeps=5)
        model = HydrostaticFreeSurfaceModel(grid; free_surface=fs, tracers=(:T,))

        zarrpath = abspath(joinpath(".", "test_zarr_tripolar.zarr"))
        isdir(zarrpath) && rm(zarrpath; recursive=true, force=true)

        simulation = Simulation(model; Δt=1, stop_iteration=2)
        simulation.output_writers[:fields] = ZarrWriter(model,
                                                        (; T=model.tracers.T, u=model.velocities.u);
                                                        filename = "test_zarr_tripolar",
                                                        dir = ".",
                                                        schedule = IterationInterval(1),
                                                        overwrite_existing = true,
                                                        with_halos = false)
        run!(simulation)

        @test isdir(zarrpath)
        g = Zarr.zopen(zarrpath)

        # time axis
        @test "time" in keys(g.arrays)
        @test length(g["time"][:]) == 3   # initial + 2 iterations

        # both fields were written
        @test "T" in keys(g.arrays)
        @test "u" in keys(g.arrays)

        # spatial shape matches grid interior (no halos)
        Nx, Ny, Nz = size(grid)
        T_arr = g["T"]
        u_arr = g["u"]
        @test size(T_arr)[1:3] == (Nx, Ny, Nz)
        @test size(u_arr)[1:3] == (Nx, Ny, Nz)

        # _ARRAY_DIMENSIONS is set and includes "time"
        T_dims = T_arr.attrs["_ARRAY_DIMENSIONS"]
        @test "time" in T_dims
        @test length(T_dims) == 4

        # Data dimensions use logical i/j axes. Physical longitude and latitude are
        # two-dimensional CF auxiliary coordinates referenced by `coordinates`.
        u_dims = u_arr.attrs["_ARRAY_DIMENSIONS"]
        @test any(startswith(d, "i_") for d in u_dims)
        @test any(startswith(d, "j_") for d in u_dims)
        @test any(startswith(d, "z") for d in u_dims)

        coordinates = split(u_arr.attrs["coordinates"])
        @test any(startswith(coordinate, "λ_") for coordinate in coordinates)
        @test any(startswith(coordinate, "φ_") for coordinate in coordinates)
        for coordinate in coordinates
            @test coordinate in keys(g.arrays)
        end

        rm(zarrpath; recursive=true, force=true)
    end
end
