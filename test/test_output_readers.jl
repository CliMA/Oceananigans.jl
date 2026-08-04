include("dependencies_for_runtests.jl")

using Oceananigans.Units: Time
using Oceananigans.Fields: indices, interpolate!, ZeroField, ConstantField
using Oceananigans.OutputReaders: Cyclical, Clamp, Linear, SplitFilePath,
                                  extract_field_time_series, update_field_time_series!,
                                  GPUAdaptedFieldTimeSeriesOperation, pointwise_evaluable,
                                  AbstractFieldTimeSeries, time_indices
using Oceananigans.AbstractOperations: Average, Integral, CumulativeIntegral, @at
using Statistics: mean
using Oceananigans.AbstractOperations: @unary, @binary
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: getbc
using Oceananigans.Forcings: materialize_forcing, evaluate_target
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_velocity_fields
using Oceananigans.OutputReaders: TimeSeriesInterpolation
using Dates: DateTime

using Random
using NCDatasets

# Transfer to the CPU for scalar indexing in value checks
on_cpu(x) = on_architecture(CPU(), x)

# Operators registered after Oceananigans loads, to test that @unary / @binary
# give them FieldTimeSeries methods (must be top-level)
plus_two(x) = x + 2
@unary plus_two
harmonic(x, y) = 2 * x * y / (x + y)
@binary harmonic

function generate_nonzero_simulation_data(Lx, Δt, FT; architecture=CPU())
    grid = RectilinearGrid(architecture, size=10, x=(0, Lx), topology=(Periodic, Flat, Flat))
    model = NonhydrostaticModel(grid; tracers = (:T, :S), advection = nothing)
    set!(model, T=30, S=35)
    simulation = Simulation(model; Δt, stop_iteration=100)

    simulation.output_writers[:constant_fields] = JLD2Writer(model, model.tracers,
                                                             filename = "constant_fields",
                                                             schedule = IterationInterval(10),
                                                             array_type = Array{FT},
                                                             overwrite_existing = true)

    run!(simulation)

    return simulation.output_writers[:constant_fields].filepath
end

function generate_some_interesting_simulation_data(Nx, Ny, Nz; architecture=CPU(), output_writer=JLD2Writer)
    grid = RectilinearGrid(architecture, size=(Nx, Ny, Nz), extent=(64, 64, 32))

    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(5e-5), bottom = GradientBoundaryCondition(0.01))
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(-3e-4))

    @inline Qˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S
    evaporation_bc = FluxBoundaryCondition(Qˢ, field_dependencies=:S, parameters=3e-7)
    S_bcs = FieldBoundaryConditions(top=evaporation_bc)

    model = NonhydrostaticModel(grid; tracers = (:T, :S),
                                buoyancy = SeawaterBuoyancy(),
                                boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))

    dTdz = 0.01
    Tᵢ(x, y, z) = 20 + dTdz * z + 1e-6 * randn()
    uᵢ(x, y, z) = 1e-3 * randn()
    set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)

    simulation = Simulation(model, Δt=10.0, stop_time=2minutes, verbose=false)
    wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1minute)
    simulation.callbacks[:wizard] = Callback(wizard)

    u, v, w = model.velocities

    computed_fields = (
        b = buoyancy_field(model),
        ζ = Field(∂x(v) - ∂y(u)),
        ke = Field(√(u^2 + v^2))
    )

    fields_to_output = merge(model.velocities, model.tracers, computed_fields)

    # Determine file extension based on output writer type
    file_ext = output_writer == JLD2Writer ? ".jld2" : ".nc"

    filepath3d = "test_3d_output_with_halos" * file_ext
    simulation.output_writers[:writer_3d_with_halos] = output_writer(model, fields_to_output,
                                                                     filename = filepath3d,
                                                                     with_halos = true,
                                                                     schedule = TimeInterval(30seconds),
                                                                     overwrite_existing = true)

    filepath2d = "test_2d_output_with_halos" * file_ext
    if output_writer == JLD2Writer
        simulation.output_writers[:writer_2d_with_halos] = output_writer(model, fields_to_output,
                                                                         filename = filepath2d,
                                                                         indices = (:, :, grid.Nz),
                                                                         with_halos = true,
                                                                         schedule = TimeInterval(30seconds),
                                                                         overwrite_existing = true)
    else
        @warn "Skipping 2D output writer since you cannot pass non-default indices to NetCDFWriter if `with_halos=true`."
    end

    profiles = NamedTuple{keys(fields_to_output)}(Field(Average(f, dims=(1, 2))) for f in fields_to_output)

    filepath1d = "test_1d_output_with_halos" * file_ext
    simulation.output_writers[:writer_1d_with_halos] = output_writer(model, profiles,
                                                                     filename = filepath1d,
                                                                     with_halos = true,
                                                                     schedule = TimeInterval(30seconds),
                                                                     overwrite_existing = true)

    unsplit_filepath = "test_unsplit_output" * file_ext
    simulation.output_writers[:unsplit_writer] = output_writer(model, profiles,
                                                               filename = unsplit_filepath,
                                                               with_halos = true,
                                                               schedule = TimeInterval(10seconds),
                                                               overwrite_existing = true)

    split_filepath = "test_split_output" * file_ext
    simulation.output_writers[:split_writer] = output_writer(model, profiles,
                                                             filename = split_filepath,
                                                             with_halos = true,
                                                             schedule = TimeInterval(10seconds),
                                                             file_splitting = TimeInterval(30seconds),
                                                             overwrite_existing = true)

    run!(simulation)

    return filepath1d, filepath2d, filepath3d, unsplit_filepath, split_filepath
end

function test_pickup_with_inaccurate_times()
    # Testing pickup using example that was failing in https://github.com/CliMA/Oceananigans.jl/issues/4077
    grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
    times = collect(0:0.1:3)
    filename = "fts_inaccurate_times_test.jld2"
    f_tmp = Field{Center,Center,Center}(grid)
    f = FieldTimeSeries{Center, Center, Center}(grid, times; backend=OnDisk(), path=filename, name="f")

    for (it, time) in enumerate(f.times)
        set!(f_tmp,   30)
        set!(f,f_tmp, it)
    end

    # Create another time array that is slightly different at t=0
    times_mod = copy(times)
    times_mod[1] = 1e-16

    # Now we load the FTS partly in memory
    f_fts = FieldTimeSeries(filename, "f"; backend = InMemory(5), times = times_mod)
    Nt = length(f_fts.times)

    for t in eachindex(times)
        @test all(interior(f_fts[t]) .== 30)
    end

    rm(filename, force=true)

    return nothing
end

function test_field_time_series_in_memory_3d(arch, filepath3d, Nx, Ny, Nz, Nt)
    # 3D Fields
    u3 = FieldTimeSeries(filepath3d, "u", architecture=arch)
    v3 = FieldTimeSeries(filepath3d, "v", architecture=arch)
    w3 = FieldTimeSeries(filepath3d, "w", architecture=arch)
    T3 = FieldTimeSeries(filepath3d, "T", architecture=arch)
    b3 = FieldTimeSeries(filepath3d, "b", architecture=arch)
    ζ3 = FieldTimeSeries(filepath3d, "ζ", architecture=arch)

    # This behavior ensures that set! works
    # but perhaps should be changed in the future
    @test size(parent(u3[1])) == size(parent(u3))[1:3]
    @test size(parent(v3[1])) == size(parent(v3))[1:3]
    @test size(parent(w3[1])) == size(parent(w3))[1:3]
    @test size(parent(T3[1])) == size(parent(T3))[1:3]
    @test size(parent(b3[1])) == size(parent(b3))[1:3]
    @test size(parent(ζ3[1])) == size(parent(ζ3))[1:3]

    @test location(u3) == (Face, Center, Center)
    @test location(v3) == (Center, Face, Center)
    @test location(w3) == (Center, Center, Face)
    @test location(T3) == (Center, Center, Center)
    @test location(b3) == (Center, Center, Center)
    @test location(ζ3) == (Face, Face, Center)

    @test size(u3) == (Nx, Ny, Nz,   Nt)
    @test size(v3) == (Nx, Ny, Nz,   Nt)
    @test size(w3) == (Nx, Ny, Nz+1, Nt)
    @test size(T3) == (Nx, Ny, Nz,   Nt)
    @test size(b3) == (Nx, Ny, Nz,   Nt)
    @test size(ζ3) == (Nx, Ny, Nz,   Nt)

    ArrayType = array_type(arch)
    for fts in (u3, v3, w3, T3, b3, ζ3)
        @test parent(fts) isa ArrayType
        @test (fts.times isa StepRangeLen) | (fts.times isa ArrayType)
    end

    if arch isa CPU
        @test u3[1, 2, 3, 4] isa Number
        @test u3[1] isa Field
        @test v3[2] isa Field
    end

    # Tests construction + that we can interpolate
    u3i = FieldTimeSeries{Face, Center, Center}(u3.grid, u3.times)
    @test !isnothing(u3i.boundary_conditions)
    @test u3i.boundary_conditions isa FieldBoundaryConditions

    interpolate!(u3i, u3)
    @test all(interior(u3i) .≈ interior(u3))

    # Interpolation to a _located_ single column grid
    grid3 = RectilinearGrid(arch, size=(3, 3, 3), x=(0.5, 3.5), y=(0.5, 3.5), z=(0.5, 3.5),
                            topology = (Periodic, Periodic, Bounded))

    grid1 = RectilinearGrid(arch, size=3, x=1.3, y=2.7, z=(0.5, 3.5),
                            topology=(Flat, Flat, Bounded))

    times = [1, 2]
    c3 = FieldTimeSeries{Center, Center, Center}(grid3, times)
    c1 = FieldTimeSeries{Center, Center, Center}(grid1, times)

    for n in 1:length(times)
        tn = times[n]
        c₀(x, y, z) = (x + y + z) * tn
        set!(c3[n], c₀)
    end

    interpolate!(c1, c3)

    # Convert to CPU for testing
    c11 = interior(c1[1], 1, 1, :) |> Array
    c12 = interior(c1[2], 1, 1, :) |> Array

    @test c11 ≈ [5.0, 6.0, 7.0]
    @test c12 ≈ [10.0, 12.0, 14.0]

    return nothing
end

function test_field_time_series_in_memory_2d(arch, filepath2d, Nx, Ny, Nt)
    ## 2D sliced Fields
    u2 = FieldTimeSeries(filepath2d, "u", architecture=arch)
    v2 = FieldTimeSeries(filepath2d, "v", architecture=arch)
    w2 = FieldTimeSeries(filepath2d, "w", architecture=arch)
    T2 = FieldTimeSeries(filepath2d, "T", architecture=arch)
    b2 = FieldTimeSeries(filepath2d, "b", architecture=arch)
    ζ2 = FieldTimeSeries(filepath2d, "ζ", architecture=arch)

    @test location(u2) == (Face, Center, Center)
    @test location(v2) == (Center, Face, Center)
    @test location(w2) == (Center, Center, Face)
    @test location(T2) == (Center, Center, Center)
    @test location(b2) == (Center, Center, Center)
    @test location(ζ2) == (Face, Face, Center)

    @test size(u2) == (Nx, Ny, 1, Nt)
    @test size(v2) == (Nx, Ny, 1, Nt)
    @test size(w2) == (Nx, Ny, 1, Nt)
    @test size(T2) == (Nx, Ny, 1, Nt)
    @test size(b2) == (Nx, Ny, 1, Nt)
    @test size(ζ2) == (Nx, Ny, 1, Nt)

    ArrayType = array_type(arch)
    for fts in (u2, v2, w2, T2, b2, ζ2)
        @test parent(fts) isa ArrayType
    end

    if arch isa CPU
        @test u2[1, 2, 5, 4] isa Number
        @test u2[1] isa Field
        @test v2[2] isa Field
    end

    return nothing
end

function test_field_time_series_in_memory_1d(arch, filepath1d, Nz, Nt)
    ## 1D AveragedFields
    u1 = FieldTimeSeries(filepath1d, "u", architecture=arch)
    v1 = FieldTimeSeries(filepath1d, "v", architecture=arch)
    w1 = FieldTimeSeries(filepath1d, "w", architecture=arch)
    T1 = FieldTimeSeries(filepath1d, "T", architecture=arch)
    b1 = FieldTimeSeries(filepath1d, "b", architecture=arch)
    ζ1 = FieldTimeSeries(filepath1d, "ζ", architecture=arch)

    @test location(u1) == (Nothing, Nothing, Center)
    @test location(v1) == (Nothing, Nothing, Center)
    @test location(w1) == (Nothing, Nothing, Face)
    @test location(T1) == (Nothing, Nothing, Center)
    @test location(b1) == (Nothing, Nothing, Center)
    @test location(ζ1) == (Nothing, Nothing, Center)

    @test size(u1) == (1, 1, Nz,   Nt)
    @test size(v1) == (1, 1, Nz,   Nt)
    @test size(w1) == (1, 1, Nz+1, Nt)
    @test size(T1) == (1, 1, Nz,   Nt)
    @test size(b1) == (1, 1, Nz,   Nt)
    @test size(ζ1) == (1, 1, Nz,   Nt)

    ArrayType = array_type(arch)
    for fts in (u1, v1, w1, T1, b1, ζ1)
        @test parent(fts) isa ArrayType
    end

    if arch isa CPU
        @test u1[1, 1, 3, 4] isa Number
        @test u1[1] isa Field
        @test v1[2] isa Field
    end

    return nothing
end

function test_field_time_series_in_memory_split(arch, split_filepath, unsplit_filepath)
    us = FieldTimeSeries(split_filepath, "u", architecture=arch)
    vs = FieldTimeSeries(split_filepath, "v", architecture=arch)
    ws = FieldTimeSeries(split_filepath, "w", architecture=arch)
    Ts = FieldTimeSeries(split_filepath, "T", architecture=arch)
    bs = FieldTimeSeries(split_filepath, "b", architecture=arch)
    ζs = FieldTimeSeries(split_filepath, "ζ", architecture=arch)

    uu = FieldTimeSeries(unsplit_filepath, "u", architecture=arch)
    vu = FieldTimeSeries(unsplit_filepath, "v", architecture=arch)
    wu = FieldTimeSeries(unsplit_filepath, "w", architecture=arch)
    Tu = FieldTimeSeries(unsplit_filepath, "T", architecture=arch)
    bu = FieldTimeSeries(unsplit_filepath, "b", architecture=arch)
    ζu = FieldTimeSeries(unsplit_filepath, "ζ", architecture=arch)

    split = (us, vs, ws, Ts, bs, ζs)
    unsplit = (uu, vu, wu, Tu, bu, ζu)
    for pair in zip(split, unsplit)
        s, u = pair
        @test s.times == u.times
        @test parent(s) == parent(u)
    end

    return nothing
end

function test_field_time_series_split_files(arch)
    dir = mktempdir()
    grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid, tracers=:c)
    simulation = Simulation(model, Δt=1, stop_time=10)

    set_tracer_to_iteration!(sim) = fill!(parent(sim.model.tracers.c), sim.model.clock.iteration)
    add_callback!(simulation, set_tracer_to_iteration!, IterationInterval(1))

    simulation.output_writers[:fields] = JLD2Writer(model, model.tracers;
                                                     filename = "split_test",
                                                     dir = dir,
                                                     schedule = IterationInterval(1),
                                                     file_splitting = TimeInterval(3),
                                                     overwrite_existing = true)
    run!(simulation)

    # Use absolute path (tests glob fix)
    abs_path = joinpath(dir, "split_test.jld2")

    # Test InMemory backend with split files
    fts_mem = FieldTimeSeries(abs_path, "c", architecture=arch)
    @test length(fts_mem.times) == 11
    @test fts_mem[1] isa Field
    @test fts_mem[11] isa Field
    @test fts_mem.path isa SplitFilePath

    # Test OnDisk backend with split files
    fts_disk = FieldTimeSeries(abs_path, "c"; backend=OnDisk(), architecture=arch)
    @test length(fts_disk.times) == 11

    # Access from each part file
    for n in 1:length(fts_disk.times)
        @test fts_disk[n] isa Field
    end

    fts_partly = FieldTimeSeries(abs_path, "c"; backend=InMemory(2), architecture=arch)
    @test fts_partly.path isa SplitFilePath
    for n in 1:length(fts_partly.times)
        @test Array(interior(fts_partly[n])) == Array(interior(fts_mem[n]))
    end

    rm(dir, recursive=true, force=true)
    return nothing
end

function test_field_time_series_pickup(arch)
    Random.seed!(1234)
    for n in -4:4
        Δt = (1.1 + rand()) * 10.0^n
        Lx = 10 * Δt
        for FT in (Float32, Float64)
            filename = generate_nonzero_simulation_data(Lx, Δt, FT)
            Tfts = FieldTimeSeries(filename, "T")
            Sfts = FieldTimeSeries(filename, "S")

            for t in eachindex(Tfts.times)
                @test all(interior(Tfts[t]) .== 30)
                @test all(interior(Sfts[t]) .== 35)
            end
        end
    end

    @info "  Testing FieldTimeSeries pickup with slightly inaccurate times..."
    test_pickup_with_inaccurate_times()

    return nothing
end

function test_field_time_series_array_boundary_conditions(arch)
    x = y = z = (0, 1)
    grid = RectilinearGrid(arch; size=(1, 1, 1), x, y, z)

    τx = on_architecture(arch, zeros(size(grid)...))
    τy = Field{Center, Face, Nothing}(grid)
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
    v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τy))
    model = NonhydrostaticModel(grid; boundary_conditions = (; u=u_bcs, v=v_bcs))
    simulation = Simulation(model; Δt=1, stop_iteration=1)

    filename = arch isa GPU ? "test_cuarray_bc.jld2" : "test_array_bc.jld2"

    simulation.output_writers[:jld2] = JLD2Writer(model, model.velocities;
                                                  filename,
                                                  schedule=IterationInterval(1),
                                                  overwrite_existing = true)
    run!(simulation)

    ut = FieldTimeSeries(filename, "u")
    vt = FieldTimeSeries(filename, "v")
    @test ut.boundary_conditions.top.classification isa Flux
    @test ut.boundary_conditions.top.condition isa Array

    τy_ow = vt.boundary_conditions.top.condition
    @test τy_ow isa Field{Center, Face, Nothing}
    @test architecture(τy_ow) isa CPU
    @test parent(τy_ow) isa Array
    rm(filename)

    return nothing
end

function test_field_time_series_function_boundary_conditions(arch)
    grid = RectilinearGrid(arch; topology=(Bounded, Periodic, Bounded), size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1))

    u_west(x, y, t) = 0
    u_east(x, y, t) = 0
    u_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(u_west), east = NormalFlowBoundaryCondition(u_east, scheme=PerturbationAdvection()))
    model = NonhydrostaticModel(grid; boundary_conditions = (; u=u_bcs))
    simulation = Simulation(model; Δt=1, stop_iteration=1)

    filename = "test_function_bc.jld2"
    simulation.output_writers[:jld2] = JLD2Writer(model, model.velocities;
                                                  filename,
                                                  schedule=IterationInterval(1),
                                                  overwrite_existing = true)
    run!(simulation)

    @test FieldTimeSeries(filename, "u") isa FieldTimeSeries
    rm(filename)

    return nothing
end

function test_field_time_series_on_disk(arch, filepath3d, filepath1d, Nx, Ny, Nz, Nt)
    ArrayType = array_type(arch)

    ζ = FieldTimeSeries(filepath3d, "ζ", backend=OnDisk(), architecture=arch)
    @test location(ζ) == (Face, Face, Center)
    @test size(ζ) == (Nx, Ny, Nz, Nt)
    @test ζ[1] isa Field
    @test ζ[2] isa Field
    @test ζ[1].data.parent isa ArrayType

    b = FieldTimeSeries(filepath1d, "b", backend=OnDisk(), architecture=arch)
    @test location(b) == (Nothing, Nothing, Center)
    @test size(b) == (1, 1, Nz, Nt)
    @test b[1] isa Field
    @test b[2] isa Field

    return nothing
end

function test_field_time_series_reductions(filepath3d, Nt)
    for name in ("u", "v", "w", "T", "b", "ζ"), fun in (sum, mean, maximum, minimum)
        f = FieldTimeSeries(filepath3d, name, architecture=CPU())

        ε = eps(maximum(abs, f.data.parent))

        val1 = fun(f)
        val2 = fun([fun(f[n]) for n in 1:Nt])

        @test val1 ≈ val2 atol=4ε
    end

    return nothing
end

function test_field_time_series_reductions_with_dims()
    grid = RectilinearGrid(size=(4, 5, 6), extent=(1, 1, 1))
    times = [1.0, 2.0, 3.0]
    fts = FieldTimeSeries{Center, Center, Center}(grid, times)

    # Fill with known data: value = i + j + k + 10*n
    for n in 1:length(times)
        set!(fts[n], (x, y, z) -> x + y + z + 10 * n)
    end

    # Test dims=1 (reduce over x)
    fts1 = sum(fts; dims=1)
    @test fts1 isa FieldTimeSeries
    @test location(fts1) == (Nothing, Center, Center)
    @test size(fts1) == (1, 5, 6, 3)
    @test length(fts1.times) == 3

    # Test dims=2 (reduce over y)
    fts2 = sum(fts; dims=2)
    @test fts2 isa FieldTimeSeries
    @test location(fts2) == (Center, Nothing, Center)
    @test size(fts2) == (4, 1, 6, 3)
    @test length(fts2.times) == 3

    # Test dims=3 (reduce over z)
    fts3 = sum(fts; dims=3)
    @test fts3 isa FieldTimeSeries
    @test location(fts3) == (Center, Center, Nothing)
    @test size(fts3) == (4, 5, 1, 3)
    @test length(fts3.times) == 3

    # Test dims=4 (reduce over time)
    fts4 = sum(fts; dims=4)
    @test fts4 isa FieldTimeSeries
    @test location(fts4) == (Center, Center, Center)
    @test size(fts4) == (4, 5, 6, 1)
    @test length(fts4.times) == 1
    @test fts4.times[1] ≈ mean(times)

    # Test correctness of dims=4 sum
    expected_sum = zeros(4, 5, 6)
    for n in 1:length(times)
        expected_sum .+= interior(fts[n])
    end
    @test interior(fts4[1]) ≈ expected_sum

    # Test mean with dims=4
    fts4_mean = mean(fts; dims=4)
    @test fts4_mean isa FieldTimeSeries
    @test size(fts4_mean) == (4, 5, 6, 1)
    @test interior(fts4_mean[1]) ≈ expected_sum ./ length(times)

    # Test combined dims=(1, 4) (reduce over x and time)
    fts14 = sum(fts; dims=(1, 4))
    @test fts14 isa FieldTimeSeries
    @test location(fts14) == (Nothing, Center, Center)
    @test size(fts14) == (1, 5, 6, 1)
    @test length(fts14.times) == 1

    # Test combined dims=(1, 2, 4) (reduce over x, y, and time)
    fts124 = sum(fts; dims=(1, 2, 4))
    @test fts124 isa FieldTimeSeries
    @test location(fts124) == (Nothing, Nothing, Center)
    @test size(fts124) == (1, 1, 6, 1)

    # Test dims=: (reduce over all dimensions, returns a scalar)
    total_sum = sum(fts; dims=:)
    expected_total = sum(sum(fts[n]) for n in 1:length(times))
    @test total_sum ≈ expected_total

    total_mean = mean(fts; dims=:)
    expected_mean = expected_total / (4 * 5 * 6 * 3)
    @test total_mean ≈ expected_mean

    return nothing
end

function test_chunked_abstraction(filepath, name)
    fts = FieldTimeSeries(filepath, name)
    fts_chunked = FieldTimeSeries(filepath, name; backend=InMemory(2), time_indexing=Cyclical())

    for t in eachindex(fts.times)
        fts_chunked[t] == fts[t]
    end

    min_fts, max_fts = extrema(fts)

    # Test cyclic time interpolation with update_field_time_series!
    times = map(Time, 0:0.1:300)
    for time in times
        @test minimum(fts_chunked[time]) ≥ min_fts
        @test maximum(fts_chunked[time]) ≤ max_fts
    end

    return nothing
end

function test_time_interpolation()
    times = rand(100) * 100
    times = sort(times) # random times between 0 and 100

    min_t, max_t = extrema(times)

    grid = RectilinearGrid(size = (1, 1, 1), extent = (1, 1, 1))

    fts_cyclic = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times; time_indexing = Cyclical())
    fts_clamp  = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times; time_indexing = Clamp())

    for t in eachindex(times)
        fill!(fts_cyclic[t], t / 2) # value of the field between 0.5 and 50
        fill!(fts_clamp[t], t / 2)  # value of the field between 0.5 and 50
    end

    # Let's test that the field remains bounded between 0.5 and 50
    for time in Time.(collect(0:0.1:100))
        @test fts_cyclic[1, 1, 1, time] ≤ 50
        @test fts_cyclic[1, 1, 1, time] ≥ 0.5

        if time.time > max_t
            @test fts_clamp[1, 1, 1, time] == 50
        elseif time.time < min_t
            @test fts_clamp[1, 1, 1, time] == 0.5
        else
            @test fts_clamp[1, 1, 1, time] ≈ fts_cyclic[1, 1, 1, time]
        end
    end

    return nothing
end

function test_field_dataset_indexing(Backend, filepath3d)
    ds = FieldDataset(filepath3d, backend=Backend())

    @test ds isa FieldDataset
    @test length(keys(ds)) == 8

    for var_str in ("u", "v", "w", "T", "S", "b", "ζ", "ke")
        @test var_str in keys(ds)
        @test ds[var_str] isa FieldTimeSeries
        @test ds[var_str][1] isa Field
    end

    for var_sym in (:u, :v, :w, :T, :S, :b, :ζ, :ke)
        @test ds[var_sym] isa FieldTimeSeries
        @test ds[var_sym][2] isa Field
    end

    @test ds.u isa FieldTimeSeries
    @test ds.v isa FieldTimeSeries
    @test ds.w isa FieldTimeSeries
    @test ds.T isa FieldTimeSeries
    @test ds.S isa FieldTimeSeries
    @test ds.b isa FieldTimeSeries
    @test ds.ζ isa FieldTimeSeries
    @test ds.ke isa FieldTimeSeries

    return nothing
end

function test_field_time_series_parallel_reading(Backend, filepath3d)
    reader_kw = Dict(:parallel_read => true)
    u3 = FieldTimeSeries(filepath3d, "u"; backend=Backend(), reader_kw)
    b3 = FieldTimeSeries(filepath3d, "b"; backend=Backend(), reader_kw)

    @test u3 isa FieldTimeSeries
    @test b3 isa FieldTimeSeries
    @test u3[1] isa Field
    @test b3[1] isa Field

    return nothing
end

function test_field_dataset_parallel_reading(Backend, filepath3d)
    reader_kw = (; parallel_read = true)
    ds = FieldDataset(filepath3d; backend=Backend(), reader_kw)

    @test ds isa FieldDataset
    @test ds.u isa FieldTimeSeries
    @test ds.b isa FieldTimeSeries
    @test ds.u[1] isa Field
    @test ds.b[1] isa Field

    return nothing
end

function test_interpolation_with_in_memory_backends(filepath_sine)
    grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    times = 0:0.1:3

    sinf(t) = sin(2π * t / 3)

    fts = FieldTimeSeries{Center, Center, Center}(grid, times; backend=OnDisk(), path=filepath_sine, name="f")

    f = CenterField(grid)
    for (i, time) in enumerate(fts.times)
        set!(f, (x, y, z) -> sinf(time))
        set!(fts, f, i)
    end

    # Now we load the FTS partly in memory
    # using different time indexing strategies
    M = 5
    fts_lin = FieldTimeSeries(filepath_sine, "f"; backend = InMemory(M), time_indexing = Linear())
    fts_cyc = FieldTimeSeries(filepath_sine, "f"; backend = InMemory(M), time_indexing = Cyclical())
    fts_clp = FieldTimeSeries(filepath_sine, "f"; backend = InMemory(M), time_indexing = Clamp())

    # Test that linear interpolation is correct within the time domain
    for time in 0:0.01:last(fts.times)
        tidx = findfirst(fts.times .> time)
        if !isnothing(tidx)
            t⁻ = fts.times[tidx - 1]
            t⁺ = fts.times[tidx]

            Δt⁺ = (time - t⁻) / (t⁺ - t⁻)

            @test fts_lin[Time(time)][1, 1, 1] ≈ (sinf(t⁻) * (1 - Δt⁺) + sinf(t⁺) * Δt⁺)
            @test fts_cyc[Time(time)][1, 1, 1] ≈ (sinf(t⁻) * (1 - Δt⁺) + sinf(t⁺) * Δt⁺)
            @test fts_clp[Time(time)][1, 1, 1] ≈ (sinf(t⁻) * (1 - Δt⁺) + sinf(t⁺) * Δt⁺)
        end
    end

    # Test that the time interpolation is correct outside the time domain
    Δt = fts.times[end] - fts.times[end-1]
    Tf = last(fts.times)
    from = Tf+1
    to = 2Tf

    for t in from:0.01:to
        dfdt = (fts_lin[end][1, 1, 1] - fts_lin[end-1][1, 1, 1]) / Δt
        extrapolated = (t - Tf) * dfdt
        @test fts_lin[Time(t)][1, 1, 1] ≈ extrapolated
        @test fts_clp[Time(t)][1, 1, 1] ≈ fts_clp[end][1, 1, 1]
    end

    return nothing
end

#####
##### Run tests
#####

@testset "OutputReaders" begin
    @info "Testing output readers..."

    Nt = 5
    Nx, Ny, Nz = 16, 10, 5

    for arch in archs
        @testset "FieldTimeSeriesOperation [$(typeof(arch))]" begin
            @info "  Testing FieldTimeSeriesOperation [$(typeof(arch))]..."
            grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 1, 1))
            times = 0:1.0:3
            a = FieldTimeSeries{Center, Center, Center}(grid, times)
            b = FieldTimeSeries{Center, Center, Center}(grid, times)
            for n in 1:length(times)
                set!(a[n], n)      # a = 1, 2, 3, 4
                set!(b[n], 2n)     # b = 2, 4, 6, 8
            end

            q = a * b
            @test q isa FieldTimeSeriesOperation
            @test size(q) == (2, 2, 2, 4)
            @test ndims(q) == 4

            # Like a FieldTimeSeries, an operation is vector-like wrt singleton indices
            @test length(q) == 4
            @test Array(interior(compute!(Field(q[end]))))[1, 1, 1] == 4 * 8

            # Node indexing: q[n] is a three-dimensional operation over slices
            qn = compute!(Field(q[2]))
            @test Array(interior(qn))[1, 1, 1] == 2 * 4
            @test on_cpu(q)[1, 1, 1, 2] == 8

            # Time indexing linearly interpolates the node values of the operation,
            # exactly like Time-indexing a stored FieldTimeSeries of the result
            @test on_cpu(q)[1, 1, 1, Time(0.5)] == 0.5 * (1 * 2) + 0.5 * (2 * 4)
            f = q[Time(0.5)]
            @test f isa Field
            @test Array(interior(f))[1, 1, 1] == 5.0
            @test Array(interior(q[Time(1.0)]))[1, 1, 1] == 8.0  # at a node: exact

            # For nonlinear operators this differs (between nodes) from operating
            # on Time-interpolated arguments
            itc = on_cpu(a)[1, 1, 1, Time(0.5)] * on_cpu(b)[1, 1, 1, Time(0.5)]
            @test itc == 1.5 * 3.0
            @test on_cpu(q)[1, 1, 1, Time(0.5)] != itc

            # Composition, unary operators, and scalar / Field mixing
            r = sqrt(a^2 + b^2)
            @test r isa FieldTimeSeriesOperation
            @test on_cpu(r)[1, 1, 1, 2] ≈ sqrt(4 + 16)

            s = 2 * a + b / 2 - 1
            @test on_cpu(s)[1, 1, 1, 3] == 2 * 3 + 6 / 2 - 1
            @test on_cpu(-a)[1, 1, 1, 1] == -1

            c = CenterField(grid)
            set!(c, 10)
            w = a * c
            @test w isa FieldTimeSeriesOperation
            @test on_cpu(w)[1, 1, 1, 3] == 30

            # Multiary chains mixing series with Fields and Numbers
            m3 = +(a, b, b)
            @test m3 isa FieldTimeSeriesOperation
            @test on_cpu(m3)[1, 1, 1, 2] == 10
            @test on_cpu(a + b + c)[1, 1, 1, 2] == 16
            @test on_cpu(a + b + 1)[1, 1, 1, 2] == 7
            @test on_cpu(a * b * 2)[1, 1, 1, 2] == 16

            # Location-prefixed forms, as @at builds them
            at_q = @at (Center, Center, Center) a * b
            @test on_cpu(at_q)[1, 1, 1, 2] == 8

            # ZeroField and ConstantField operands
            @test a + ZeroField() === a
            @test on_cpu(a * ConstantField(2.0))[1, 1, 1, 2] == 4
            @test on_cpu(ConstantField(2.0) * a)[1, 1, 1, 2] == 4

            # Function operands become FunctionField arguments
            two(x, y, z) = 2.0
            @test on_cpu(a * two)[1, 1, 1, 2] == 4

            # Materialization: Time-indexing the materialized series is identical
            # to Time-indexing the lazy operation
            qfts = FieldTimeSeries(q)
            @test qfts isa FieldTimeSeries
            for n in 1:length(times)
                @test Array(interior(qfts[n]))[1, 1, 1] == n * 2n
            end
            @test Array(interior(qfts[Time(2.7)]))[1, 1, 1] ≈ Array(interior(q[Time(2.7)]))[1, 1, 1]

            # The operation takes the place of the series' path, so file-backed
            # materialization and path/name overrides are refused
            @test_throws ArgumentError FieldTimeSeries(q; backend=OnDisk())
            @test_throws ArgumentError FieldTimeSeries(q; path="q.jld2")

            # Mismatched times and missing FieldTimeSeries arguments throw
            other = FieldTimeSeries{Center, Center, Center}(grid, 0:0.5:1.5)
            @test_throws ArgumentError a * other
            @test_throws ArgumentError FieldTimeSeriesOperation(+, c, 1)

            @test summary(q) isa String

            # An operation is extracted whole (updating it updates its arguments),
            # from nested operation trees and containers alike
            @test extract_field_time_series(q) === (q,)
            @test extract_field_time_series(sqrt(a^2 + b^2))[1] isa FieldTimeSeriesOperation
            @test extract_field_time_series((q, c, 1.0)) === (q,)

            # KernelFunctionOperation form: FieldTimeSeries arguments are sliced,
            # other arguments (e.g. parameters) pass through
            @inline scaled_product(i, j, k, grid, a, b, scale) = @inbounds scale * a[i, j, k] * b[i, j, k]
            qk = FieldTimeSeriesOperation{Center, Center, Center}(scaled_product, grid, a, b, 10)
            @test qk isa FieldTimeSeriesOperation
            @test on_cpu(qk)[1, 1, 1, 2] == 10 * 2 * 4
            @test on_cpu(qk)[1, 1, 1, Time(0.5)] == 0.5 * 10 * (1 * 2) + 0.5 * 10 * (2 * 4)
            @test extract_field_time_series(qk) === (qk,)
        end

        @testset "FieldTimeSeriesOperation with partly-in-memory arguments [$(typeof(arch))]" begin
            @info "  Testing FieldTimeSeriesOperation with partly-in-memory arguments [$(typeof(arch))]..."
            host_grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
            times = 0:1.0:3
            path_a = "ftsop_windowed_a_$(typeof(arch)).jld2"
            path_b = "ftsop_windowed_b_$(typeof(arch)).jld2"
            rm(path_a, force=true); rm(path_b, force=true)

            fta = FieldTimeSeries{Center, Center, Center}(host_grid, times; backend=OnDisk(), path=path_a, name="a")
            ftb = FieldTimeSeries{Center, Center, Center}(host_grid, times; backend=OnDisk(), path=path_b, name="b")
            snapshot = CenterField(host_grid)
            for n in 1:length(times)
                set!(snapshot, n);  set!(fta, snapshot, n)   # a = 1, 2, 3, 4
                set!(snapshot, 2n); set!(ftb, snapshot, n)   # b = 2, 4, 6, 8
            end

            aw = FieldTimeSeries(path_a, "a"; backend=InMemory(2), architecture=arch)
            bw = FieldTimeSeries(path_b, "b"; backend=InMemory(2), architecture=arch)

            q = aw * bw

            # Global Time getindex jointly updates both bracketing indices,
            # sliding the windows forward and jumping backward
            @test Array(interior(q[Time(2.5)]))[1, 1, 1] == 0.5 * 18 + 0.5 * 32
            @test Array(interior(q[Time(0.5)]))[1, 1, 1] == 0.5 * 2 + 0.5 * 8
            @test Array(interior(q[Time(1.5)]))[1, 1, 1] == 0.5 * (2 * 4) + 0.5 * (3 * 6)

            # Explicit joint update positions the windows for kernel-path access
            update_field_time_series!(q, Time(2.7))
            adapted = Adapt.adapt(Array, q)
            @test adapted[1, 1, 1, 3] == 18
            @test adapted[1, 1, 1, Time(2.7)] ≈ 0.3 * 18 + 0.7 * 32

            # size and indices are metadata queries: they must not move the windows
            windows = (time_indices(aw), time_indices(bw))
            @test size(q) == (2, 2, 2, 4)
            @test indices(q) == (:, :, :)
            @test (time_indices(aw), time_indices(bw)) === windows

            # Host-side pointwise Time indexing positions the windows itself
            if arch isa CPU
                @test q[1, 1, 1, Time(0.5)] == 5.0
                @test q[1, 1, 1, Time(2.5)] == 25.0
            end

            # Materialization from windowed arguments
            qfts = FieldTimeSeries(q)
            for n in 1:length(times)
                @test Array(interior(qfts[n]))[1, 1, 1] == n * 2n
            end

            rm(path_a, force=true); rm(path_b, force=true)
        end

        @testset "FieldTimeSeriesOperation windowed materialization [$(typeof(arch))]" begin
            @info "  Testing FieldTimeSeriesOperation windowed materialization [$(typeof(arch))]..."
            grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 1, 1))
            times = 0:1.0:3
            a = FieldTimeSeries{Center, Center, Center}(grid, times)
            b = FieldTimeSeries{Center, Center, Center}(grid, times)
            for n in 1:length(times)
                set!(a[n], n)      # a = 1, 2, 3, 4
                set!(b[n], 2n)     # b = 2, 4, 6, 8
            end
            q = a * b

            # The operation is the series' provenance: set! (re)computes from it
            qw = FieldTimeSeries(q; backend=InMemory(2))
            @test qw.path === q
            @test size(parent(qw), 4) == 2   # only the window is stored

            for n in 1:4   # slicing slides + recomputes the window
                @test Array(interior(qw[n]))[1, 1, 1] == n * 2n
                @test size(parent(qw), 4) == 2
            end
            @test Array(interior(qw[1]))[1, 1, 1] == 2    # backward jump
            @test Array(interior(qw[Time(2.5)]))[1, 1, 1] == 0.5 * 18 + 0.5 * 32
            @test Array(interior(qw[Time(0.5)]))[1, 1, 1] == 0.5 * 2 + 0.5 * 8

            # Pointwise (kernel-path) indexing after an explicit update, as models do
            update_field_time_series!(qw, Time(1.5))
            @test Adapt.adapt(Array, qw)[1, 1, 1, Time(1.5)] == 13.0

            @test occursin("FieldTimeSeriesOperation", summary(qw))

            # Structural window validation is inherited from InMemory
            @test_throws ArgumentError FieldTimeSeries(q; backend=InMemory(5))

            # Cyclical time indexing wraps across the window
            ac = FieldTimeSeries{Center, Center, Center}(grid, times; time_indexing=Cyclical(4.0))
            bc = FieldTimeSeries{Center, Center, Center}(grid, times; time_indexing=Cyclical(4.0))
            for n in 1:length(times)
                set!(ac[n], n)
                set!(bc[n], 2n)
            end
            qcw = FieldTimeSeries(ac * bc; backend=InMemory(2))
            @test Array(interior(qcw[Time(3.5)]))[1, 1, 1] == 0.5 * 32 + 0.5 * 2
            @test Array(interior(qcw[Time(4.5)]))[1, 1, 1] == 0.5 * 2 + 0.5 * 8   # a full period later
            @test Array(interior(qcw[Time(1.0)]))[1, 1, 1] == 8.0

            # Windowed on-disk sources feeding a windowed materialization
            path_a = "ftsmat_a_$(typeof(arch)).jld2"
            path_b = "ftsmat_b_$(typeof(arch)).jld2"
            rm(path_a, force=true); rm(path_b, force=true)
            host_grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
            fta = FieldTimeSeries{Center, Center, Center}(host_grid, times; backend=OnDisk(), path=path_a, name="a")
            ftb = FieldTimeSeries{Center, Center, Center}(host_grid, times; backend=OnDisk(), path=path_b, name="b")
            snapshot = CenterField(host_grid)
            for n in 1:length(times)
                set!(snapshot, n);  set!(fta, snapshot, n)
                set!(snapshot, 2n); set!(ftb, snapshot, n)
            end
            aw = FieldTimeSeries(path_a, "a"; backend=InMemory(2), architecture=arch)
            bw = FieldTimeSeries(path_b, "b"; backend=InMemory(2), architecture=arch)

            qww = FieldTimeSeries(aw * bw; backend=InMemory(2))
            for n in (1, 3, 4, 2, 1)   # forward and backward
                @test Array(interior(qww[n]))[1, 1, 1] == n * 2n
            end
            @test Array(interior(qww[Time(2.5)]))[1, 1, 1] == 25.0

            # Derived window longer than a source window warns at construction
            qw3 = @test_logs (:warn, r"shorter than the materialized window") FieldTimeSeries(aw * bw; backend=InMemory(3))
            @test Array(interior(qw3[4]))[1, 1, 1] == 32

            rm(path_a, force=true); rm(path_b, force=true)
        end

        @testset "FieldTimeSeriesOperation operator registration and GPU adaptation [$(typeof(arch))]" begin
            @info "  Testing FieldTimeSeriesOperation operator registration and GPU adaptation [$(typeof(arch))]..."
            grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 1, 1))
            times = 0:1.0:3
            a = FieldTimeSeries{Center, Center, Center}(grid, times)
            b = FieldTimeSeries{Center, Center, Center}(grid, times)
            for n in 1:length(times)
                set!(a[n], n)      # a = 1, 2, 3, 4
                set!(b[n], 2n)     # b = 2, 4, 6, 8
            end

            # Operators registered with @unary / @binary after load work on FieldTimeSeries
            p = plus_two(a)
            @test p isa FieldTimeSeriesOperation
            @test on_cpu(p)[1, 1, 1, 2] == 4
            h = harmonic(a, b)
            @test h isa FieldTimeSeriesOperation
            @test on_cpu(h)[1, 1, 1, 2] ≈ 2 * 2 * 4 / (2 + 4)
            @test on_cpu(harmonic(a, 2))[1, 1, 1, 2] ≈ 2 * 2 * 2 / (2 + 2)

            # Default operator coverage beyond arithmetic: comparisons, atan, mod
            @test (a > b) isa FieldTimeSeriesOperation
            @test on_cpu(a > b)[1, 1, 1, 2] == false
            @test on_cpu(b >= 2a)[1, 1, 1, 2] == true
            @test on_cpu(atan(a, b))[1, 1, 1, 2] ≈ atan(2, 4)

            # Adapted operations evaluate pointwise without host-side slicing
            q = a * b
            adapted = Adapt.adapt(Array, q)
            @test adapted isa GPUAdaptedFieldTimeSeriesOperation
            @test adapted[1, 1, 1, 2] == 8
            @test adapted[1, 1, 1, Time(0.5)] == 5.0
            @test size(adapted) == (2, 2, 2, 4)
            @test axes(adapted, 4) == Base.OneTo(4)
            @test length(adapted) == 4
            @test Adapt.adapt(Array, sqrt(a^2 + b^2))[1, 1, 1, 2] ≈ sqrt(4 + 16)
            @test Adapt.adapt(Array, 2 * a + b / 2 - 1)[1, 1, 1, 3] == 8

            # Kernel-function form: arguments are passed time-sliced to the kernel function
            @inline scaled_product(i, j, k, grid, a, b, scale) = @inbounds scale * a[i, j, k] * b[i, j, k]
            qk = FieldTimeSeriesOperation{Center, Center, Center}(scaled_product, grid, a, b, 10)
            ak = Adapt.adapt(Array, qk)
            @test ak[1, 1, 1, 2] == 80
            @test ak[1, 1, 1, Time(0.5)] == 0.5 * 20 + 0.5 * 80

            # Field arguments at other locations are spatially interpolated with
            # operators stored at construction, on the host and GPU-adapted forms alike
            u2 = XFaceField(grid)
            set!(u2, 1)
            Oceananigans.BoundaryConditions.fill_halo_regions!(u2)
            w2 = a * u2
            @test pointwise_evaluable(q)
            @test pointwise_evaluable(w2)
            @test on_cpu(w2)[2, 1, 1, 2] == 2   # interpolates u2 to Center
            @test Adapt.adapt(Array, w2)[2, 1, 1, 2] == 2
            @test Adapt.adapt(Array, FieldTimeSeriesOperation{Center, Center, Center}(scaled_product, grid, a, u2, 1)) isa
                  GPUAdaptedFieldTimeSeriesOperation

            # In-kernel Time sampling through a KernelFunctionOperation (the forcing pattern)
            @inline sample_series(i, j, k, grid, q, t) = q[i, j, k, Time(t)]
            kfo = KernelFunctionOperation{Center, Center, Center}(sample_series, grid, q, 0.5)
            f = compute!(Field(kfo))
            @test Array(interior(f))[1, 1, 1] == 5.0

            # on_architecture round-trips
            q2 = on_cpu(q)
            @test q2 isa FieldTimeSeriesOperation
            @test q2[1, 1, 1, 2] == 8

            # Reductions mirror the FieldTimeSeries reductions and match reducing
            # the materialized series
            qfts = FieldTimeSeries(q)
            @test maximum(q) == maximum(qfts) == 32
            @test minimum(q) == 2
            @test sum(q) == sum(qfts)
            @test mean(q) == mean(qfts)
            qs = sum(q; dims=(1, 2))
            @test qs isa FieldTimeSeries
            @test Array(interior(qs))[1, 1, 1, 3] == Array(interior(sum(qfts; dims=(1, 2))))[1, 1, 1, 3] == 4 * 18
            @test Array(interior(sum(q; dims=4)))[1, 1, 1, 1] == 2 + 8 + 18 + 32          # time reduction
            @test Array(interior(mean(q; dims=4)))[1, 1, 1, 1] == 15.0
            @test Array(interior(maximum(q; dims=4)))[1, 1, 1, 1] == 32

            # Mapped reductions apply the function in the pure-time branch too
            @test Array(interior(sum(abs2, q; dims=4)))[1, 1, 1, 1] == 4 + 64 + 324 + 1024
            @test Array(interior(sum(abs2, qfts; dims=4)))[1, 1, 1, 1] == 4 + 64 + 324 + 1024
            @test Array(interior(maximum(abs, -1 * q; dims=4)))[1, 1, 1, 1] == 32
            @test mean(abs2, q) == (4 + 64 + 324 + 1024) / 4

            # Lazy Average / Integral / CumulativeIntegral of a series: slices are Scans
            qa = Average(q, dims=(1, 2, 3))
            @test qa isa FieldTimeSeriesOperation
            @test Array(interior(compute!(Field(qa[2]))))[1, 1, 1] == 8
            @test Array(interior(qa[Time(0.5)]))[1, 1, 1] == 5.0
            qafts = FieldTimeSeries(qa)
            @test Array(interior(qafts[3]))[1, 1, 1] == 18
            @test Array(interior(sum(qa; dims=4)))[1, 1, 1, 1] == 60
            qi = Integral(q, dims=(1, 2, 3))
            @test Array(interior(compute!(Field(qi[2]))))[1, 1, 1] ≈ 8
            @test CumulativeIntegral(q; dims=3) isa FieldTimeSeriesOperation

            # Reductions require a full sweep: pointwise indexing, GPU adaptation,
            # and composition are refused with instructive errors
            @test_throws ArgumentError qa[1, 1, 1, 2]
            @test_throws ArgumentError Adapt.adapt(Array, qa)
            @test_throws ArgumentError b - qa

            # Both flavors of series share the AbstractFieldTimeSeries supertype
            @test a isa AbstractFieldTimeSeries
            @test q isa AbstractFieldTimeSeries

            # Spatial derivatives build lazy 4-D operations over Derivative slices
            c2 = FieldTimeSeries{Center, Center, Center}(grid, times)
            for n in 1:length(times)
                set!(c2[n], (x, y, z) -> n * x)
            end
            dc = ∂x(c2)
            @test dc isa FieldTimeSeriesOperation
            @test on_cpu(dc)[2, 1, 1, 3] ≈ 3           # ∂x(3x) = 3
            @test on_cpu(dc)[2, 1, 1, Time(0.5)] ≈ 1.5 # blend of node derivatives
            @test on_cpu(∂y(c2))[2, 2, 2, 2] ≈ 0

            # Temporal derivative: finite-difference node values (centered interior,
            # one-sided ends), Time-interpolated like any other series
            qt = FieldTimeSeries{Center, Center, Center}(grid, times)
            for n in 1:length(times)
                set!(qt[n], n^2)   # 1, 4, 9, 16
            end
            dq = ∂t(qt)
            @test dq isa FieldTimeSeriesOperation
            cpu_dq = on_cpu(dq)   # also exercises on_architecture of ∂t operations
            @test cpu_dq[1, 1, 1, 1] == 3   # (4 - 1) / 1
            @test cpu_dq[1, 1, 1, 2] == 4   # (9 - 1) / 2
            @test cpu_dq[1, 1, 1, 4] == 7   # (16 - 9) / 1
            @test cpu_dq[1, 1, 1, Time(0.5)] == 3.5
            @test Adapt.adapt(Array, dq)[1, 1, 1, 2] == 4
            dfts = FieldTimeSeries(dq)   # lazy ≡ materialized
            @test Array(interior(dfts[Time(2.3)]))[1, 1, 1] == Array(interior(dq[Time(2.3)]))[1, 1, 1]

            # ∂t works on fully-in-memory series with fewer than 4 times
            short_series = FieldTimeSeries{Center, Center, Center}(grid, 0:1.0:2)
            for n in 1:3
                set!(short_series[n], n^2)
            end
            @test on_cpu(∂t(short_series))[1, 1, 1, 2] == 4

            # ∂t differentiates DateTime-based times in seconds
            date_series = FieldTimeSeries{Center, Center, Center}(grid, [DateTime(2020, 1, d) for d in 1:4])
            for n in 1:4
                set!(date_series[n], n)
            end
            @test Array(interior(compute!(Field(∂t(date_series)[2]))))[1, 1, 1] ≈ 2 / (2 * 86400)

            # Cyclical wraps the stencil across the period
            qc = FieldTimeSeries{Center, Center, Center}(grid, times; time_indexing=Cyclical(4.0))
            for n in 1:length(times)
                set!(qc[n], n)
            end
            @test on_cpu(∂t(qc))[1, 1, 1, 1] == -1   # (2 - 4) / (1 - 3 + 4)

            # Partly-in-memory arguments need a window of at least 4
            window_guard_path = "dt_window_guard_$(typeof(arch)).jld2"
            rm(window_guard_path, force=true)
            host_grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
            source_series = FieldTimeSeries{Center, Center, Center}(host_grid, times; backend=OnDisk(), path=window_guard_path, name="q")
            snapshot = CenterField(host_grid)
            for n in 1:length(times)
                set!(snapshot, n); set!(source_series, snapshot, n)
            end
            @test_throws ArgumentError ∂t(FieldTimeSeries(window_guard_path, "q"; backend=InMemory(2), architecture=arch))
            rm(window_guard_path, force=true)
        end
    end

    @testset "FieldTimeSeriesOperation model integration" begin
        @info "  Testing FieldTimeSeriesOperation model integration..."
        grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
        times = 0:1.0:3
        a = FieldTimeSeries{Center, Center, Center}(grid, times)
        b = FieldTimeSeries{Center, Center, Center}(grid, times)
        for n in 1:length(times)
            set!(a[n], n)      # a = 1, 2, 3, 4
            set!(b[n], 2n)     # b = 2, 4, 6, 8
        end
        q = a * b

        # Forcing by an operation time-interpolates like forcing by a FieldTimeSeries
        tracer = CenterField(grid)
        forcing = materialize_forcing(q, tracer, :T, (:T,))
        @test forcing.func(1, 1, 1, grid, Clock(time=0.5), nothing, forcing.parameters) == 5.0

        # Boundary conditions by a reduced operation time-interpolate
        ar = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        br = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        for n in 1:length(times)
            set!(ar[n], n)
            set!(br[n], 2n)
        end
        bc = FluxBoundaryCondition(ar * br)
        @test getbc(bc, 1, 1, grid, Clock(time=2.5), nothing) == 25.0

        # Prescribed velocities by an operation interpolate in time; location mismatches throw
        uf = FieldTimeSeries{Face, Center, Center}(grid, times)
        vf = FieldTimeSeries{Face, Center, Center}(grid, times)
        for n in 1:length(times)
            set!(uf[n], n)
            set!(vf[n], 2n)
        end
        prescribed = hydrostatic_velocity_fields(PrescribedVelocityFields(u = uf * vf), grid, Clock(time=0.0), nothing)
        @test prescribed.u isa TimeSeriesInterpolation
        prescribed.u.clock.time = 0.5
        @test prescribed.u[1, 1, 1] == 5.0
        @test_throws ArgumentError hydrostatic_velocity_fields(PrescribedVelocityFields(u = q), grid, Clock(time=0.0), nothing)

        # Relaxation toward a colocated operation target time-interpolates
        relaxation = Relaxation(rate=1, target=q)
        materialized = materialize_forcing(relaxation, tracer, :T, (:T,))
        @test evaluate_target(materialized.target, 1, 1, 1, (0, 0, 0), 0.5) == 5.0

        # A model updates operation windows through extract_field_time_series, so the
        # in-kernel ∂t stencil sees correctly-positioned (widened) argument windows
        derivative_path = "dt_model_protocol.jld2"
        rm(derivative_path, force=true)
        long_times = 0:1.0:9
        source_series = FieldTimeSeries{Center, Center, Center}(grid, long_times; backend=OnDisk(), path=derivative_path, name="q")
        snapshot = CenterField(grid)
        for n in 1:length(long_times)
            set!(snapshot, n^2); set!(source_series, snapshot, n)
        end
        windowed_series = FieldTimeSeries(derivative_path, "q"; backend=InMemory(4))
        windowed_derivative = ∂t(windowed_series)
        for fts in extract_field_time_series(windowed_derivative)
            update_field_time_series!(fts, Time(5.5))
        end
        @test Adapt.adapt(Array, windowed_derivative)[1, 1, 1, Time(5.5)] == 13.0
        rm(derivative_path, force=true)

        # A model with an operation forcing steps forward
        model = NonhydrostaticModel(grid; tracers=:T, forcing=(; T = q))
        time_step!(model, 0.25)
        @test isfinite(first(Array(interior(model.tracers.T))))
    end

    for output_writer in (JLD2Writer, NetCDFWriter)
        filepath1d, filepath2d, filepath3d, unsplit_filepath, split_filepath = generate_some_interesting_simulation_data(Nx, Ny, Nz; output_writer)

        for arch in archs
            @testset "FieldTimeSeries{InMemory} [$(typeof(arch))] with $output_writer" begin
                @info "  Testing FieldTimeSeries{InMemory} [$(typeof(arch))]..."
                test_field_time_series_in_memory_3d(arch, filepath3d, Nx, Ny, Nz, Nt)

                if output_writer == JLD2Writer
                    test_field_time_series_in_memory_2d(arch, filepath2d, Nx, Ny, Nt) # NetCDFWriter does not support 2D sliced fields with halos yet
                    test_field_time_series_in_memory_1d(arch, filepath1d, Nz, Nt) # FieldTimeSeries with NetCDF does not support 1D fields yet
                    test_field_time_series_in_memory_split(arch, split_filepath, unsplit_filepath) # FieldTimeSeries with NetCDF does not support split fields yet
                end
            end

            if output_writer == JLD2Writer
                @testset "FieldTimeSeries with Function boundary conditions [$(typeof(arch))] with $output_writer" begin
                    @info "  Testing FieldTimeSeries with Function boundary conditions..."
                    test_field_time_series_function_boundary_conditions(arch)
                end
            end

            if arch isa CPU
                @testset "FieldTimeSeries pickup" begin
                    @info "  Testing FieldTimeSeries pickup with $output_writer"
                    test_field_time_series_pickup(arch)
                end
            end

            if output_writer == JLD2Writer
                @testset "FieldTimeSeries with split files [$(typeof(arch))]" begin
                    @info "  Testing FieldTimeSeries with split files [$(typeof(arch))]..."
                    test_field_time_series_split_files(arch)
                end
            end

            @testset "FieldTimeSeries with Array boundary conditions [$(typeof(arch))] with $output_writer" begin
                @info "  Testing FieldTimeSeries with Array boundary conditions..."
                test_field_time_series_array_boundary_conditions(arch)
            end

            # TODO: Make FieldTimeSeries{OnDisk} work with NetCDFWriter
            if output_writer == JLD2Writer
                @testset "FieldTimeSeries{OnDisk} [$(typeof(arch))] with $output_writer" begin
                    @info "  Testing FieldTimeSeries{OnDisk} [$(typeof(arch))]..."
                    test_field_time_series_on_disk(arch, filepath3d, filepath1d, Nx, Ny, Nz, Nt)
                end
            end

            @testset "FieldTimeSeries{InMemory} reductions with $output_writer" begin
                @info "  Testing FieldTimeSeries{InMemory} reductions..."
                test_field_time_series_reductions(filepath3d, Nt)
            end
        end

        # TODO: Make all of these features work with NetCDFWriter
        if output_writer == JLD2Writer
            @testset "Test chunked abstraction with $output_writer" begin
                @info "  Testing Chunked abstraction..."
                test_chunked_abstraction(filepath3d, "T")
            end

            for Backend in [InMemory, OnDisk]
                @testset "FieldTimeSeries{$Backend} parallel reading with $output_writer" begin
                    @info "  Testing FieldTimeSeries{$Backend} parallel reading..."
                    test_field_time_series_parallel_reading(Backend, filepath3d)
                end
            end

            for Backend in [InMemory, OnDisk]
                @testset "FieldDataset{$Backend} indexing with $output_writer" begin
                    @info "  Testing FieldDataset{$Backend} indexing..."
                    test_field_dataset_indexing(Backend, filepath3d)
                end
            end

            for Backend in [InMemory, OnDisk]
                @testset "FieldDataset{$Backend} parallel reading with $output_writer" begin
                    @info "  Testing FieldDataset{$Backend} parallel reading..."
                    test_field_dataset_parallel_reading(Backend, filepath3d)
                end
            end
        end

        rm(filepath1d)
        rm(filepath2d, force=true) # This file doesn't exist if we use NetCDFWriter
        rm(filepath3d)
    end

    @testset "FieldTimeSeries reductions with dims" begin
        @info "  Testing FieldTimeSeries reductions with dims..."
        test_field_time_series_reductions_with_dims()
    end

    @testset "FieldTimeSeries with singleton integer indices" begin
        @info "  Testing FieldTimeSeries with singleton integer indices..."
        grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
        times = [0.0, 1.0]

        # Integer indices should work the same as UnitRange indices
        fts_int = FieldTimeSeries{Center, Center, Center}(grid, times; indices=(:, :, 4))
        fts_range = FieldTimeSeries{Center, Center, Center}(grid, times; indices=(:, :, 4:4))
        @test size(fts_int) == size(fts_range)
        @test indices(fts_int) == indices(fts_range)

        # Also test integer indices in other dimensions
        fts_i = FieldTimeSeries{Center, Center, Center}(grid, times; indices=(2, :, :))
        @test size(fts_i) == (1, 4, 4, 2)

        fts_j = FieldTimeSeries{Center, Center, Center}(grid, times; indices=(:, 3, :))
        @test size(fts_j) == (4, 1, 4, 2)

        # Test with loc/grid constructor directly
        fts_loc = FieldTimeSeries((Center(), Center(), Center()), grid, times; indices=(:, :, 4))
        @test size(fts_loc) == (4, 4, 1, 2)
    end

    @testset "Time Interpolation" begin
        test_time_interpolation()
    end

    filepath_sine = "one_dimensional_sine.jld2"
    @testset "Test interpolation using `InMemory` backend" begin
        test_interpolation_with_in_memory_backends(filepath_sine)
    end
    rm(filepath_sine)
end
