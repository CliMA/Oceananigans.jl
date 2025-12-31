include("dependencies_for_runtests.jl")

using Oceananigans.Units: Time
using Oceananigans.Fields: indices, interpolate!
using Oceananigans.OutputReaders: Cyclical, Clamp, Linear
using Random

function generate_nonzero_simulation_data(Lx, Δt, FT; architecture=CPU())
    grid = RectilinearGrid(architecture, size=10, x=(0, Lx), topology=(Periodic, Flat, Flat))
    model = NonhydrostaticModel(; grid, tracers = (:T, :S), advection = nothing)
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

function generate_some_interesting_simulation_data(Nx, Ny, Nz; architecture=CPU())
    grid = RectilinearGrid(architecture, size=(Nx, Ny, Nz), extent=(64, 64, 32))

    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(5e-5), bottom = GradientBoundaryCondition(0.01))
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(-3e-4))

    @inline Qˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S
    evaporation_bc = FluxBoundaryCondition(Qˢ, field_dependencies=:S, parameters=3e-7)
    S_bcs = FieldBoundaryConditions(top=evaporation_bc)

    model = NonhydrostaticModel(; grid, tracers = (:T, :S), buoyancy = SeawaterBuoyancy(),
                                boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))

    dTdz = 0.01
    Tᵢ(x, y, z) = 20 + dTdz * z + 1e-6 * randn()
    uᵢ(x, y, z) = 1e-3 * randn()
    set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)

    simulation = Simulation(model, Δt=10.0, stop_time=2minutes)
    wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1minute)
    simulation.callbacks[:wizard] = Callback(wizard)

    u, v, w = model.velocities

    computed_fields = (
        b = buoyancy_field(model),
        ζ = Field(∂x(v) - ∂y(u)),
        ke = Field(√(u^2 + v^2))
    )

    fields_to_output = merge(model.velocities, model.tracers, computed_fields)

    filepath3d = "test_3d_output_with_halos.jld2"
    filepath2d = "test_2d_output_with_halos.jld2"
    filepath1d = "test_1d_output_with_halos.jld2"
    split_filepath = "test_split_output.jld2"
    unsplit_filepath = "test_unsplit_output.jld2"

    simulation.output_writers[:jld2_3d_with_halos] = JLD2Writer(model, fields_to_output,
                                                                filename = filepath3d,
                                                                with_halos = true,
                                                                schedule = TimeInterval(30seconds),
                                                                overwrite_existing = true)

    simulation.output_writers[:jld2_2d_with_halos] = JLD2Writer(model, fields_to_output,
                                                                filename = filepath2d,
                                                                indices = (:, :, grid.Nz),
                                                                with_halos = true,
                                                                schedule = TimeInterval(30seconds),
                                                                overwrite_existing = true)

    profiles = NamedTuple{keys(fields_to_output)}(Field(Average(f, dims=(1, 2))) for f in fields_to_output)

    simulation.output_writers[:jld2_1d_with_halos] = JLD2Writer(model, profiles,
                                                                filename = filepath1d,
                                                                with_halos = true,
                                                                schedule = TimeInterval(30seconds),
                                                                overwrite_existing = true)

    simulation.output_writers[:unsplit_jld2] = JLD2Writer(model, profiles,
                                                          filename = unsplit_filepath,
                                                          with_halos = true,
                                                          schedule = TimeInterval(10seconds),
                                                          overwrite_existing = true)

    simulation.output_writers[:split_jld2] = JLD2Writer(model, profiles,
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

function test_field_time_series_in_memory(arch, filepath3d, filepath2d, filepath1d, split_filepath, unsplit_filepath, Nx, Ny, Nz, Nt)
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

    for fts in (u3, v3, w3, T3, b3, ζ3)
        @test parent(fts) isa ArrayType
    end

    if arch isa CPU
        @test u2[1, 2, 5, 4] isa Number
        @test u2[1] isa Field
        @test v2[2] isa Field
    end

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

    for fts in (u1, v1, w1, T1, b1, ζ1)
        @test parent(fts) isa ArrayType
    end

    if arch isa CPU
        @test u1[1, 1, 3, 4] isa Number
        @test u1[1] isa Field
        @test v1[2] isa Field
    end

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
    model = NonhydrostaticModel(; grid, boundary_conditions = (; u=u_bcs, v=v_bcs))
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
    u_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(u_west), east = OpenBoundaryCondition(u_east, scheme=PerturbationAdvection()))
    model = NonhydrostaticModel(; grid, boundary_conditions = (; u=u_bcs))
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
    filepath1d, filepath2d, filepath3d, unsplit_filepath, split_filepath = generate_some_interesting_simulation_data(Nx, Ny, Nz)

    for arch in archs
        @testset "FieldTimeSeries{InMemory} [$(typeof(arch))]" begin
            @info "  Testing FieldTimeSeries{InMemory} [$(typeof(arch))]..."
            test_field_time_series_in_memory(arch, filepath3d, filepath2d, filepath1d, split_filepath, unsplit_filepath, Nx, Ny, Nz, Nt)
        end

        if arch isa CPU
            @testset "FieldTimeSeries pickup" begin
                @info "  Testing FieldTimeSeries pickup..."
                test_field_time_series_pickup(arch)
            end
        end

        @testset "FieldTimeSeries with Array boundary conditions [$(typeof(arch))]" begin
            @info "  Testing FieldTimeSeries with Array boundary conditions..."
            test_field_time_series_array_boundary_conditions(arch)
        end

        @testset "FieldTimeSeries with Function boundary conditions [$(typeof(arch))]" begin
            @info "  Testing FieldTimeSeries with Function boundary conditions..."
            test_field_time_series_function_boundary_conditions(arch)
        end
    end

    for arch in archs
        @testset "FieldTimeSeries{OnDisk} [$(typeof(arch))]" begin
            @info "  Testing FieldTimeSeries{OnDisk} [$(typeof(arch))]..."
            test_field_time_series_on_disk(arch, filepath3d, filepath1d, Nx, Ny, Nz, Nt)
        end
    end

    for arch in archs
        @testset "FieldTimeSeries{InMemory} reductions" begin
            @info "  Testing FieldTimeSeries{InMemory} reductions..."
            test_field_time_series_reductions(filepath3d, Nt)
        end
    end

    @testset "FieldTimeSeries reductions with dims" begin
        @info "  Testing FieldTimeSeries reductions with dims..."
        test_field_time_series_reductions_with_dims()
    end

    @testset "Test chunked abstraction" begin
        @info "  Testing Chunked abstraction..."
        test_chunked_abstraction(filepath3d, "T")
    end

    @testset "Time Interpolation" begin
        test_time_interpolation()
    end

    for Backend in [InMemory, OnDisk]
        @testset "FieldDataset{$Backend} indexing" begin
            @info "  Testing FieldDataset{$Backend} indexing..."
            test_field_dataset_indexing(Backend, filepath3d)
        end
    end

    for Backend in [InMemory, OnDisk]
        @testset "FieldTimeSeries{$Backend} parallel reading" begin
            @info "  Testing FieldTimeSeries{$Backend} parallel reading..."
            test_field_time_series_parallel_reading(Backend, filepath3d)
        end
    end

    for Backend in [InMemory, OnDisk]
        @testset "FieldDataset{$Backend} parallel reading" begin
            @info "  Testing FieldDataset{$Backend} parallel reading..."
            test_field_dataset_parallel_reading(Backend, filepath3d)
        end
    end

    filepath_sine = "one_dimensional_sine.jld2"

    @testset "Test interpolation using `InMemory` backends" begin
        test_interpolation_with_in_memory_backends(filepath_sine)
    end

    rm(filepath1d)
    rm(filepath2d)
    rm(filepath3d)
    rm(filepath_sine)
end
