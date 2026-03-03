include("dependencies_for_runtests.jl")

using Oceananigans.AbstractOperations: Histogram
using Oceananigans.Operators: Vᶜᶜᶜ

function reference_bin(value, edges)
    if value < first(edges) || value > last(edges)
        return 0
    elseif value == last(edges)
        return length(edges) - 1
    end

    idx = searchsortedlast(edges, value)
    return (1 <= idx < length(edges)) ? idx : 0
end

function reference_histogram(a, b, grid, edges1, edges2; weights=:count)
    FT = promote_type(eltype(a), eltype(b), eltype(edges1), eltype(edges2))
    histogram = zeros(FT, length(edges1) - 1, length(edges2) - 1)

    Nx, Ny, Nz = size(a)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        i1 = reference_bin(a[i, j, k], edges1)
        i2 = reference_bin(b[i, j, k], edges2)

        in_range = (i1 > 0) & (i2 > 0)
        if in_range
            weight = if weights === :count
                1
            elseif weights === :cell_volume
                Vᶜᶜᶜ(i, j, k, grid)
            else
                weights[i, j, k]
            end

            histogram[i1, i2] += weight
        end
    end

    return histogram
end

function histogram_constructor_validation()
    grid = RectilinearGrid(CPU(), size=(4, 4, 4), extent=(1, 1, 1))
    a = CenterField(grid)
    b = CenterField(grid)
    b_face = XFaceField(grid)
    w_face = XFaceField(grid)
    grid2 = RectilinearGrid(CPU(), size=(4, 4, 4), extent=(1, 1, 1))
    w_other = CenterField(grid2)

    valid_bins = (x = [0.0, 1.0], y = [0.0, 1.0])

    @test_throws ArgumentError Histogram(a, b; bins=(x=[0.0, 1.0],), weights=:count)
    @test_throws ArgumentError Histogram(a, b; bins=(x=[0.0, 1.0, 0.5], y=[0.0, 1.0]), weights=:count)
    @test_throws ArgumentError Histogram(a, b_face; bins=valid_bins, weights=:count)
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, method=:mean)
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, dims=(1, 2))
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, weights=:mass)
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, weights=w_face)
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, weights=w_other)
end

function histogram_is_correct(arch, FT)
    grid = RectilinearGrid(arch, FT, size=(6, 5, 4), extent=(6, 5, 4))
    model = NonhydrostaticModel(grid; tracers=(:a, :b, :w))

    set!(model, a=(x, y, z) -> FT(0.8x + 0.2z),
                b=(x, y, z) -> FT(31 + 0.3y - 0.1z),
                w=(x, y, z) -> FT(2 + 0.1x + 0.2y + 0.3z))

    edges1 = FT[0, 1, 2, 3, 4, 5, 6]
    edges2 = FT[30, 30.5, 31, 31.5, 32, 32.5, 33]
    bins = (a = edges1, b = edges2)

    a_cpu = Array(interior(model.tracers.a))
    b_cpu = Array(interior(model.tracers.b))
    w_cpu = Array(interior(model.tracers.w))

    count_field = Field(Histogram(model.tracers.a, model.tracers.b; bins, weights=:count))
    volume_field = Field(Histogram(model.tracers.a, model.tracers.b; bins, weights=:cell_volume))
    weighted_field = Field(Histogram(model.tracers.a, model.tracers.b; bins, weights=model.tracers.w))

    count_hist = Array(interior(count_field))[:, :, 1]
    volume_hist = Array(interior(volume_field))[:, :, 1]
    weighted_hist = Array(interior(weighted_field))[:, :, 1]

    expected_count = reference_histogram(a_cpu, b_cpu, grid, edges1, edges2; weights=:count)
    expected_volume = reference_histogram(a_cpu, b_cpu, grid, edges1, edges2; weights=:cell_volume)
    expected_weighted = reference_histogram(a_cpu, b_cpu, grid, edges1, edges2; weights=w_cpu)

    @test count_hist == expected_count
    @test volume_hist ≈ expected_volume
    @test weighted_hist ≈ expected_weighted
end

function histogram_edge_semantics()
    grid = RectilinearGrid(CPU(), size=(2, 2, 1), extent=(1, 1, 1))
    a = CenterField(grid)
    b = CenterField(grid)

    set!(a, reshape([0.0, 1.0, 2.0, 2.5], 2, 2, 1))
    set!(b, reshape([0.0, 1.2, 2.0, -1.0], 2, 2, 1))

    bins = (a = [0.0, 1.0, 2.0], b = [0.0, 1.0, 2.0])
    h = Field(Histogram(a, b; bins, weights=:count))
    histogram = Array(interior(h))[:, :, 1]

    @test histogram == [1.0 0.0; 0.0 2.0]
end

function histogram_writer_smoke_test()
    mktempdir() do dir
        grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid; tracers=(:a, :b))
        set!(model, a=(x, y, z) -> x + z,
                    b=(x, y, z) -> y - z)

        bins = (a = collect(range(0.0, stop=2.0, length=5)),
                b = collect(range(-1.0, stop=2.0, length=7)))

        histogram = Field(Histogram(model.tracers.a, model.tracers.b; bins, weights=:count))
        simulation = Simulation(model; Δt=1, stop_iteration=1)
        simulation.output_writers[:histogram] = JLD2Writer(model, (; histogram);
                                                           schedule = IterationInterval(1),
                                                           dir = dir,
                                                           filename = "histogram_operation_test.jld2",
                                                           with_halos = false,
                                                           overwrite_existing = true)

        run!(simulation)

        filepath = joinpath(dir, "histogram_operation_test.jld2")
        jldopen(filepath, "r") do file
            snapshot_keys = collect(keys(file["timeseries/histogram"]))
            @test !isempty(snapshot_keys)

            latest = string(maximum(parse.(Int, snapshot_keys)))
            snapshot = file["timeseries/histogram/$latest"]

            @test size(snapshot, 1) == length(bins.a) - 1
            @test size(snapshot, 2) == length(bins.b) - 1
            @test sum(snapshot) > 0
        end
    end

    return nothing
end

@testset "Histogram operation" begin
    @info "Testing Histogram operation..."

    @testset "Constructor validation" begin
        histogram_constructor_validation()
    end

    @testset "Bin edge semantics" begin
        histogram_edge_semantics()
    end

    for arch in archs
        @testset "Correctness [$(typeof(arch))]" begin
            @info "  Testing Histogram correctness [$(typeof(arch))]..."
            for FT in float_types
                histogram_is_correct(arch, FT)
            end
        end
    end

    @testset "Writer integration" begin
        histogram_writer_smoke_test()
    end
end
