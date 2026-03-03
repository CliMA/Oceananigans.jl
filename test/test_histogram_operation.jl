include("dependencies_for_runtests.jl")

using Oceananigans.AbstractOperations: Histogram
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans: location

function reference_bin(value, edges)
    if value < first(edges) || value > last(edges)
        return 0
    elseif value == last(edges)
        return length(edges) - 1
    end

    idx = searchsortedlast(edges, value)
    return (1 <= idx < length(edges)) ? idx : 0
end

@inline function reference_retained_index(i, j, k, dims, Nx, Ny, Nz)
    reduced = (1 in dims, 2 in dims, 3 in dims)

    iR = reduced[1] ? 1 : i
    jR = reduced[2] ? 1 : j
    kR = reduced[3] ? 1 : k

    len1 = reduced[1] ? 1 : Nx
    len2 = reduced[2] ? 1 : Ny

    return iR + len1 * ((jR - 1) + len2 * (kR - 1))
end

function reference_histogram(a, b, grid, edges1, edges2; weights=:count, dims=(1, 2, 3))
    FT = promote_type(eltype(a), eltype(b), eltype(edges1), eltype(edges2))

    Nx, Ny, Nz = size(a)
    nret = (1 in dims ? 1 : Nx) *
           (2 in dims ? 1 : Ny) *
           (3 in dims ? 1 : Nz)

    histogram = zeros(FT, length(edges1) - 1, length(edges2) - 1, nret)

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

            r = reference_retained_index(i, j, k, dims, Nx, Ny, Nz)
            histogram[i1, i2, r] += weight
        end
    end

    return histogram
end

function reference_histogram_1d(a, grid, edges1; weights=:count, dims=(1, 2, 3))
    FT = promote_type(eltype(a), eltype(edges1))

    Nx, Ny, Nz = size(a)
    nret = (1 in dims ? 1 : Nx) *
           (2 in dims ? 1 : Ny) *
           (3 in dims ? 1 : Nz)

    histogram = zeros(FT, length(edges1) - 1, nret, 1)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        i1 = reference_bin(a[i, j, k], edges1)

        if i1 > 0
            weight = if weights === :count
                1
            elseif weights === :cell_volume
                Vᶜᶜᶜ(i, j, k, grid)
            else
                weights[i, j, k]
            end

            r = reference_retained_index(i, j, k, dims, Nx, Ny, Nz)
            histogram[i1, r, 1] += weight
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
    @test_throws ArgumentError Histogram((S=a, T=b); bins=valid_bins, weights=:count)
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, method=:mean)
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, dims=())
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, dims=(1, 4))
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, weights=:mass)
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, weights=w_face)
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, weights=w_other)
    @test_throws ArgumentError Histogram(a, b; bins=valid_bins, weights=:cell_volume, method=:integral)

    @test_throws ArgumentError Histogram(a; bins=valid_bins, weights=:count)
    @test_throws ArgumentError Histogram(a; bins=[0.0, 1.0, 0.5], weights=:count)
    @test_throws ArgumentError Histogram(a; bins=[0.0, 1.0], weights=w_face)
    @test_throws ArgumentError Histogram(a; bins=[0.0, 1.0], weights=w_other)
    @test_throws ArgumentError Histogram(a; bins=[0.0, 1.0], weights=:cell_volume, method=:integral)

    @test Histogram(a, b; bins=valid_bins, weights=:count, method=:integral) isa AbstractOperation
    @test Histogram(a; bins=[0.0, 1.0], weights=:count, method=:integral) isa AbstractOperation
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

    count_hist = Array(interior(count_field))
    volume_hist = Array(interior(volume_field))
    weighted_hist = Array(interior(weighted_field))

    expected_count = reference_histogram(a_cpu, b_cpu, grid, edges1, edges2; weights=:count)
    expected_volume = reference_histogram(a_cpu, b_cpu, grid, edges1, edges2; weights=:cell_volume)
    expected_weighted = reference_histogram(a_cpu, b_cpu, grid, edges1, edges2; weights=w_cpu)

    @test count_hist == expected_count
    @test volume_hist ≈ expected_volume
    @test weighted_hist ≈ expected_weighted
end

function histogram_partial_reduction_is_correct(arch, FT)
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

    h_z = Field(Histogram(model.tracers.a, model.tracers.b; bins, weights=:count, dims=(3,)))
    h_xy = Field(Histogram(model.tracers.a, model.tracers.b; bins, weights=model.tracers.w, dims=(1, 2)))

    h_z_cpu = Array(interior(h_z))
    h_xy_cpu = Array(interior(h_xy))

    expected_z = reference_histogram(a_cpu, b_cpu, grid, edges1, edges2; weights=:count, dims=(3,))
    expected_xy = reference_histogram(a_cpu, b_cpu, grid, edges1, edges2; weights=w_cpu, dims=(1, 2))

    @test h_z_cpu == expected_z
    @test h_xy_cpu ≈ expected_xy
end

function histogram_1d_is_correct(arch, FT)
    grid = RectilinearGrid(arch, FT, size=(6, 5, 4), extent=(6, 5, 4))
    model = NonhydrostaticModel(grid; tracers=(:a, :w))

    set!(model, a=(x, y, z) -> FT(0.8x + 0.2z),
                w=(x, y, z) -> FT(2 + 0.1x + 0.2y + 0.3z))

    edges1 = FT[0, 1, 2, 3, 4, 5, 6]

    a_cpu = Array(interior(model.tracers.a))
    w_cpu = Array(interior(model.tracers.w))

    h_all = Field(Histogram(model.tracers.a; bins=edges1, weights=:count))
    h_xy = Field(Histogram(model.tracers.a; bins=edges1, weights=model.tracers.w, dims=(1, 2)))

    h_all_cpu = Array(interior(h_all))
    h_xy_cpu = Array(interior(h_xy))

    expected_all = reference_histogram_1d(a_cpu, grid, edges1; weights=:count)
    expected_xy = reference_histogram_1d(a_cpu, grid, edges1; weights=w_cpu, dims=(1, 2))

    @test h_all_cpu == expected_all
    @test h_xy_cpu ≈ expected_xy

    h_named = Field(Histogram((rho=model.tracers.a,); bins=(rho=edges1,), weights=:count, dims=(1, 2)))
    @test Array(interior(h_named)) == reference_histogram_1d(a_cpu, grid, edges1; weights=:count, dims=(1, 2))
end

function histogram_integral_method_matches_sum_with_metric_2d(arch, FT)
    grid = RectilinearGrid(arch, FT, size=(6, 5, 4), extent=(6, 5, 4))
    model = NonhydrostaticModel(grid; tracers=(:a, :b, :w))

    set!(model, a=(x, y, z) -> FT(0.8x + 0.2z),
                b=(x, y, z) -> FT(31 + 0.3y - 0.1z),
                w=(x, y, z) -> FT(2 + 0.1x + 0.2y + 0.3z))

    edges1 = FT[0, 1, 2, 3, 4, 5, 6]
    edges2 = FT[30, 30.5, 31, 31.5, 32, 32.5, 33]
    bins = (a = edges1, b = edges2)
    dims = (1, 2)

    metric = Oceananigans.AbstractOperations.grid_metric_operation(location(model.tracers.a),
                                                                   Oceananigans.AbstractOperations.reduction_grid_metric(dims),
                                                                   grid)

    weighted_integral = Field(Histogram(model.tracers.a, model.tracers.b; bins, weights=model.tracers.w, dims, method=:integral))
    weighted_sum = Field(Histogram(model.tracers.a, model.tracers.b; bins, weights=model.tracers.w * metric, dims, method=:sum))

    count_integral = Field(Histogram(model.tracers.a, model.tracers.b; bins, weights=:count, dims, method=:integral))
    count_sum = Field(Histogram(model.tracers.a, model.tracers.b; bins, weights=metric, dims, method=:sum))

    @test Array(interior(weighted_integral)) ≈ Array(interior(weighted_sum))
    @test Array(interior(count_integral)) ≈ Array(interior(count_sum))
end

function histogram_integral_method_matches_sum_with_metric_1d(arch, FT)
    grid = RectilinearGrid(arch, FT, size=(6, 5, 4), extent=(6, 5, 4))
    model = NonhydrostaticModel(grid; tracers=(:a, :w))

    set!(model, a=(x, y, z) -> FT(0.8x + 0.2z),
                w=(x, y, z) -> FT(2 + 0.1x + 0.2y + 0.3z))

    bins = FT[0, 1, 2, 3, 4, 5, 6]
    dims = (1, 2)

    metric = Oceananigans.AbstractOperations.grid_metric_operation(location(model.tracers.a),
                                                                   Oceananigans.AbstractOperations.reduction_grid_metric(dims),
                                                                   grid)

    weighted_integral = Field(Histogram(model.tracers.a; bins, weights=model.tracers.w, dims, method=:integral))
    weighted_sum = Field(Histogram(model.tracers.a; bins, weights=model.tracers.w * metric, dims, method=:sum))

    count_integral = Field(Histogram(model.tracers.a; bins, weights=:count, dims, method=:integral))
    count_sum = Field(Histogram(model.tracers.a; bins, weights=metric, dims, method=:sum))

    @test Array(interior(weighted_integral)) ≈ Array(interior(weighted_sum))
    @test Array(interior(count_integral)) ≈ Array(interior(count_sum))
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

function histogram_named_operands_map_bins_by_key(arch, FT)
    grid = RectilinearGrid(arch, FT, size=(6, 5, 4), extent=(6, 5, 4))
    model = NonhydrostaticModel(grid; tracers=(:a, :b))

    set!(model, a=(x, y, z) -> FT(0.8x + 0.2z),
                b=(x, y, z) -> FT(31 + 0.3y - 0.1z))

    a_edges = FT[0, 1, 2, 3, 4, 5, 6]
    b_edges = FT[30, 30.5, 31, 31.5, 32, 32.5, 33]

    # Named operands map bins by key, not by tuple order.
    h1 = Field(Histogram((S=model.tracers.b, T=model.tracers.a); bins=(T=a_edges, S=b_edges), weights=:count))
    h2 = Field(Histogram((S=model.tracers.b, T=model.tracers.a); bins=(S=b_edges, T=a_edges), weights=:count))

    h1_cpu = Array(interior(h1))
    h2_cpu = Array(interior(h2))

    @test h1_cpu == h2_cpu

    a_cpu = Array(interior(model.tracers.a))
    b_cpu = Array(interior(model.tracers.b))
    expected = reference_histogram(b_cpu, a_cpu, grid, b_edges, a_edges; weights=:count)
    @test h1_cpu == expected
end

function histogram_face_location_includes_bounded_boundary_faces(arch, FT)
    grid = RectilinearGrid(arch, FT, topology=(Bounded, Periodic, Bounded), size=(6, 5, 4), extent=(6, 5, 4))
    a = XFaceField(grid)
    b = XFaceField(grid)

    set!(a, FT(0.5))
    set!(b, FT(34.5))

    a_edges = FT[0, 1]
    b_edges = FT[34, 35]

    histogram = Field(Histogram(a, b; bins=(a=a_edges, b=b_edges), weights=:count))
    histogram_cpu = Array(interior(histogram))

    a_cpu = Array(interior(a))
    b_cpu = Array(interior(b))
    expected = reference_histogram(a_cpu, b_cpu, grid, a_edges, b_edges; weights=:count)

    @test histogram_cpu == expected
    @test sum(histogram_cpu) == length(a_cpu)
end

function histogram_cell_volume_conservation(arch, FT)
    grid = RectilinearGrid(arch, FT, size=(6, 5, 4), extent=(6, 5, 4))
    a = CenterField(grid)
    b = CenterField(grid)

    set!(a, (x, y, z) -> FT(0.5x + 0.25y + 0.1z))
    set!(b, (x, y, z) -> FT(34 + 0.2x - 0.3y - 0.05z))

    a_edges = FT[-1, 0, 1, 2, 3, 4, 5]
    b_edges = FT[32, 33, 34, 35, 36]

    histogram = Field(Histogram(a, b; bins=(a=a_edges, b=b_edges), weights=:cell_volume))
    total_histogram_volume = sum(histogram)
    total_grid_volume = sum(KernelFunctionOperation{Center, Center, Center}(Vᶜᶜᶜ, grid))

    @test total_histogram_volume ≈ total_grid_volume
end

function histogram_writer_smoke_test()
    mktempdir() do dir
        grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid; tracers=(:a, :b))
        set!(model, a=(x, y, z) -> x + z,
                    b=(x, y, z) -> y - z)

        bins2d = (a = collect(range(0.0, stop=2.0, length=5)),
                  b = collect(range(-1.0, stop=2.0, length=7)))
        bins1d = collect(range(0.0, stop=2.0, length=5))

        histogram = Field(Histogram(model.tracers.a, model.tracers.b; bins=bins2d, weights=:count))
        histogram_1d = Field(Histogram(model.tracers.a; bins=bins1d, weights=:count, dims=(1, 2)))

        simulation = Simulation(model; Δt=1, stop_iteration=1)
        simulation.output_writers[:histogram] = JLD2Writer(model, (; histogram, histogram_1d);
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
            snapshot_2d = file["timeseries/histogram/$latest"]
            snapshot_1d = file["timeseries/histogram_1d/$latest"]

            @test size(snapshot_2d, 1) == length(bins2d.a) - 1
            @test size(snapshot_2d, 2) == length(bins2d.b) - 1
            @test sum(snapshot_2d) > 0

            @test size(snapshot_1d, 1) == length(bins1d) - 1
            @test size(snapshot_1d, 2) == size(grid, 3)
            @test sum(snapshot_1d) > 0
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
                histogram_partial_reduction_is_correct(arch, FT)
                histogram_1d_is_correct(arch, FT)
                histogram_integral_method_matches_sum_with_metric_2d(arch, FT)
                histogram_integral_method_matches_sum_with_metric_1d(arch, FT)
                histogram_named_operands_map_bins_by_key(arch, FT)
                histogram_face_location_includes_bounded_boundary_faces(arch, FT)
                histogram_cell_volume_conservation(arch, FT)
            end
        end
    end

    @testset "Writer integration" begin
        histogram_writer_smoke_test()
    end
end
