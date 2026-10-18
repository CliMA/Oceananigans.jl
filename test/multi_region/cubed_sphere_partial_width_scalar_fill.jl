include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: OffsetArray
using Oceananigans.Grids: FullyConnected
using Oceananigans.MultiRegion: CubedSpherePartition, CubedSphereConnectivity, MultiRegionGrid, MultiRegionObject, getregion
using Oceananigans.BoundaryConditions: fill_halo_regions!

# Run with --check-bounds=yes: the halo kernels use @inbounds, so only forced bounds checking catches storage overruns.

N, Nz, H = 5, 2, 4
cubed_sphere(; H = H) = ConformalCubedSphereGrid(; panel_size = (N, N, Nz), z = (0, 1), horizontal_direction_halo = H)

function subdivided_cubed_sphere(R)
    partition = CubedSpherePartition(; R)
    topology = (FullyConnected, FullyConnected, Bounded)
    panels = Tuple(OrthogonalSphericalShellGrid(CPU(); size = (N, N, Nz), z = (0, 1), halo = (4, 4, 1), topology)
                   for _ in 1:length(partition))
    return MultiRegionGrid{Float64, topology..., typeof(panels[1].z)}(CPU(), partition, CubedSphereConnectivity(partition),
                                                                       MultiRegionObject(panels))
end

code(p, i, j, k) = 1e5 * p + 1e3 * (i + 10) + 10 * (j + 10) + k
poison(p, i, j, k) = reinterpret(Float64, 0x7ff8000000000000 | UInt64(round(Int, code(p, i, j, k))))

function coded_scalar(grid; kwargs...)
    c = CenterField(grid; kwargs...)
    for p in 1:length(grid.partition)
        data = getregion(c, p).data
        for k in axes(data, 3), j in axes(data, 2), i in axes(data, 1)
            data[i, j, k] = (1 <= i <= N && 1 <= j <= N && 1 <= k <= Nz) ? code(p, i, j, k) : poison(p, i, j, k)
        end
    end
    return c
end

raw(a) = reinterpret(UInt64, a)
code_or_poison(p, i, j, k) = (1 <= i <= N && 1 <= j <= N && 1 <= k <= Nz) ? code(p, i, j, k) : poison(p, i, j, k)
snapshot(f) = [copy(parent(getregion(f, r).data)) for r in 1:length(f.grid.partition)]
unchanged(f, before) = all(raw(parent(getregion(f, r).data)) == raw(before[r]) for r in 1:length(f.grid.partition))

function rejects_without_writing(call, f)
    before = snapshot(f)
    threw = try
        call()
        false
    catch e
        e isa ArgumentError || rethrow()
        true
    end
    return threw && unchanged(f, before)
end

edge_strip(w) = [(i, j) for d in 1:w for (i, j) in vcat([(1 - d, t) for t in 1:N], [(N + d, t) for t in 1:N],
                                                         [(t, 1 - d) for t in 1:N], [(t, N + d) for t in 1:N])]

@testset "Partial-width scalar fill: footprint" begin
    grid = cubed_sphere()
    stock = coded_scalar(grid)
    fill_halo_regions!(stock)
    default_call = coded_scalar(grid)
    fill_halo_regions!(default_call; halo_width = nothing)
    @test all(raw(parent(getregion(default_call, p).data)) == raw(parent(getregion(stock, p).data)) for p in 1:6)

    for w in 1:H
        c = coded_scalar(grid)
        fill_halo_regions!(c; halo_width = w)
        strip = Set(edge_strip(w))
        for p in 1:6
            data = getregion(c, p).data
            reference = getregion(stock, p).data
            @test all(raw(data[i, j, k]) == raw(reference[i, j, k]) for (i, j) in strip, k in 1:Nz)
            others = [(i, j) for j in axes(data, 2), i in axes(data, 1) if !((i, j) in strip)]
            @test all(raw(data[i, j, k]) == raw(code_or_poison(p, i, j, k)) for (i, j) in others, k in 1:Nz)
        end
    end
end


@testset "Partial-width scalar fill: vertical behaviour follows the vertical boundary conditions" begin
    grid = cubed_sphere()
    bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()); top = nothing, bottom = nothing)
    c = coded_scalar(grid; boundary_conditions = bcs)
    fill_halo_regions!(c; halo_width = 3)
    @test all(raw(getregion(c, p).data[i, j, k]) == raw(poison(p, i, j, k)) for p in 1:6, k in (0, Nz + 1), j in 1:N, i in 1:N)

    d = coded_scalar(grid)
    fill_halo_regions!(d; halo_width = 3)
    stock = coded_scalar(grid)
    fill_halo_regions!(stock)
    @test all(raw(getregion(d, p).data[i, j, k]) == raw(getregion(stock, p).data[i, j, k]) for p in 1:6, k in (0, Nz + 1), j in 1:N, i in 1:N)
end

@testset "Partial-width scalar fill: fail-closed preflight" begin
    grid = cubed_sphere()
    c = coded_scalar(grid)
    for w in (0, H + 1, 3.0)
        @test rejects_without_writing(() -> fill_halo_regions!(c; halo_width = w), c)
    end

    subdivided = coded_scalar(subdivided_cubed_sphere(2))
    @test rejects_without_writing(() -> fill_halo_regions!(subdivided; halo_width = 1), subdivided)

    windowed = CenterField(grid; indices = (:, :, 1:1))
    @test rejects_without_writing(() -> fill_halo_regions!(windowed; halo_width = 1), windowed)

    reduced = Field{Center, Center, Nothing}(grid)
    @test rejects_without_writing(() -> fill_halo_regions!(reduced; halo_width = 1), reduced)

    # Cross-region alias: region 4 shares region 1's storage.
    template = CenterField(grid)
    regions = collect(template.data.regional_objects)
    regions[4] = regions[1]
    aliased = Field((Center(), Center(), Center()), grid; data = MultiRegionObject(Tuple(regions)))
    @test rejects_without_writing(() -> fill_halo_regions!(aliased; halo_width = 2), aliased)

    # Region 3 storage has the right size but its vertical axis does not cover the common active range 1:Nz.
    regions = collect(CenterField(grid).data.regional_objects)
    full = regions[3]
    regions[3] = OffsetArray(zeros(size(full)), axes(full, 1), axes(full, 2), Nz+1:Nz+size(full, 3))
    short = Field((Center(), Center(), Center()), grid; data = MultiRegionObject(Tuple(regions)))
    @test rejects_without_writing(() -> fill_halo_regions!(short; halo_width = 2), short)
end
