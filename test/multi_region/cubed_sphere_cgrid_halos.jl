include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: FullyConnected
using Oceananigans.MultiRegion: CubedSpherePartition, CubedSphereConnectivity, MultiRegionGrid, MultiRegionObject,
                                getregion, fill_cgrid_pair_halo_regions!
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

# Injective codes: field 1 = uc, 2 = vc; panel p; storage indices offset so all are positive.
code(field, p, i, j, k) = 1e6 * field + 1e5 * p + 1e3 * (i + 10) + 10 * (j + 10) + k
poison(field, p, i, j, k) = reinterpret(Float64, 0x7ff8000000000000 | UInt64(round(Int, code(field, p, i, j, k))))

# Every owned value (uc i ∈ 1:N+1, j ∈ 1:N; vc i ∈ 1:N, j ∈ 1:N+1) gets a code; everything else gets a distinct NaN.
function coded_cgrid_pair(grid; LZ = Center)
    uc = Field{Face, Center, LZ}(grid)
    vc = Field{Center, Face, LZ}(grid)
    for p in 1:length(grid.partition), (field, f, owned_i, owned_j) in ((1, uc, 1:N+1, 1:N), (2, vc, 1:N, 1:N+1))
        data = getregion(f, p).data
        for k in axes(data, 3), j in axes(data, 2), i in axes(data, 1)
            data[i, j, k] = (i in owned_i && j in owned_j && 1 <= k <= size(getregion(f, p), 3)) ? code(field, p, i, j, k) :
                                                                                                   poison(field, p, i, j, k)
        end
    end
    return uc, vc
end

wrap(p) = mod1(p, 6)
neighbor(p, side) = isodd(p) ? wrap(p + Dict(:east => 1, :north => 2, :west => -2, :south => -1)[side]) :
                               wrap(p + Dict(:east => 2, :north => 1, :west => -1, :south => -2)[side])

# The §5A table of F2-STEP1-CONTRACTS (field, donor i, donor j, sign) for a written point at depth d.
function expected_donor(field, p, side, t, d)
    odd = isodd(p)
    if field == 1
        side === :east  && return odd ? (1, 1 + d, t, 1)         : (2, N - t + 1, 1 + d, 1)
        side === :west  && return odd ? (2, N - t + 1, N - d + 1, 1) : (1, N - d + 1, t, 1)
        side === :north && return odd ? (2, d, N - t + 2, -1)     : (1, t, d, 1)
        side === :south && return odd ? (1, t, N - d + 1, 1)      : (2, N - d + 1, N - t + 2, -1)
    else
        side === :east  && return odd ? (2, d, t, 1)              : (1, N - t + 2, d, -1)
        side === :west  && return odd ? (1, N - t + 2, N - d + 1, -1) : (2, N - d + 1, t, 1)
        side === :north && return odd ? (1, 1 + d, N - t + 1, 1)  : (2, t, 1 + d, 1)
        side === :south && return odd ? (2, t, N - d + 1, 1)      : (1, N - d + 1, N - t + 1, 1)
    end
end

function written_point(field, side, t, d)
    if field == 1
        side === :east  && return (N + 1 + d, t), 1:N
        side === :west  && return (1 - d, t), 1:N
        side === :north && return (t, N + d), 1:N+1
        side === :south && return (t, 1 - d), 1:N+1
    else
        side === :east  && return (N + d, t), 1:N+1
        side === :west  && return (1 - d, t), 1:N+1
        side === :north && return (t, N + 1 + d), 1:N
        side === :south && return (t, 1 - d), 1:N
    end
end

function expected_after_fill(uc, vc, w; signed = true)
    before = ([copy(parent(getregion(uc, p).data)) for p in 1:6], [copy(parent(getregion(vc, p).data)) for p in 1:6])
    expected = deepcopy(before)
    offset = 1 - first(axes(getregion(uc, 1).data, 1))
    koffset = 1 - first(axes(getregion(uc, 1).data, 3))
    for p in 1:6, field in 1:2, side in (:east, :west, :north, :south), d in 1:w, t in 1:N+1, k in 1:Nz
        (i, j), trange = written_point(field, side, t, d)
        t in trange || continue
        donor_field, di, dj, s = expected_donor(field, p, side, t, d)
        value = before[donor_field][neighbor(p, side)][di + offset, dj + offset, k + koffset]
        expected[field][p][i + offset, j + offset, k + koffset] = (signed ? s : abs(s)) * value
    end
    return expected
end

raw(a) = reinterpret(UInt64, a)
matches(uc, vc, expected) = all(raw(parent(getregion(uc, p).data)) == raw(expected[1][p]) &&
                                raw(parent(getregion(vc, p).data)) == raw(expected[2][p]) for p in 1:6)

snapshot(f) = [copy(parent(getregion(f, r).data)) for r in 1:length(f.grid.partition)]
unchanged(f, before) = all(raw(parent(getregion(f, r).data)) == raw(before[r]) for r in 1:length(f.grid.partition))

function rejects_without_writing(call, uc, vc)
    su, sv = snapshot(uc), snapshot(vc)
    threw = try
        call()
        false
    catch e
        e isa ArgumentError || rethrow()
        true
    end
    return threw && unchanged(uc, su) && unchanged(vc, sv)
end

@testset "FMS C-grid pair exchange: coded donors, owned duplicates, poison" begin
    grid = cubed_sphere()
    for w in 1:3
        uc, vc = coded_cgrid_pair(grid)
        expected = expected_after_fill(uc, vc, w)
        fill_cgrid_pair_halo_regions!(uc, vc; halo_width = w)
        @test matches(uc, vc, expected)
    end

    uc, vc = coded_cgrid_pair(grid)
    expected = expected_after_fill(uc, vc, 3)
    fill_cgrid_pair_halo_regions!(uc, vc)
    @test matches(uc, vc, expected)
    for p in 1:6
        u = getregion(uc, p).data; v = getregion(vc, p).data
        @test all(u[i, j, k] === code(1, p, i, j, k) for k in 1:Nz, j in 1:N, i in (1, N + 1))
        @test all(v[i, j, k] === code(2, p, i, j, k) for k in 1:Nz, j in (1, N + 1), i in 1:N)
        @test all(raw(u[i, j, k]) == raw(poison(1, p, i, j, k)) for k in 1:Nz, j in (-2:0..., N+1:N+3...), i in (-2:0..., N+2:N+4...))
        @test all(raw(v[i, j, k]) == raw(poison(2, p, i, j, k)) for k in 1:Nz, j in (-2:0..., N+2:N+4...), i in (-2:0..., N+1:N+3...))
        @test all(raw(u[i, j, k]) == raw(poison(1, p, i, j, k)) for k in 1:Nz, j in 1:N, i in (-3,))
        @test all(raw(u[i, j, k]) == raw(poison(1, p, i, j, k)) for k in (0, Nz + 1), j in 1:N, i in 1:N+1)
    end
    # Corner-line points written once from the source donors (contracts §5A).
    for p in 1:6
        u = getregion(uc, p).data; v = getregion(vc, p).data
        @test u[N + 1, N + 1, 1] === (isodd(p) ? -code(2, wrap(p + 2), 1, 1, 1) : code(1, wrap(p + 1), N + 1, 1, 1))
        @test v[N + 1, N + 1, 1] === (isodd(p) ? code(2, wrap(p + 1), 1, N + 1, 1) : -code(1, wrap(p + 2), 1, 1, 1))
    end
end

@testset "FMS C-grid pair exchange: negative controls" begin
    grid = cubed_sphere()
    uc, vc = coded_cgrid_pair(grid)
    expected = expected_after_fill(uc, vc, 3)
    fill_halo_regions!(uc, vc; signed = true)
    @test !matches(uc, vc, expected)

    uc, vc = coded_cgrid_pair(grid)
    fill_cgrid_pair_halo_regions!(uc, vc; halo_width = 3, signed = false)
    @test !matches(uc, vc, expected)
    @test matches(uc, vc, expected_after_fill(coded_cgrid_pair(grid)..., 3; signed = false))

    uc, vc = coded_cgrid_pair(grid)
    fill_cgrid_pair_halo_regions!(uc, vc; halo_width = 2)
    @test !matches(uc, vc, expected)
end

@testset "FMS C-grid pair exchange: fail-closed preflight" begin
    grid = cubed_sphere()
    uc, vc = coded_cgrid_pair(grid)
    for w in (0, 4, 3.0)
        @test rejects_without_writing(() -> fill_cgrid_pair_halo_regions!(uc, vc; halo_width = w), uc, vc)
    end
    narrow = cubed_sphere(; H = 3)
    un, vn = coded_cgrid_pair(narrow)
    @test rejects_without_writing(() -> fill_cgrid_pair_halo_regions!(un, vn; halo_width = 3), un, vn)

    ur, vr = coded_cgrid_pair(subdivided_cubed_sphere(2))
    @test rejects_without_writing(() -> fill_cgrid_pair_halo_regions!(ur, vr; halo_width = 1), ur, vr)

    uo, vo = coded_cgrid_pair(cubed_sphere())
    @test rejects_without_writing(() -> fill_cgrid_pair_halo_regions!(uc, vo), uc, vo)

    uf, vf = coded_cgrid_pair(grid; LZ = Face)
    @test rejects_without_writing(() -> fill_cgrid_pair_halo_regions!(uc, vf), uc, vf)

    windowed = Field{Center, Face, Center}(grid; indices = (:, :, 1:1))
    @test rejects_without_writing(() -> fill_cgrid_pair_halo_regions!(uc, windowed), uc, windowed)

    reduced = Field{Center, Face, Nothing}(grid)
    @test rejects_without_writing(() -> fill_cgrid_pair_halo_regions!(uc, reduced), uc, reduced)

    # Cross-region, cross-component alias: region 3 of vc shares region 2 of uc's storage.
    ua, va = coded_cgrid_pair(grid)
    aliased_vc = Field{Center, Face, Center}(grid)
    regions = collect(aliased_vc.data.regional_objects)
    regions[3] = ua.data.regional_objects[2]
    alias_field = Field((Center(), Face(), Center()), grid; data = MultiRegionObject(Tuple(regions)))
    @test rejects_without_writing(() -> fill_cgrid_pair_halo_regions!(ua, alias_field), ua, alias_field)
end
