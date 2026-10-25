include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Random
using Oceananigans.Grids: Sides, Connected
using Oceananigans.MultiRegion: CubedSpherePartition, CubedSphereConnectivity, MultiRegionGrid, MultiRegionObject,
                                getregion, synchronize_shared_edges!, fill_cgrid_pair_halo_regions!,
                                fill_dgrid_pair_halo_regions!, panel_side_map, dgrid_neighbor_value, cgrid_neighbor_value
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils: apply_regionally!

# Run with --check-bounds=yes: the halo kernels use @inbounds, so only forced bounds checking catches storage overruns.

const sides = (:west, :east, :south, :north)
neighbor_of(region, side) = getproperty(getregion(CubedSphereConnectivity(CubedSpherePartition(R = 1)).connections, region), side).from_rank

# The donor (component, i, j) a halo point reads, by the same doubled-coordinate rule as the kernels.
function donor(map, X, Y; cgrid)
    Xn = map.R₁₁ * X + map.R₁₂ * Y + map.Tx
    Yn = map.R₂₁ * X + map.R₂₂ * Y + map.Ty
    if cgrid
        return iseven(Xn) ? (1, Xn ÷ 2 + 1, (Yn + 1) ÷ 2) : (2, (Xn + 1) ÷ 2, Yn ÷ 2 + 1)
    else
        return isodd(Xn) ? (1, (Xn + 1) ÷ 2, Yn ÷ 2 + 1) : (2, Xn ÷ 2 + 1, (Yn + 1) ÷ 2)
    end
end

# Halo points written by side `side` at depth d: ((component, i, j, X, Y), ...) for D-grid (u, v) or C-grid (uc, vc).
function written_points(side, d, N; cgrid)
    points = Tuple{Int, Int, Int, Int, Int}[]
    for t in 1:N+1
        if cgrid
            (iu, ju, ku), (iv, jv, kv) = side === :west  ? ((1 - d, t, t <= N), (1 - d, t, t <= N + 1)) :
                                         side === :east  ? ((N + 1 + d, t, t <= N), (N + d, t, t <= N + 1)) :
                                         side === :south ? ((t, 1 - d, t <= N + 1), (t, 1 - d, t <= N)) :
                                                           ((t, N + d, t <= N + 1), (t, N + 1 + d, t <= N))
            ku && push!(points, (1, iu, ju, 2(iu - 1), 2ju - 1))
            kv && push!(points, (2, iv, jv, 2iv - 1, 2(jv - 1)))
        else
            (iu, ju, ku), (iv, jv, kv) = side === :west  ? ((1 - d, t, t <= N + 1), (1 - d, t, t <= N)) :
                                         side === :east  ? ((N + d, t, t <= N + 1), (N + 1 + d, t, t <= N)) :
                                         side === :south ? ((t, 1 - d, t <= N), (t, 1 - d, t <= N + 1)) :
                                                           ((t, N + 1 + d, t <= N), (t, N + d, t <= N + 1))
            ku && push!(points, (1, iu, ju, 2iu - 1, 2(ju - 1)))
            kv && push!(points, (2, iv, jv, 2(iv - 1), 2jv - 1))
        end
    end
    return points
end

# Owned (never written) storage: interiors plus the duplicated shared edges.
owned_dgrid(c, i, j, N) = c == 1 ? (1 <= i <= N && 1 <= j <= N + 1) : (1 <= i <= N + 1 && 1 <= j <= N)
owned_cgrid(c, i, j, N) = c == 1 ? (1 <= i <= N + 1 && 1 <= j <= N) : (1 <= i <= N && 1 <= j <= N + 1)

@testset "Pair exchanges read only owned points and write only halo points" begin
    for N in (4, 12), w in 1:3, cgrid in (false, true)
        owned = cgrid ? owned_cgrid : owned_dgrid
        reads_owned = writes_halo = true
        for region in 1:6, side in sides, d in 1:w
            map = panel_side_map(isodd(region), side, N)
            for (c, i, j, X, Y) in written_points(side, d, N; cgrid)
                writes_halo &= !owned(c, i, j, N)
                cn, iN, jN = donor(map, X, Y; cgrid)
                reads_owned &= owned(cn, iN, jN, N)
            end
        end
        @test reads_owned
        @test writes_halo
    end

    # Synchronization writes u[t, N+1] and v[N+1, t]; the donors must lie elsewhere (the neighbour's first row/column).
    for N in (4, 12)
        disjoint = true
        for region in 1:6, t in 1:N
            cu, iu, ju = donor(panel_side_map(isodd(region), :north, N), 2t - 1, 2N; cgrid = false)
            cv, iv, jv = donor(panel_side_map(isodd(region), :east, N), 2N, 2t - 1; cgrid = false)
            for (c, i, j) in ((cu, iu, ju), (cv, iv, jv))
                disjoint &= owned_dgrid(c, i, j, N) && !(c == 1 && j == N + 1) && !(c == 2 && i == N + 1)
            end
        end
        @test disjoint
    end
end

N, Nz = 6, 3
grid = ConformalCubedSphereGrid(; panel_size = (N, N, Nz), z = (0, 1), horizontal_direction_halo = 4)

const sentinel = reinterpret(Float64, 0x7ff8_dead_beef_0001)

function seeded!(field, seed)
    rng = Xoshiro(seed)
    for r in 1:6
        data = getregion(field, r).data
        fill!(parent(data), sentinel)
        for k in 1:size(getregion(field, r), 3), j in 1:N+1, i in 1:N+1
            data[i, j, k] = randn(rng)
        end
    end
    return field
end

bits(field) = [reinterpret(UInt64, copy(parent(getregion(field, r).data))) for r in 1:6]

# The pre-batching algorithm: one region and one side at a time, synchronization from copies.
function sequential_fill!(x, y, w, signed; cgrid)
    xs, ys = x.data.regional_objects, y.data.regional_objects
    for region in 1:6, side in sides
        map = panel_side_map(isodd(region), side, N)
        n = neighbor_of(region, side)
        for k in 1:size(getregion(x, 1), 3), d in 1:w
            for (c, i, j, X, Y) in written_points(side, d, N; cgrid)
                target = c == 1 ? xs[region] : ys[region]
                target[i, j, k] = cgrid ? cgrid_neighbor_value(xs[n], ys[n], map, c, X, Y, k, signed) :
                                          dgrid_neighbor_value(xs[n], ys[n], map, c, X, Y, k, signed)
            end
        end
    end
    return nothing
end

function sequential_sync!(u, v, signed)
    us, vs = map(copy, u.data.regional_objects), map(copy, v.data.regional_objects)
    for region in 1:6, k in 1:size(getregion(u, 1), 3), t in 1:N
        north, east = neighbor_of(region, :north), neighbor_of(region, :east)
        u.data.regional_objects[region][t, N + 1, k] =
            dgrid_neighbor_value(us[north], vs[north], panel_side_map(isodd(region), :north, N), 1, 2t - 1, 2N, k, signed)
        v.data.regional_objects[region][N + 1, t, k] =
            dgrid_neighbor_value(us[east], vs[east], panel_side_map(isodd(region), :east, N), 2, 2N, 2t - 1, k, signed)
    end
    return nothing
end

@testset "Batched pair exchanges are bit-identical to the sequential algorithm" begin
    for LZ in (Center, Face), signed in (true, false)
        u, v = Field{Center, Face, LZ}(grid), Field{Face, Center, LZ}(grid)
        ur, vr = Field{Center, Face, LZ}(grid), Field{Face, Center, LZ}(grid)
        for w in 1:3
            seeded!(u, 1); seeded!(v, 2); seeded!(ur, 1); seeded!(vr, 2)
            fill_halo_regions!(u, v; halo_width = w, signed)
            sequential_fill!(ur, vr, w, signed; cgrid = false)
            @test bits(u) == bits(ur) && bits(v) == bits(vr)
            # negative control: the opposite sign convention must be detected
            seeded!(ur, 1); seeded!(vr, 2)
            sequential_fill!(ur, vr, w, !signed; cgrid = false)
            @test bits(u) != bits(ur) || bits(v) != bits(vr)
        end
        seeded!(u, 3); seeded!(v, 4); seeded!(ur, 3); seeded!(vr, 4)
        synchronize_shared_edges!(u, v; signed)
        sequential_sync!(ur, vr, signed)
        @test bits(u) == bits(ur) && bits(v) == bits(vr)

        uc, vc = Field{Face, Center, LZ}(grid), Field{Center, Face, LZ}(grid)
        ucr, vcr = Field{Face, Center, LZ}(grid), Field{Center, Face, LZ}(grid)
        for w in 1:3
            seeded!(uc, 5); seeded!(vc, 6); seeded!(ucr, 5); seeded!(vcr, 6)
            fill_cgrid_pair_halo_regions!(uc, vc; halo_width = w, signed)
            sequential_fill!(ucr, vcr, w, signed; cgrid = true)
            @test bits(uc) == bits(ucr) && bits(vc) == bits(vcr)
        end
    end
end

@testset "Tuples of pairs equal one call per pair" begin
    u₁, v₁ = Field{Center, Face, Center}(grid), Field{Face, Center, Center}(grid)
    u₂, v₂ = Field{Center, Face, Face}(grid), Field{Face, Center, Face}(grid)
    a₁, b₁ = Field{Center, Face, Center}(grid), Field{Face, Center, Center}(grid)
    a₂, b₂ = Field{Center, Face, Face}(grid), Field{Face, Center, Face}(grid)
    for (seed, f) in enumerate((u₁, v₁, u₂, v₂)); seeded!(f, seed); end
    for (seed, f) in enumerate((a₁, b₁, a₂, b₂)); seeded!(f, seed); end
    fill_dgrid_pair_halo_regions!(((u₁, v₁), (u₂, v₂)); halo_width = 2, signed = false)
    fill_halo_regions!(a₁, b₁; halo_width = 2, signed = false)
    fill_halo_regions!(a₂, b₂; halo_width = 2, signed = false)
    @test all(bits(x) == bits(y) for (x, y) in ((u₁, a₁), (v₁, b₁), (u₂, a₂), (v₂, b₂)))
    synchronize_shared_edges!(((u₁, v₁), (u₂, v₂)))
    synchronize_shared_edges!(a₁, b₁)
    synchronize_shared_edges!(a₂, b₂)
    @test all(bits(x) == bits(y) for (x, y) in ((u₁, a₁), (v₁, b₁), (u₂, a₂), (v₂, b₂)))

    uc₁, vc₁ = Field{Face, Center, Center}(grid), Field{Center, Face, Center}(grid)
    uc₂, vc₂ = Field{Face, Center, Face}(grid), Field{Center, Face, Face}(grid)
    c₁, d₁ = Field{Face, Center, Center}(grid), Field{Center, Face, Center}(grid)
    c₂, d₂ = Field{Face, Center, Face}(grid), Field{Center, Face, Face}(grid)
    for (seed, f) in enumerate((uc₁, vc₁, uc₂, vc₂)); seeded!(f, seed); end
    for (seed, f) in enumerate((c₁, d₁, c₂, d₂)); seeded!(f, seed); end
    fill_cgrid_pair_halo_regions!(((uc₁, vc₁), (uc₂, vc₂)); halo_width = 3)
    fill_cgrid_pair_halo_regions!(c₁, d₁; halo_width = 3)
    fill_cgrid_pair_halo_regions!(c₂, d₂; halo_width = 3)
    @test all(bits(x) == bits(y) for (x, y) in ((uc₁, c₁), (vc₁, d₁), (uc₂, c₂), (vc₂, d₂)))
end

@testset "Batched pair exchanges reject shared storage and mixed grids before writing" begin
    u, v = seeded!(Field{Center, Face, Center}(grid), 1), seeded!(Field{Face, Center, Center}(grid), 2)
    uc, vc = seeded!(Field{Face, Center, Center}(grid), 3), seeded!(Field{Center, Face, Center}(grid), 4)
    other = ConformalCubedSphereGrid(; panel_size = (N, N, Nz), z = (0, 1), horizontal_direction_halo = 4)
    u′, v′ = Field{Center, Face, Center}(other), Field{Face, Center, Center}(other)
    before = map(bits, (u, v, uc, vc))
    @test_throws ArgumentError fill_dgrid_pair_halo_regions!(((u, v), (u, v)))
    @test_throws ArgumentError synchronize_shared_edges!(((u, v), (u, v)))
    @test_throws ArgumentError fill_cgrid_pair_halo_regions!(((uc, vc), (uc, vc)))
    @test_throws ArgumentError fill_dgrid_pair_halo_regions!(((u, v), (u′, v′)))
    @test_throws ArgumentError fill_dgrid_pair_halo_regions!(((u, v, u),))
    @test_throws ArgumentError fill_dgrid_pair_halo_regions!(())
    @test map(bits, (u, v, uc, vc)) == before
end

struct Accumulator
    total :: Base.RefValue{Float64}
end

accumulate!(a::Accumulator, x) = (a.total[] += x; nothing)
apply_and_measure(a, regional) = @allocated apply_regionally!(accumulate!, a, regional)

accumulate_panel!(a::Accumulator, panel) = (a.total[] += size(panel, 1); nothing)
apply_to_panels_and_measure(a, grid) = @allocated apply_regionally!(accumulate_panel!, a, grid)

@testset "apply_regionally! over heterogeneous regions is type-stable" begin
    regional = MultiRegionObject((1, 2.0, Int32(3), 4.0f0, 5, 6.0))
    a = Accumulator(Ref(0.0))
    apply_and_measure(a, regional)
    @test apply_and_measure(a, regional) == 0
    @test a.total[] == 2 * 21

    # conformal cubed-sphere panels have different rotation types
    @test !allequal(map(typeof, grid.region_grids.regional_objects))
    p = Accumulator(Ref(0.0))
    apply_to_panels_and_measure(p, grid)
    @test apply_to_panels_and_measure(p, grid) == 0
    @test p.total[] == 2 * 6N
end
