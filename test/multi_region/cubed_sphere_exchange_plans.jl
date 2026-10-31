include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Random
using Oceananigans.MultiRegion: getregion, synchronize_shared_edges!, fill_cgrid_pair_halo_regions!, fill_dgrid_pair_halo_regions!,
                                CGridPairExchange, DGridPairExchange, SharedEdgeExchange, CenterScalarExchange
using Oceananigans.BoundaryConditions: fill_halo_regions!

# Run with --check-bounds=yes: the halo kernels use @inbounds, so only forced bounds checking catches storage overruns.
# Every plan must reproduce the function it materialises bit for bit over the whole storage of every region (halos and
# corner blocks included: the plan may write exactly what the function writes and nothing else), and a warm call must
# not allocate.

bits(f) = [reinterpret(UInt64, parent(getregion(f, r).data)) for r in 1:6]

function randomize!(fields...)
    for f in fields, r in 1:6
        rand!(parent(getregion(f, r).data))
    end
    return nothing
end

copy_storage!(dst, src) = (for r in 1:6; copyto!(parent(getregion(dst, r).data), parent(getregion(src, r).data)); end; nothing)

allocations(plan) = @allocated fill_halo_regions!(plan)

for (N, Nz) in ((8, 3), (12, 4))
    grid = ConformalCubedSphereGrid(; panel_size = (N, N, Nz), z = (0, 1), horizontal_direction_halo = 4)
    Random.seed!(1000N + Nz)

    @testset "C-grid pair plan vs fill_cgrid_pair_halo_regions! (N = $N)" begin
        for w in 1:3, signed in (true, false)
            uc = Field{Face, Center, Center}(grid); vc = Field{Center, Face, Center}(grid)
            ucr = Field{Face, Center, Center}(grid); vcr = Field{Center, Face, Center}(grid)
            randomize!(uc, vc); copy_storage!(ucr, uc); copy_storage!(vcr, vc)
            fill_cgrid_pair_halo_regions!(ucr, vcr; halo_width = w, signed)
            plan = CGridPairExchange(uc, vc; halo_width = w, signed)
            fill_halo_regions!(plan)
            @test bits(uc) == bits(ucr) && bits(vc) == bits(vcr)
            fill_halo_regions!(plan)
            @test allocations(plan) == 0
            @test bits(uc) == bits(ucr) && bits(vc) == bits(vcr)
        end
    end

    @testset "D-grid pair plan and shared-edge plan (N = $N)" begin
        for w in 1:3, signed in (true, false)
            u = Field{Center, Face, Center}(grid); v = Field{Face, Center, Center}(grid)
            ur = Field{Center, Face, Center}(grid); vr = Field{Face, Center, Center}(grid)
            randomize!(u, v); copy_storage!(ur, u); copy_storage!(vr, v)
            fill_dgrid_pair_halo_regions!(ur, vr; halo_width = w, signed)
            plan = DGridPairExchange(u, v; halo_width = w, signed)
            fill_halo_regions!(plan)
            @test bits(u) == bits(ur) && bits(v) == bits(vr)
            @test allocations(plan) == 0
            synchronize_shared_edges!(ur, vr; signed)
            sync = SharedEdgeExchange(u, v; signed)
            fill_halo_regions!(sync)
            @test bits(u) == bits(ur) && bits(v) == bits(vr)
            @test allocations(sync) == 0
        end
        # two pairs with different vertical locations in one plan, as fill_dgrid_pair_halo_regions! allows
        u₁ = Field{Center, Face, Center}(grid); v₁ = Field{Face, Center, Center}(grid)
        u₂ = Field{Center, Face, Face}(grid); v₂ = Field{Face, Center, Face}(grid)
        a₁ = Field{Center, Face, Center}(grid); b₁ = Field{Face, Center, Center}(grid)
        a₂ = Field{Center, Face, Face}(grid); b₂ = Field{Face, Center, Face}(grid)
        randomize!(u₁, v₁, u₂, v₂); copy_storage!(a₁, u₁); copy_storage!(b₁, v₁); copy_storage!(a₂, u₂); copy_storage!(b₂, v₂)
        fill_dgrid_pair_halo_regions!(((a₁, b₁), (a₂, b₂)); halo_width = 3)
        fill_halo_regions!(DGridPairExchange(((u₁, v₁), (u₂, v₂)); halo_width = 3))
        @test all(bits(x) == bits(y) for (x, y) in ((u₁, a₁), (v₁, b₁), (u₂, a₂), (v₂, b₂)))
    end

    # the plan fills horizontal halos only, so the reference fields carry no vertical boundary conditions (as the FV3 dynamics
    # Fields do); the default FluxBoundaryCondition(nothing) would make fill_halo_regions! write one vertical halo level too
    no_vertical_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()); top = nothing, bottom = nothing)
    scalar() = CenterField(grid; boundary_conditions = no_vertical_bcs)

    @testset "centre-scalar plan vs fill_halo_regions!(q; halo_width) (N = $N)" begin
        for w in 1:4
            q = scalar(); p = scalar()
            qr = scalar(); pr = scalar()
            randomize!(q, p); copy_storage!(qr, q); copy_storage!(pr, p)
            fill_halo_regions!(qr; halo_width = w); fill_halo_regions!(pr; halo_width = w)
            plan = CenterScalarExchange((q, p); halo_width = w)
            fill_halo_regions!(plan)
            @test bits(q) == bits(qr) && bits(p) == bits(pr)
            @test allocations(plan) == 0
        end
        # the default width is the full halo, as fill_halo_regions!(q) without halo_width
        q = scalar(); qr = scalar()
        randomize!(q); copy_storage!(qr, q)
        fill_halo_regions!(qr)
        fill_halo_regions!(CenterScalarExchange(q))
        @test bits(q) == bits(qr)
        # a field with vertical boundary conditions, the default included, is refused (the plan fills horizontal halos only)
        @test_throws ArgumentError CenterScalarExchange(CenterField(grid))
        c = CenterField(grid; boundary_conditions = FieldBoundaryConditions(grid, (Center(), Center(), Center()); top = ValueBoundaryCondition(0.0)))
        @test_throws ArgumentError CenterScalarExchange(c)
    end
end
