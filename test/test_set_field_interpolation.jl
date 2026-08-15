using Oceananigans.Fields: interpolate!, fill_halo_regions!
using Oceananigans.BoundaryConditions: needs_simulation_context, normal_flow_needs_simulation_context, NormalRadiation

@testset "needs_simulation_context dispatch" begin
    # Flux with CBF → true (fill_halo_regions! would crash without clock)
    cbf_flux_bc = FluxBoundaryCondition((x, y, t) -> 0.0)
    @test  needs_simulation_context(cbf_flux_bc)

    # NormalFlow BC backed by a CBF → false (NormalFlow fills use fill_normal_flow_bcs=false, not this guard)
    nf_cbf_bc = NormalFlowBoundaryCondition((x, y, t) -> 0.0)
    @test !needs_simulation_context(nf_cbf_bc)

    # NormalFlow BC with a plain number → false
    @test !needs_simulation_context(NormalFlowBoundaryCondition(0.0))

    # FieldBoundaryConditions: CBF top BC → true
    @test  needs_simulation_context(FieldBoundaryConditions(top=cbf_flux_bc))

    # FieldBoundaryConditions: only NormalFlow BCs → false
    @test !needs_simulation_context(FieldBoundaryConditions(west=nf_cbf_bc, east=nf_cbf_bc))
end

@testset "RNFBC: normal_flow_needs_simulation_context and clockless fill" begin
    # NormalFlowBoundaryCondition with NormalRadiation scheme wraps a DiscreteBoundaryFunction.
    # needs_simulation_context must return false (NFBC catch-all) but
    # normal_flow_needs_simulation_context must return true (condition is a DBF).
    rnfbc = NormalFlowBoundaryCondition((i, k, grid, clock, fields) -> zero(grid);
                                        discrete_form = true,
                                        scheme = NormalRadiation())
    @test !needs_simulation_context(rnfbc)
    @test  normal_flow_needs_simulation_context(rnfbc)

    # fill_halo_regions!(v) without clock must not crash for a field carrying RNFBC
    # (previously triggered InvalidIRError on GPU at first kernel compilation).
    # Periodic in x/z so only the y (normal) boundaries need explicit BCs.
    grid = RectilinearGrid(CPU(); size=(4, 4, 4), x=(0,1), y=(0,1), z=(0,1),
                           topology=(Periodic, Bounded, Periodic))
    south_bc = NormalFlowBoundaryCondition(0.0)
    v_bcs = FieldBoundaryConditions(grid, (Center(), Face(), Center()); south=south_bc, north=rnfbc)
    v = YFaceField(grid; boundary_conditions=v_bcs)
    @test_nowarn fill_halo_regions!(v)

    # set! on a field with RNFBC must also not crash (uses fill_normal_flow_bcs=false path).
    @test_nowarn set!(v, 0)
end

@testset "set! field interpolation" begin
    for arch in archs, FT in float_types
        interp_domain = (; x=(0, 1), y=(0, 1), z=(0, 1))
        f_linear = (x, y, z) -> x + 2y + 3z

        coarse_grid = RectilinearGrid(arch, FT; size=(4, 4, 4), interp_domain...)
        fine_grid   = RectilinearGrid(arch, FT; size=(8, 8, 8), interp_domain...)

        coarse = CenterField(coarse_grid)
        stale_halo_value = FT(-9999)
        fill!(parent(coarse), stale_halo_value)
        set!(coarse, f_linear)

        coarse_with_filled_halos = CenterField(coarse_grid)
        fill!(parent(coarse_with_filled_halos), stale_halo_value)
        set!(coarse_with_filled_halos, f_linear)
        fill_halo_regions!(coarse_with_filled_halos)

        fine = CenterField(fine_grid)
        set!(fine, coarse)

        expected_fine = CenterField(fine_grid)
        interpolate!(expected_fine, coarse_with_filled_halos)
        @test Array(interior(fine)) == Array(interior(expected_fine))

        # Different halo size on same grid → copy path, not interpolation.
        big_halo_grid = RectilinearGrid(arch, FT; size=(4, 4, 4),
                                        halo=(3, 3, 3),
                                        interp_domain...)

        big_halo_c = CenterField(big_halo_grid)
        set!(big_halo_c, coarse)
        @test Array(interior(big_halo_c)) == Array(interior(coarse))

        # `set!` must not crash when `to_field` carries a `ContinuousBoundaryFunction` BC.
        # All sides specified explicitly to avoid unresolved DefaultBoundaryCondition types.
        bounded_fine_grid = RectilinearGrid(arch, FT; size=(8, 8, 8), interp_domain...,
                                            topology=(Bounded, Bounded, Bounded))
        cbf_bc = FluxBoundaryCondition((x, y, t) -> zero(FT))
        nf_bc  = NoFluxBoundaryCondition()
        explicit_bcs = FieldBoundaryConditions(west=nf_bc, east=nf_bc, south=nf_bc, north=nf_bc,
                                               bottom=nf_bc, top=cbf_bc)
        cbf_field = CenterField(bounded_fine_grid, boundary_conditions=explicit_bcs)
        set!(cbf_field, coarse)
        @test Array(interior(cbf_field)) ≈ Array(interior(fine))
    end
end

@testset "set! between reduced and windowed single-layer fields" begin
    for arch in archs, FT in float_types
        grid = RectilinearGrid(arch, FT; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1))
        Nz = size(grid, 3)

        # A windowed single-layer field at the surface, as produced by output with
        # `indices = (:, :, Nz)` and reloaded as a `FieldTimeSeries` slice. Its z-location
        # is `Center`, but it spans a single vertical level.
        surface_u = Field{Face, Center, Center}(grid, indices=(:, :, Nz:Nz))
        set!(surface_u, (x, y, z) -> x + 2y)

        # Setting a reduced (`Nothing`-z) field from the single-layer 3D field must copy
        # the single slab directly, not attempt to interpolate across the `Nothing`/`Center`
        # location and `:`/`Nz:Nz` index mismatch in the degenerate vertical dimension.
        reduced_u = Field{Face, Center, Nothing}(grid)
        set!(reduced_u, surface_u)
        @test Array(interior(reduced_u)) == Array(interior(surface_u))

        # The reverse direction (located single-layer field from a reduced field) too.
        surface_back = Field{Face, Center, Center}(grid, indices=(:, :, Nz:Nz))
        set!(surface_back, reduced_u)
        @test Array(interior(surface_back)) == Array(interior(reduced_u))

        # Two *located* single-layer fields windowed at different levels do NOT share a
        # discretization (the index is meaningful), so this still routes to interpolation.
        if Nz > 1
            bottom_u = Field{Face, Center, Center}(grid, indices=(:, :, 1:1))
            top_u    = Field{Face, Center, Center}(grid, indices=(:, :, Nz:Nz))
            @test !Oceananigans.Fields.matching_field_discretization(bottom_u, top_u)
        end
    end
end

@testset "node returns the physical (deformed) znode on mutable grids" begin
    # On a mutable vertical grid the physical height is znode = rnode·σ + η, which differs
    # from the reference rnode once η ≠ 0. `node` (and hence interpolation targets and every
    # node-evaluated forcing / closure / boundary function) must report the physical znode.
    z = Oceananigans.Grids.MutableVerticalDiscretization(collect(0:0.25:1))   # 5 faces ⇒ Nz = 4
    grid = RectilinearGrid(CPU(); size=(2, 2, 4), x=(0, 1), y=(0, 1), z=z,
                           topology=(Periodic, Periodic, Bounded))

    Δη = 0.3
    fill!(grid.z.ηⁿ, Δη)                                       # σ stays 1, η = Δη ⇒ znode = rnode + Δη
    c = Center()
    for k in 1:4
        zr = Oceananigans.Grids.rnode(1, 1, k, grid, c, c, c)
        zz = Oceananigans.Grids.znode(1, 1, k, grid, c, c, c)
        @test zz ≈ zr + Δη                                     # deformation is actually present
        @test Oceananigans.Grids.node(1, 1, k, grid, c, c, c)[end] == zz   # node reports znode, not rnode
    end
end
