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

        # When `u` and `v` differ in halo size but otherwise share the same
        # discretization, `set!` should copy (not interpolate). This matches
        # how with_halo-extended grids feed into materialize_immersed_boundary.
        big_halo_grid = RectilinearGrid(arch, FT; size=(4, 4, 4),
                                        halo=(3, 3, 3),
                                        interp_domain...)

        big_halo_c = CenterField(big_halo_grid)
        set!(big_halo_c, coarse)
        @test Array(interior(big_halo_c)) == Array(interior(coarse))
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
            @test !Oceananigans.Fields.copyable_fields(bottom_u, top_u)
        end
    end
end

@testset "copyable_fields policy" begin
    copyable_fields = Oceananigans.Fields.copyable_fields

    for arch in archs, FT in float_types
        grid = RectilinearGrid(arch, FT; size=(4, 4, 8), x=(0, 1), y=(0, 1), z=(0, 1))
        Nz = size(grid, 3)

        # equivalent_dimension: identical discretization copies.
        @test copyable_fields(CenterField(grid), CenterField(grid))

        # Differing locations in a non-degenerate dimension must interpolate, which is the
        # case the policy exists to protect: u and v sit at different physical nodes.
        @test !copyable_fields(XFaceField(grid), CenterField(grid))

        # degenerate_dimension: a reduced field and a windowed single layer both span one
        # vertical point, so the slab copies in either direction.
        reduced = Field{Center, Center, Nothing}(grid)
        single_layer = Field{Center, Center, Center}(grid, indices=(:, :, Nz:Nz))
        @test copyable_fields(reduced, single_layer)
        @test copyable_fields(single_layer, reduced)

        # expandable_source_dimension: a reduced source stretches across a full field.
        column = Field{Nothing, Nothing, Center}(grid)
        slice  = Field{Center, Center, Nothing}(grid)
        @test copyable_fields(CenterField(grid), column)
        @test copyable_fields(CenterField(grid), slice)

        # ...but only the source may stretch. A reduced *destination* fed by a many-celled
        # source would have to compress, which broadcasting cannot do, so it interpolates.
        @test !copyable_fields(column, CenterField(grid))
        @test !copyable_fields(slice, CenterField(grid))

        # A singleton dimension that still carries a node is not degenerate: stretching a
        # `Center` value across many cells is not a copy.
        thin_grid = RectilinearGrid(arch, FT; size=(4, 4, 1), x=(0, 1), y=(0, 1), z=(0, 1))
        @test !copyable_fields(CenterField(grid), CenterField(thin_grid))
    end
end

@testset "set! expands a reduced source by copying" begin
    for arch in archs, FT in float_types
        grid = RectilinearGrid(arch, FT; size=(4, 4, 8), x=(0, 1), y=(0, 1), z=(0, 1))

        # A 1D reference column set into a 3D field must reproduce what interpolation
        # produced before expansion became a copy: a reduced dimension carries no node,
        # so replicating the single value is exactly the interpolated result.
        column = Field{Nothing, Nothing, Center}(grid)
        set!(column, z -> 2z + 1)
        fill_halo_regions!(column)

        expanded = CenterField(grid)
        set!(expanded, column)

        expected = CenterField(grid)
        interpolate!(expected, column)

        @test Array(interior(expanded)) == Array(interior(expected))

        # The same for a horizontal slice stretched over every vertical level.
        slice = Field{Center, Center, Nothing}(grid)
        set!(slice, (x, y) -> x + 2y)
        fill_halo_regions!(slice)

        expanded_slice = CenterField(grid)
        set!(expanded_slice, slice)

        expected_slice = CenterField(grid)
        interpolate!(expected_slice, slice)

        @test Array(interior(expanded_slice)) == Array(interior(expected_slice))
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
