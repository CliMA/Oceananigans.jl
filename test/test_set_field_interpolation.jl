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
            @test !Oceananigans.Fields.matching_field_discretization(bottom_u, top_u)
        end
    end
end
