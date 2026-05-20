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
