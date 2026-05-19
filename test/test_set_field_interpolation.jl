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

        same_size_grid = RectilinearGrid(arch, FT; size=(4, 4, 4),
                                         topology=(Periodic, Periodic, Bounded),
                                         interp_domain...)

        shifted_grid = RectilinearGrid(arch, FT; size=(4, 4, 4),
                                       topology=(Periodic, Periodic, Bounded),
                                       x=(FT(-0.5), FT(1.5)),
                                       y=(0, 1), z=(0, 1))

        shifted_c = CenterField(shifted_grid)
        set!(shifted_c, f_linear)
        fill_halo_regions!(shifted_c)

        same_size_different_grid = CenterField(same_size_grid)
        set!(same_size_different_grid, shifted_c)

        expected_same_size_different_grid = CenterField(same_size_grid)
        interpolate!(expected_same_size_different_grid, shifted_c)

        @test Array(interior(same_size_different_grid)) ==
              Array(interior(expected_same_size_different_grid))
    end
end
