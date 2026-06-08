include("dependencies_for_runtests.jl")

using ConservativeRegridding

@testset "ConservativeRegridding extension" begin
    @info "Testing ConservativeRegridding extension..."

    # Test with LatitudeLongitudeGrid
    @info "  Testing LatitudeLongitudeGrid regridding..."
    coarse_grid = LatitudeLongitudeGrid(size=(90, 45, 1),
                                        longitude=(0, 360),
                                        latitude=(-90, 90),
                                        z=(0, 1))

    fine_grid = LatitudeLongitudeGrid(size=(180, 90, 1),
                                      longitude=(0, 360),
                                      latitude=(-90, 90),
                                      z=(0, 1))

    coarse_field = CenterField(coarse_grid)
    fine_field = CenterField(fine_grid)

    # Build regridder
    regridder = ConservativeRegridding.Regridder(coarse_field, fine_field)

    # Test: field of 1's regrids to 1's (fine -> coarse)
    set!(fine_field, 1)
    regrid!(coarse_field, regridder, fine_field)
    @test all(interior(coarse_field) .≈ 1)

    # Test: field of 1's regrids to 1's (coarse -> fine)
    set!(coarse_field, 1)
    regrid!(fine_field, transpose(regridder), coarse_field)
    @test all(interior(fine_field) .≈ 1)

    # Test with RectilinearGrid
    @info "  Testing RectilinearGrid regridding..."
    coarse_rect_grid = RectilinearGrid(size=(50, 50),
                                       x=(0, 1), y=(0, 1),
                                       topology=(Periodic, Periodic, Flat))

    fine_rect_grid = RectilinearGrid(size=(100, 100),
                                     x=(0, 1), y=(0, 1),
                                     topology=(Periodic, Periodic, Flat))

    coarse_rect = CenterField(coarse_rect_grid)
    fine_rect = CenterField(fine_rect_grid)

    # Build regridder
    rect_regridder = ConservativeRegridding.Regridder(coarse_rect, fine_rect)

    # Test: field of 1's regrids to 1's (fine -> coarse)
    set!(fine_rect, 1)
    regrid!(coarse_rect, rect_regridder, fine_rect)
    @test all(interior(coarse_rect) .≈ 1)

    # Test: field of 1's regrids to 1's (coarse -> fine)
    set!(coarse_rect, 1)
    regrid!(fine_rect, transpose(rect_regridder), coarse_rect)
    @test all(interior(fine_rect) .≈ 1)
end
