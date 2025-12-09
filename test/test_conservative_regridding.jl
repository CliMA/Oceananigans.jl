include("dependencies_for_runtests.jl")

using ConservativeRegridding
using Statistics: mean

@testset "ConservativeRegridding extension" begin
    @info "Testing ConservativeRegridding extension..."

    # Test with LatitudeLongitudeGrid
    coarse_grid = LatitudeLongitudeGrid(size=(90, 45, 1),
                                        longitude=(0, 360),
                                        latitude=(-90, 90),
                                        z=(0, 1))

    fine_grid = LatitudeLongitudeGrid(size=(360, 180, 1),
                                      longitude=(0, 360),
                                      latitude=(-90, 90),
                                      z=(0, 1))

    dst = CenterField(coarse_grid)
    src = CenterField(fine_grid)

    # Set random values on fine grid
    set!(src, (x, y, z) -> rand())

    # Build regridder and regrid
    regridder = ConservativeRegridding.Regridder(dst, src)
    regrid!(dst, regridder, src)

    # Check mean is approximately preserved
    @test mean(dst) ≈ mean(src) rtol=1e-2

    # Test backwards regridding (dst -> src)
    set!(dst, (x, y, z) -> rand())
    regrid!(src, transpose(regridder), dst)
    @test mean(dst) ≈ mean(src) rtol=1e-2

    # Test with RectilinearGrid and partial overlap
    @info "  Testing RectilinearGrid with partial domain overlap..."
    large_domain_grid = RectilinearGrid(size=(100, 100),
                                        x=(0, 2), y=(0, 2),
                                        topology=(Periodic, Periodic, Flat))

    small_domain_grid = RectilinearGrid(size=(200, 200),
                                        x=(0, 1), y=(0, 1),
                                        topology=(Periodic, Periodic, Flat))

    src_rect = CenterField(small_domain_grid)
    dst_rect = CenterField(large_domain_grid)

    set!(src_rect, 1)

    regridder_rect = ConservativeRegridding.Regridder(dst_rect, src_rect)
    regrid!(dst_rect, regridder_rect, src_rect)

    # Check that integral is preserved for overlapping region
    dst_int = Field(Integral(dst_rect))
    src_int = Field(Integral(src_rect))
    compute!(dst_int)
    compute!(src_int)

    @test dst_int[1, 1, 1] ≈ src_int[1, 1, 1]
end

