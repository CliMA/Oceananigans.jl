include("dependencies_for_runtests.jl")

using Oceananigans.Fields: interpolate
using Oceananigans.Units: Time

# `interpolate(f, …)` applies `f` to each source value before the weighted blend, like `mean(f, itr)`:
# the result stays in `f`-space (no inverse), so `exp(interpolate(log, …))` is geometric interpolation.

@testset "Mapped interpolation interpolate(f, …)" begin
    for arch in archs, FT in float_types
        @info "  Testing interpolate(f, …) [$(typeof(arch)), $FT]..."

        grid = RectilinearGrid(arch, FT; size=(1, 1, 8), x=(0, 1), y=(0, 1), z=(0, 1),
                               topology=(Periodic, Periodic, Bounded))

        loc = (Center(), Center(), Center())

        # Strictly positive, exponentially varying in z: `log` of this is linear in z, so blending
        # log-values is exact while a plain linear blend overestimates this convex profile.
        profile(x, y, z) = exp(3z + 1)

        c = CenterField(grid)
        set!(c, profile)
        fill_halo_regions!(c)

        logc = Field(log(c))
        compute!(logc)
        fill_halo_regions!(logc)

        # Interior midpoints between adjacent cell centers (strictly between nodes).
        ztargets = [FT(k / 8) for k in 1:7]

        @allowscalar for zt in ztargets
            at_node = (FT(1//2), FT(1//2), zt)

            # `identity` reproduces the unmapped interpolation exactly.
            @test interpolate(identity, at_node, c, loc, grid) == interpolate(at_node, c, loc, grid)

            # Mapping through `log` is equivalent to interpolating the lazy `log(c)` field.
            @test interpolate(log, at_node, c, loc, grid) ≈ interpolate(at_node, logc, loc, grid)

            # For an exponential profile, geometric interpolation is exact and beats the linear blend,
            # which overestimates a convex function between nodes.
            true_value = exp(3 * zt + 1)
            ℑgeo = exp(interpolate(log, at_node, c, loc, grid))
            ℑlin = interpolate(at_node, c, loc, grid)
            @test ℑgeo ≈ true_value
            @test ℑlin > ℑgeo
        end

        # FieldTimeSeries space + time path with a time-constant exponential profile.
        times = range(FT(0), FT(2), length=5)
        fts = FieldTimeSeries{Center, Center, Center}(grid, times)
        set!(fts, (x, y, z, t) -> profile(x, y, z))
        fill_halo_regions!(fts)

        @allowscalar for zt in ztargets
            at_node = (FT(1//2), FT(1//2), zt)
            t = Time(FT(7//10))

            # `identity` reproduces the unmapped FTS interpolation exactly.
            @test interpolate(identity, at_node, t, fts, loc, grid) == interpolate(at_node, t, fts, loc, grid)

            # Time-constant exponential: geometric interpolation recovers the profile exactly.
            @test exp(interpolate(log, at_node, t, fts, loc, grid)) ≈ exp(3 * zt + 1)
        end
    end
end
