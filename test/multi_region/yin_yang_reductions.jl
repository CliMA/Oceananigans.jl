include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: znodes
using Oceananigans.MultiRegion: yin_yang_masks
using Oceananigans.Utils: getregion

single_value(r) = Array(interior(r))[1]
top_level(i, j, k, grid, c) = k == size(grid, 3)

# A second-degree spherical harmonic about the x-axis, whose average over the sphere is zero
equatorial_harmonic(λ, φ) = (3 * (cosd(φ) * cosd(λ))^2 - 1) / 2

# Set every halo point, and the interior points for which `unowned(i, j)` holds, of both panels of `field` to `value`
function corrupt_unowned_points!(field, value, unowned)
    for region in 1:2
        regional_field = getregion(field, region)
        data = Array(interior(regional_field))
        for j in axes(data, 2), i in axes(data, 1)
            unowned(i, j) && (data[i, j, :] .= value)
        end
        parent(regional_field) .= value
        set!(regional_field, data)
    end
    return field
end

@testset "Yin-Yang reductions [$(summary(arch)), $FT]" for arch in archs, FT in float_types
    grid = YinYangGrid(arch, FT; size=(48, 16, 2), z=(-1, 0), radius=1)
    Nx, Ny, Nz = size(grid) # of the panels
    w = on_architecture(CPU(), yin_yang_masks(grid).cell_weights)[1:Nx, 1:Ny, 1] # owned area fractions of the cells
    unowned(i, j) = iszero(w[i, j])
    tolerance = FT == Float64 ? 1e-13 : 1e-4
    c = CenterField(grid)

    # Halos, and overlap cells that the other panel owns, never enter, even when they hold NaN
    set!(c, (λ, φ, z) -> sind(φ) + z)
    owned_extrema = (maximum(c), minimum(c))
    corrupt_unowned_points!(c, NaN, unowned)
    @test (maximum(c), minimum(c)) == owned_extrema
    @test all(x -> x < 2, c)
    @test !isnan(sum(c))

    # The owned restriction composes with a user condition; staggered points without any owned neighbor are excluded too
    set!(c, (λ, φ, z) -> z)
    corrupt_unowned_points!(c, 1000, unowned)
    @test maximum(c; condition=top_level) ≈ -1 / 2Nz
    u = XFaceField(grid)
    set!(u, 1)
    corrupt_unowned_points!(u, 1000, (i, j) -> unowned(i, j) & (i == 1 || unowned(i - 1, j)))
    @test maximum(u) == 1

    # Products count each owned point once; column reductions are computed panel by panel over all interior points,
    # horizontal reductions are owned-only at every level, and reductions over one horizontal dimension would mix the panels
    set!(c, 1)
    corrupt_unowned_points!(c, 2, unowned)
    @test prod(c) == 1
    @test maximum(interior(getregion(maximum(c, dims=3), 2))) == 2
    @test all(==(1), interior(maximum(c, dims=(1, 2))))
    @test_throws ArgumentError sum(c; dims=1)

    # Sums and integrals weight each owned point once, also of operations that interpolate unowned neighbors;
    # integrals of constants are exact at every location
    set!(c, 3)
    corrupt_unowned_points!(c, NaN, unowned)
    @test sum(c) ≈ 3 * 2 * sum(Float64, w) * Nz rtol=tolerance
    @test mean(c; condition=top_level) ≈ 3 rtol=tolerance
    @test single_value(Field(Integral(c; condition=top_level))) ≈ 3 * 4π / Nz rtol=tolerance
    @test single_value(Field(Integral(c * u))) ≈ 3 * 4π rtol=tolerance
    d = CenterField(grid)
    set!(d, 3)
    @test c ≈ d # unowned points are excluded
    @test c != d
    for loc in ((Face, Center, Center), (Face, Face, Center), (Center, Center, Nothing))
        f = Field{loc...}(grid)
        set!(f, 3)
        @test single_value(Field(Integral(f))) ≈ 3 * 4π rtol=tolerance
    end

    # Horizontal averages live on the grid of Yin: a function of z is averaged exactly at every level,
    # and the harmonic to second order
    zᶜ = Array(znodes(getregion(grid, 1), Center()))
    profile(z) = exp(z) + z^3
    set!(c, (λ, φ, z) -> profile(z) + equatorial_harmonic(λ, φ))
    corrupt_unowned_points!(c, NaN, unowned)
    c̄ = Field(Average(c, dims=(1, 2)))
    @test c̄.grid === getregion(grid, 1)
    @test Array(interior(c̄, 1, 1, :)) ≈ profile.(zᶜ) atol=2e-4
end
