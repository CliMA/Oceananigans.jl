include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: znodes
using Oceananigans.MultiRegion: YinYangField, yin_yang_masks
using Oceananigans.Utils: getregion

single_value(r) = Array(interior(r))[1]
top_level(i, j, k, grid, c) = k == size(grid, 3)

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
    corrupt_unowned_points!(c, 1000, unowned)
    @test (maximum(c), minimum(c)) == owned_extrema
    @test all(x -> x < 100, c)
    corrupt_unowned_points!(c, NaN, unowned)
    @test (maximum(c), minimum(c)) == owned_extrema
    @test !isnan(sum(c))

    # The owned restriction composes with a user condition
    set!(c, (λ, φ, z) -> z)
    corrupt_unowned_points!(c, 1000, unowned)
    @test maximum(c; condition=top_level) ≈ -1 / 2Nz
    @test any(x -> x > -0.3, c; condition=top_level)
    @test !all(x -> x > -0.3, c)

    # Staggered points without any owned neighbor are excluded too
    u = XFaceField(grid)
    set!(u, 1)
    corrupt_unowned_points!(u, 1000, (i, j) -> unowned(i, j) & (i == 1 || unowned(i - 1, j)))
    @test maximum(u) == 1

    # Products count each owned point once; column reductions are computed panel by panel over all interior points,
    # horizontal reductions are owned-only at every level, and reductions over one horizontal dimension would mix the panels
    set!(c, 1)
    corrupt_unowned_points!(c, 2, unowned)
    @test prod(c) == 1
    @test maximum(c, dims=3) isa YinYangField
    @test maximum(interior(getregion(maximum(c, dims=3), 2))) == 2
    @test all(==(1), interior(maximum(c, dims=(1, 2))))
    @test_throws ArgumentError sum(c; dims=1)

    # Sums and integrals weight each owned point once; integrals of constants are exact at every location
    set!(c, 3)
    corrupt_unowned_points!(c, NaN, unowned)
    owned_count = 2 * sum(Float64, w)
    @test sum(c) ≈ 3 * owned_count * Nz rtol=tolerance
    @test mean(c) ≈ 3 rtol=tolerance
    @test mean(c; condition=top_level) ≈ 3 rtol=tolerance
    @test single_value(Field(Integral(c; condition=top_level))) ≈ 3 * 4π / Nz rtol=tolerance
    for loc in ((Center, Center, Center), (Face, Center, Center), (Face, Face, Center), (Center, Center, Nothing))
        f = Field{loc...}(grid)
        set!(f, 3)
        @test single_value(Field(Integral(f))) ≈ 3 * 4π rtol=tolerance
    end

    # Column integrals are computed panel by panel
    column_integral = Field(Integral(c, dims=3))
    @test column_integral isa YinYangField
    yin_columns = Array(interior(getregion(column_integral, 1)))[:, :, 1]
    @test all(unowned(i, j) ? isnan(yin_columns[i, j]) : yin_columns[i, j] ≈ 3 for i in 1:Nx, j in 1:Ny)

    # Functions of z alone are averaged exactly at every level, on the grid of Yin
    zᶜ = Array(znodes(getregion(grid, 1), Center()))
    profile(z) = exp(z) + z^3
    set!(c, (λ, φ, z) -> profile(z))
    corrupt_unowned_points!(c, NaN, unowned)
    c̄ = Field(Average(c, dims=(1, 2)))
    @test c̄.grid === getregion(grid, 1)
    @test location(c̄) == (Nothing, Nothing, Center)
    @test Array(interior(c̄, 1, 1, :)) ≈ profile.(zᶜ) rtol=tolerance
    @test Array(interior(Field(Integral(c, dims=(1, 2))), 1, 1, :)) ≈ 4π .* profile.(zᶜ) rtol=tolerance

    # Interpolation of abstract operations pulls in non-owned neighbors, but they are weighted by zero
    v = YFaceField(grid)
    set!(c, 2)
    set!(u, 3)
    set!(v, 5)
    corrupt_unowned_points!(c, NaN, unowned)
    @test single_value(Field(Integral(c * u * u))) ≈ 18 * 4π rtol=tolerance
    @test Array(interior(Field(Integral(c * u * v, dims=(1, 2))), 1, 1, :)) ≈ fill(30 * 4π, Nz) rtol=tolerance
end

# A second-degree spherical harmonic about the x-axis, whose average is zero
equatorial_harmonic(λ, φ, z) = (3 * (cosd(φ) * cosd(λ))^2 - 1) / 2

@testset "Convergence of Yin-Yang averages [$(summary(arch))]" for arch in archs
    errors = map((16, 32)) do Nφ
        f = CenterField(YinYangGrid(arch; size=(3Nφ, Nφ, 1), z=(-1, 0)))
        set!(f, equatorial_harmonic)
        return abs(single_value(Field(Average(f, dims=(1, 2)))))
    end
    @test errors[1] < 2e-4
    @test log2(errors[1] / errors[2]) > 1.8 # second order
end
