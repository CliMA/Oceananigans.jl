include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: znodes
using Oceananigans.MultiRegion: YinYangField, yin_yang_masks
using Oceananigans.Utils: getregion

# Owned area fractions of the interior (Center, Center) cells, on the CPU
cpu_owned_weights(grid) = on_architecture(CPU(), yin_yang_masks(grid).cell_weights)[1:size(grid, 1), 1:size(grid, 2), 1]

single_value(r) = Array(interior(r))[1]

# Set the interior points of both panels of `field` for which `exclude(i, j)` holds to `value`,
# and every halo point to `value`, keeping the other interior values
function corrupt_unowned_points!(field, value, exclude)
    for region in 1:2
        regional_field = getregion(field, region)
        data = Array(interior(regional_field))

        for j in axes(data, 2), i in axes(data, 1)
            exclude(i, j) && (data[i, j, :] .= value)
        end

        parent(regional_field) .= value
        set!(regional_field, data)
    end

    return field
end

# A second-degree spherical harmonic about the x-axis, whose average is zero
equatorial_harmonic(λ, φ, z) = (3 * (cosd(φ) * cosd(λ))^2 - 1) / 2

@testset "Yin-Yang reductions" begin
    for arch in archs
        grid = YinYangGrid(arch; size=(48, 16, 4), z=(-1, 0), radius=1)
        w = cpu_owned_weights(grid)
        unowned(i, j) = iszero(w[i, j])
        Nz = size(grid, 3)

        @testset "Owned-cell extrema [$(typeof(arch))]" begin
            c = CenterField(grid)
            set!(c, (λ, φ, z) -> sind(φ) + z)
            owned_maximum = maximum(c)
            owned_minimum = minimum(c)

            # Halos, and overlap cells that the other panel owns, never enter, even when they hold NaN
            corrupt_unowned_points!(c, 1000, unowned)
            @test maximum(c) == owned_maximum
            @test minimum(c) == owned_minimum
            @test all(x -> x < 100, c)

            corrupt_unowned_points!(c, NaN, unowned)
            @test maximum(c) == owned_maximum
            @test !isnan(sum(c))

            # The owned restriction composes with a user condition
            top_level(i, j, k, grid, c) = k == Nz
            set!(c, (λ, φ, z) -> z)
            corrupt_unowned_points!(c, 1000, unowned)
            @test maximum(c; condition=top_level) ≈ -1 / 8
            @test any(x -> x > -0.2, c; condition=top_level)
            @test !all(x -> x > -0.2, c)

            # Products count each owned point once
            set!(c, 1)
            corrupt_unowned_points!(c, 2, unowned)
            @test prod(c) == 1

            # Staggered points without any owned neighbor are excluded too
            u = XFaceField(grid)
            set!(u, 1)
            corrupt_unowned_points!(u, 1000, (i, j) -> unowned(i, j) & (i == 1 || unowned(i - 1, j)))
            @test maximum(u) == 1

            # Column reductions are computed panel by panel over all interior points
            column_maximum = maximum(c, dims=3)
            @test column_maximum isa YinYangField
            @test maximum(interior(getregion(column_maximum, 2))) == 2

            # Horizontal extrema are owned-only at every level
            horizontal_maximum = maximum(c, dims=(1, 2))
            @test horizontal_maximum.grid === getregion(grid, 1)
            @test all(==(1), interior(horizontal_maximum))

            # Reductions over one horizontal dimension would mix the panels
            @test_throws ArgumentError sum(c; dims=1)
        end

        @testset "Owned-weighted sums and integrals [$(typeof(arch)), $FT]" for FT in float_types
            typed_grid = YinYangGrid(arch, FT; size=(48, 16, Nz), z=(-1, 0), radius=1)
            owned_count = 2 * sum(Float64, w) # Σ w over both panels
            tolerance = FT == Float64 ? 1e-13 : 1e-4
            sphere_volume = 4π

            c = CenterField(typed_grid)
            set!(c, 3)
            corrupt_unowned_points!(c, NaN, unowned)

            @test sum(c) ≈ 3 * owned_count * Nz rtol=tolerance
            @test mean(c) ≈ 3 rtol=tolerance
            @test sum(x -> x > 0, c) ≈ owned_count * Nz rtol=tolerance

            # Integrals of constants are exact at every horizontal location
            for loc in ((Center, Center, Center), (Face, Center, Center), (Face, Face, Center))
                f = Field{loc...}(typed_grid)
                set!(f, 3)
                @test single_value(Field(Integral(f))) ≈ 3 * sphere_volume rtol=tolerance
            end
            @test single_value(Field(Average(c))) ≈ 3 rtol=tolerance

            surface = Field{Center, Center, Nothing}(typed_grid)
            set!(surface, 1)
            @test single_value(Field(Integral(surface))) ≈ 4π rtol=tolerance

            # User conditions compose with the owned weights
            top_level(i, j, k, grid, c) = k == Nz
            @test single_value(Field(Integral(c; condition=top_level))) ≈ 3 * sphere_volume / Nz rtol=tolerance
            @test mean(c; condition=top_level) ≈ 3 rtol=tolerance

            # Column integrals are computed panel by panel
            column_integral = Field(Integral(c, dims=3))
            @test column_integral isa YinYangField
            yin_column = Array(interior(getregion(column_integral, 1)))[:, :, 1]
            unowned_columns = [unowned(i, j) for i in axes(yin_column, 1), j in axes(yin_column, 2)]
            @test all(isnan, yin_column[unowned_columns])
            @test all(yin_column[.!unowned_columns] .≈ 3)
        end

        @testset "Horizontal averages [$(typeof(arch))]" begin
            zᶜ = Array(znodes(getregion(grid, 1), Center()))
            profile(z) = exp(z) + z^3

            c = CenterField(grid)
            set!(c, (λ, φ, z) -> profile(z))
            corrupt_unowned_points!(c, NaN, unowned)

            # Functions of z alone are averaged exactly at every level
            c̄ = Field(Average(c, dims=(1, 2)))
            @test c̄.grid === getregion(grid, 1)
            @test location(c̄) == (Nothing, Nothing, Center)
            @test Array(interior(c̄, 1, 1, :)) ≈ profile.(zᶜ) rtol=1e-13
            @test Array(interior(mean(c, dims=(1, 2)), 1, 1, :)) ≈ profile.(zᶜ) rtol=1e-13
            @test Array(interior(Field(Integral(c, dims=(1, 2))), 1, 1, :)) ≈ 4π .* profile.(zᶜ) rtol=1e-13

            # A spherical harmonic averages to zero with second-order convergence
            errors = map((16, 32)) do Nφ
                f = CenterField(YinYangGrid(arch; size=(3Nφ, Nφ, 1), z=(-1, 0)))
                set!(f, equatorial_harmonic)
                return abs(single_value(Field(Average(f, dims=(1, 2)))))
            end
            @test errors[1] < 2e-4
            @test log2(errors[1] / errors[2]) > 1.8
        end

        @testset "Integrals of abstract operations [$(typeof(arch))]" begin
            ρ = CenterField(grid)
            u = XFaceField(grid)
            v = YFaceField(grid)
            set!(ρ, 2)
            set!(u, 3)
            set!(v, 5)
            corrupt_unowned_points!(ρ, NaN, unowned)

            # Interpolation pulls in non-owned neighbors, but they are weighted by zero
            @test single_value(Field(Integral(ρ * u * u))) ≈ 18 * 4π rtol=1e-13
            @test Array(interior(Field(Integral(ρ * u * v, dims=(1, 2))), 1, 1, :)) ≈ fill(30 * 4π, Nz) rtol=1e-13

            # ∫ (1 + z) cos²φ / 2 dV = (1/2) (1/2) (8π/3) over the unit sphere and z ∈ [-1, 0]
            set!(ρ, (λ, φ, z) -> 1 + z)
            set!(u, (λ, φ, z) -> cosd(φ))
            @test single_value(Field(Integral(ρ * u * u / 2))) ≈ 2π / 3 rtol=1e-3
        end
    end
end
