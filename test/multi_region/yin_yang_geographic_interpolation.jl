include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.MultiRegion: yin_yang_geographic_interpolation, latitude_longitude_grid, interpolated_field, yin_yang_owned

# The largest error of the interior of `field` with respect to `exact(λ, φ)`
function interpolation_error(field, exact)
    grid = on_architecture(CPU(), field.grid)
    ℓx, ℓy, _ = instantiated_location(field)
    return maximum(abs, Array(interior(field)) .- exact.(λnodes(grid, ℓx), φnodes(grid, ℓy)'))
end

function scalar_interpolation_error(grid, ℓ)
    c = location_field(ℓ..., grid)
    set!(c, (λ, φ, z) -> spherical_harmonics(λ, φ))
    return interpolation_error(interpolated_field(c), spherical_harmonics)
end

zonal_average(source) = Field(Average(source, dims=1))

@testset "Yin-Yang interpolation [$(typeof(arch))]" for arch in archs
    grid = YinYangGrid(arch; size=(48, 16, 2), z=(-1, 0))
    destination_grid = latitude_longitude_grid(grid)
    cpu_destination_grid = on_architecture(CPU(), destination_grid)
    @test size(destination_grid) == (64, 32, 2)

    @testset "Interpolation tables" begin
        c = CenterField(grid)
        for ℓ in yin_yang_locations
            interpolation = yin_yang_geographic_interpolation(location_field(ℓ..., destination_grid), c)
            yin, yang = map(table -> on_architecture(CPU(), table), interpolation)

            # Every destination node is filled once, by the panel that owns it, and Yin's basis is geographic
            Nx, Ny, _ = size(destination_grid, (ℓ..., Center()))
            yin_nodes = collect(zip(yin.receivers.i, yin.receivers.j))
            yang_nodes = collect(zip(yang.receivers.i, yang.receivers.j))
            @test sort(vcat(yin_nodes, yang_nodes)) == sort(vec([(i, j) for i in 1:Nx, j in 1:Ny]))
            λ, φ = λnodes(cpu_destination_grid, ℓ[1]), φnodes(cpu_destination_grid, ℓ[2])
            owned(i, j) = yin_yang_owned(Float64(λ[i]), Float64(φ[j]))
            @test all(owned(i, j) for (i, j) in yin_nodes) && !any(owned(i, j) for (i, j) in yang_nodes)
            @test all(==(1), yin.rotation.cosψ) && all(==(0), yin.rotation.sinψ)
        end

        typed_grid = YinYangGrid(arch, Float32; size=(48, 16, 2), z=(-1, 0))
        @test eltype(interpolated_field(CenterField(typed_grid))) == Float32
    end

    @testset "Interpolated values" begin
        # Constants are interpolated exactly, and the halos of the source are never read
        for ℓ in yin_yang_locations, ℓᵈ in ((Center(), Center()), (Face(), Face()))
            c = location_field(ℓ..., grid)
            for r in 1:2
                parent(getregion(c, r)) .= NaN
                interior(getregion(c, r)) .= 1
            end
            cˡ = location_field(ℓᵈ..., destination_grid)
            interpolate!(cˡ, c)
            @test all(x -> abs(x - 1) ≤ 20eps(), interior(cˡ))
        end

        errors = [scalar_interpolation_error(YinYangGrid(arch; size=(3Nφ, Nφ, 1), z=(-1, 0)), (Center(), Center()))
                  for Nφ in (16, 32)]
        @test errors[1] < 1e-3
        @test only(convergence_rates(errors)) > 3.5
        @test all(scalar_interpolation_error(grid, ℓ) < 1e-3 for ℓ in yin_yang_locations)

        # Vectors are interpolated onto their geographic components; the (Center, Face) and (Face, Face)
        # destinations include the poles
        u, v = XFaceField(grid), YFaceField(grid)
        set_panel_components!(u, v, solid_body_velocity)
        for ℓ in ((Center(), Center()), (Center(), Face()), (Face(), Face()))
            uᵍ, vᵍ = location_field(ℓ..., destination_grid), location_field(ℓ..., destination_grid)
            interpolate!((uᵍ, vᵍ), (u, v))
            @test interpolation_error(uᵍ, (λ, φ) -> solid_body_velocity(λ, φ)[1]) < 1e-5
            @test interpolation_error(vᵍ, (λ, φ) -> solid_body_velocity(λ, φ)[2]) < 1e-5
        end

        # Interpolated fields are recomputed from their source: a smooth function, x (z + x) in Cartesian
        # coordinates, has the zonal mean cos²φ / 2
        f(λ, φ) = cosd(φ) * cosd(λ) * (sind(φ) + cosd(φ) * cosd(λ))
        c = CenterField(grid)
        set!(c, (λ, φ, z) -> f(λ, φ))
        cˡ = interpolated_field(c, destination_grid)
        @test cˡ.grid === destination_grid
        @test location(cˡ) == (Center, Center, Center)
        @test interpolation_error(cˡ, f) < 1e-4
        c̄ = zonal_average(cˡ)
        φ = φnodes(cpu_destination_grid, Center())
        @test Array(interior(c̄, 1, :, 1)) ≈ cosd.(φ).^2 ./ 2 rtol=1e-4
        set!(c, (λ, φ, z) -> sind(φ))
        compute!(c̄)
        @test Array(interior(c̄, 1, :, 2)) ≈ sind.(φ) rtol=1e-4

        # Scalars are interpolated to their own location, and vectors to cell centers
        set!(v, (λ, φ, z) -> sind(φ))
        vˡ = interpolated_field(v)
        @test location(vˡ) == (Center, Face, Center)
        @test interpolation_error(vˡ, (λ, φ) -> sind(φ)) < 2e-5

        set_panel_components!(u, v, solid_body_velocity)
        uᵍ, vᵍ = interpolated_field((u, v))
        ū, v̄ = map(zonal_average, (uᵍ, vᵍ))
        @test location(v̄) == (Nothing, Center, Center)
        @test Array(interior(ū, 1, :, 1)) ≈ cosd.(φ) .* cos(π/4) rtol=1e-5
        @test maximum(abs, interior(v̄)) < 1e-10

        # Interpolated fields are windowed in the vertical, within the window of the source
        c₂ = interpolated_field(Field(c; indices=(:, :, 2)), destination_grid)
        @test size(c₂) == (64, 32, 1)
        @test interior(c₂) == interior(cˡ, :, :, 2:2)
        @test interior(interpolated_field(c, destination_grid; indices=(:, :, 2))) == interior(cˡ, :, :, 2:2)
        @test_throws ArgumentError interpolated_field(Field(c; indices=(1:2, :, :)), destination_grid)
        @test_throws ArgumentError interpolated_field(Field(c; indices=(:, :, 1)), destination_grid; indices=(:, :, 2))
        @test summary(cˡ.operand) == "Interpolation of $(summary(c)) with YinYangGeographicInterpolation"
        @test summary(uᵍ.operand) == "Interpolation of the eastward component of a vector with YinYangGeographicInterpolation"
        @test summary(vᵍ.operand) == "Interpolation of the northward component of a vector with YinYangGeographicInterpolation"

        # `set!` interpolates, and the vertical sizes must agree
        set!(cˡ, c)
        @test interior(cˡ) == interior(interpolate!(CenterField(destination_grid), c))
        shallow_grid = LatitudeLongitudeGrid(arch; size=(8, 4, 1), longitude=(-180, 180), latitude=(-90, 90), z=(-1, 0))
        @test_throws ArgumentError interpolate!(CenterField(shallow_grid), c)
    end

    @testset "Interpolated coastal scalars [$(nameof(Bottom))]" for Bottom in (GridFittedBottom, PartialCellBottom)
        land(λ, φ) = (φ > 50) | ((abs(λ - 30) < 10) & (φ > -30))
        immersed_grid = ImmersedBoundaryGrid(grid, Bottom((λ, φ) -> ifelse(land(λ, φ), 0, -1)))
        c = CenterField(immersed_grid)
        set!(c, (λ, φ, z) -> ifelse(land(λ, φ), 0, 3))

        # The destination nodes do not coincide with the cell centers of either panel
        cˡ = interpolated_field(c, latitude_longitude_grid(immersed_grid; size=(72, 36)))
        @test cˡ.grid isa ImmersedBoundaryGrid
        @test minimum(cˡ) ≈ 3
        @test maximum(cˡ) ≈ 3
    end
end
