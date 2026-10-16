include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.MultiRegion: YinYangRemapping, YinYangInterpolationTable, latitude_longitude_grid,
                                remap!, remapped_field, yin_yang_owned

# The largest error of the interior of `field` with respect to `exact(λ, φ)`
function remap_error(field, exact)
    grid = on_architecture(CPU(), field.grid)
    ℓx, ℓy, _ = instantiated_location(field)
    return maximum(abs, Array(interior(field)) .- exact.(λnodes(grid, ℓx), φnodes(grid, ℓy)'))
end

function scalar_remap_error(grid, source_location)
    c = location_field(source_location..., grid)
    set!(c, (λ, φ, z) -> spherical_harmonics(λ, φ))
    cˡ = CenterField(latitude_longitude_grid(grid))
    remap!(cˡ, c)
    return remap_error(cˡ, spherical_harmonics)
end

zonal_average(source) = Field(Average(source, dims=1))

@testset "Yin-Yang remapping" begin
    for arch in archs
        grid = YinYangGrid(arch; size=(48, 16, 2), z=(-1, 0))
        destination_grid = latitude_longitude_grid(grid)
        cpu_destination_grid = on_architecture(CPU(), destination_grid)

        @testset "Remapping tables [$(typeof(arch)), $FT]" for FT in float_types
            typed_grid = YinYangGrid(arch, FT; size=(48, 16, 2), z=(-1, 0))
            typed_destination_grid = latitude_longitude_grid(typed_grid)
            @test size(typed_destination_grid) == (64, 32, 2)
            @test eltype(typed_destination_grid) == FT

            for ℓ in yin_yang_locations
                remapping = YinYangRemapping(typed_grid; destination_grid = typed_destination_grid, destination_location = ℓ)
                yin, yang = map(tables -> on_architecture(CPU(), first(tables)), remapping.tables)
                @test yin isa YinYangInterpolationTable{4}
                @test eltype(yin.weights.x) == FT

                # Every destination node is filled once, by the panel that owns it with the default overlap
                Nx, Ny, _ = size(typed_destination_grid, (ℓ..., Center()))
                yin_nodes = collect(zip(yin.receivers.i, yin.receivers.j))
                yang_nodes = collect(zip(yang.receivers.i, yang.receivers.j))
                @test sort(vcat(yin_nodes, yang_nodes)) == sort(vec([(i, j) for i in 1:Nx, j in 1:Ny]))

                λ = λnodes(cpu_destination_grid, ℓ[1])
                φ = φnodes(cpu_destination_grid, ℓ[2])
                owned(i, j) = yin_yang_owned(Float64(λ[i]), Float64(φ[j]))
                @test all(owned(i, j) for (i, j) in yin_nodes)
                @test !any(owned(i, j) for (i, j) in yang_nodes)

                # Yin's basis is geographic
                @test all(==(1), yin.rotation.cosψ)
                @test all(==(0), yin.rotation.sinψ)
            end

            remapping = YinYangRemapping(typed_grid; destination_grid = typed_destination_grid)
            @test startswith(summary(remapping), "YinYangRemapping from (Center, Center) to (Center, Center) on 64×32×2 LatitudeLongitudeGrid")
        end

        @testset "Remapped values [$(typeof(arch))]" begin
            # Constants are remapped exactly, and the halos of the source are never read
            for ℓ in yin_yang_locations, ℓᵈ in ((Center(), Center()), (Face(), Face()))
                c = location_field(ℓ..., grid)
                for r in 1:2
                    parent(getregion(c, r)) .= NaN
                    interior(getregion(c, r)) .= 1
                end
                cˡ = location_field(ℓᵈ..., destination_grid)
                remap!(cˡ, c)
                @test all(x -> abs(x - 1) ≤ 20eps(), interior(cˡ))
            end

            errors = [scalar_remap_error(YinYangGrid(arch; size=(3Nφ, Nφ, 1), z=(-1, 0)), (Center(), Center())) for Nφ in (16, 32)]
            @test errors[1] < 1e-3
            @test only(convergence_rates(errors)) > 3.5

            for ℓ in yin_yang_locations
                @test scalar_remap_error(grid, ℓ) < 1e-3
            end

            # Vectors are remapped onto their geographic components; the (Center, Face) and (Face, Face)
            # destinations include the poles
            u = XFaceField(grid)
            v = YFaceField(grid)
            set_panel_components!(u, v, solid_body_velocity)
            for ℓ in ((Center(), Center()), (Center(), Face()), (Face(), Face()))
                uᵍ = location_field(ℓ..., destination_grid)
                vᵍ = location_field(ℓ..., destination_grid)
                remap!((uᵍ, vᵍ), (u, v))
                @test remap_error(uᵍ, (λ, φ) -> solid_body_velocity(λ, φ)[1]) < 1e-5
                @test remap_error(vᵍ, (λ, φ) -> solid_body_velocity(λ, φ)[2]) < 1e-5
            end
        end

        @testset "Remapped fields [$(typeof(arch))]" begin
            remapping = YinYangRemapping(grid; destination_grid)

            # A smooth function, x (z + x) in Cartesian coordinates, with zonal mean cos²φ / 2
            f(λ, φ) = cosd(φ) * cosd(λ) * (sind(φ) + cosd(φ) * cosd(λ))
            c = CenterField(grid)
            set!(c, (λ, φ, z) -> f(λ, φ))

            cˡ = remapped_field(c, remapping)
            @test cˡ.grid === destination_grid
            @test location(cˡ) == (Center, Center, Center)
            @test remap_error(cˡ, f) < 1e-4

            # Averages of remapped fields are recomputed from the source
            c̄ = zonal_average(cˡ)
            φ = φnodes(cpu_destination_grid, Center())
            @test Array(interior(c̄, 1, :, 1)) ≈ cosd.(φ).^2 ./ 2 rtol=1e-4

            set!(c, (λ, φ, z) -> sind(φ))
            compute!(c̄)
            @test Array(interior(c̄, 1, :, 2)) ≈ sind.(φ) rtol=1e-4

            # Scalars are remapped to their own location, and vectors to cell centers
            v = YFaceField(grid)
            set!(v, (λ, φ, z) -> sind(φ))
            vˡ = remapped_field(v, destination_grid)
            @test location(vˡ) == (Center, Face, Center)
            @test remap_error(vˡ, (λ, φ) -> sind(φ)) < 2e-5

            u = XFaceField(grid)
            set_panel_components!(u, v, solid_body_velocity)
            ū, v̄ = map(zonal_average, remapped_field((u, v)))
            @test location(v̄) == (Nothing, Center, Center)
            @test Array(interior(ū, 1, :, 1)) ≈ cosd.(φ) .* cos(π/4) rtol=1e-5
            @test maximum(abs, interior(v̄)) < 1e-10

            # Remapping checks its inputs
            @test_throws ArgumentError remap!(XFaceField(destination_grid), c, remapping)
            @test_throws ArgumentError remap!(CenterField(destination_grid), XFaceField(grid), remapping)
            @test_throws ArgumentError remapped_field(XFaceField(grid), remapping)
            @test_throws ArgumentError YinYangRemapping((CenterField(destination_grid), XFaceField(destination_grid)), (u, v))
            @test_throws ArgumentError YinYangRemapping(grid; destination_grid = grid)

            deeper_grid = LatitudeLongitudeGrid(arch; size=(8, 4, 2), longitude=(-180, 180), latitude=(-90, 90), z=(-10, 0))
            @test_throws ArgumentError remap!(CenterField(deeper_grid), c)
        end

        @testset "Remapped coastal scalars [$(typeof(arch)), $(nameof(Bottom))]" for Bottom in (GridFittedBottom, PartialCellBottom)
            land(λ, φ) = (φ > 50) | ((abs(λ - 30) < 10) & (φ > -30))
            immersed_grid = ImmersedBoundaryGrid(grid, Bottom((λ, φ) -> ifelse(land(λ, φ), 0, -1)))
            c = CenterField(immersed_grid)
            set!(c, (λ, φ, z) -> ifelse(land(λ, φ), 0, 3))

            # The destination nodes do not coincide with the cell centers of either panel
            cˡ = remapped_field(c, latitude_longitude_grid(immersed_grid; size=(72, 36)))
            @test cˡ.grid isa ImmersedBoundaryGrid
            @test minimum(cˡ) ≈ 3
            @test maximum(cˡ) ≈ 3
        end
    end
end
