include("dependencies_for_runtests.jl")

using Oceananigans.Fields: regrid_in_x!, regrid_in_y!, regrid_in_z!

@testset "Field regridding" begin
    @info "  Testing field regridding..."

    L = 1.1
    ℓ = 0.5

    regular_ξ              = (0, L)
    fine_stretched_ξ       = [0, ℓ, L]
    very_fine_stretched_ξ  = [0, 0.2, 0.6, L]
    super_fine_stretched_ξ = [0, 0.1, 0.3, 0.65, L]

    topologies_1d = (x = (Bounded, Flat, Flat),
                     y = (Flat, Bounded, Flat),
                     z = (Flat, Flat, Bounded))

    sizes = (x = (2, 4, 6),
             y = (4, 2, 6),
             z = (4, 6, 2))

    topologies_3d = (x = (Bounded, Periodic, Periodic),
                     y = (Periodic, Bounded, Periodic),
                     z = (Periodic, Periodic, Bounded))

    regrid_xyz! = (x = regrid_in_x!,
                   y = regrid_in_y!,
                   z = regrid_in_z!)

    for arch in archs
        for dim in (:x, :y, :z)
            @testset "Regridding in $dim" begin
                regrid! = regrid_xyz![dim]
                topology = topologies_1d[dim]

                # 1D grids
                coarse_1d_regular_grid       = RectilinearGrid(arch, size=1; topology, Dict(dim => regular_ξ)...)
                fine_1d_regular_grid         = RectilinearGrid(arch, size=2; topology, Dict(dim => regular_ξ)...)
                fine_1d_stretched_grid       = RectilinearGrid(arch, size=2; topology, Dict(dim => fine_stretched_ξ)...)
                very_fine_1d_stretched_grid  = RectilinearGrid(arch, size=3; topology, Dict(dim => very_fine_stretched_ξ)...)
                super_fine_1d_stretched_grid = RectilinearGrid(arch, size=4; topology, Dict(dim => super_fine_stretched_ξ)...)
                super_fine_1d_regular_grid   = RectilinearGrid(arch, size=5; topology, Dict(dim => regular_ξ)...)

                # 3D grids
                topology = topologies_3d[dim]
                sz = sizes[dim]

                regular_kw = Dict{Any, Any}(d => (0, 1) for d in (:x, :y, :z) if d != dim)
                regular_kw[dim] = regular_ξ
                fine_regular_grid   = RectilinearGrid(arch, size=sz; topology, regular_kw...)

                fine_stretched_kw = Dict{Any, Any}(d => (0, 1) for d in (:x, :y, :z) if d != dim)
                fine_stretched_kw[dim] = fine_stretched_ξ
                fine_stretched_grid = RectilinearGrid(arch, size=sz; topology, fine_stretched_kw...)

                fine_stretched_c                    = CenterField(fine_stretched_grid)

                coarse_1d_regular_c                 = CenterField(coarse_1d_regular_grid)
                fine_1d_regular_c                   = CenterField(fine_1d_regular_grid)
                fine_1d_stretched_c                 = CenterField(fine_1d_stretched_grid)
                very_fine_1d_stretched_c            = CenterField(very_fine_1d_stretched_grid)
                super_fine_1d_stretched_c           = CenterField(super_fine_1d_stretched_grid)
                super_fine_1d_regular_c             = CenterField(super_fine_1d_regular_grid)
                super_fine_from_reduction_regular_c = CenterField(super_fine_1d_regular_grid)

                # We initialize an array on the `fine_1d_stretched_grid`, regrid it to the rest
                # grids, and check whether we get the anticipated results.
                c₁ = 1
                c₂ = 3

                @allowscalar begin
                    interior(fine_1d_stretched_c)[1] = c₁
                    interior(fine_1d_stretched_c)[2] = c₂
                end

                # Coarse-graining
                regrid!(coarse_1d_regular_c, fine_1d_stretched_c)

                @allowscalar begin
                    @test interior(coarse_1d_regular_c)[1] ≈ ℓ/L * c₁ + (1 - ℓ/L) * c₂
                end

                regrid!(fine_1d_regular_c, fine_1d_stretched_c)

                @allowscalar begin
                    @test interior(fine_1d_regular_c)[1] ≈ ℓ/(L/2) * c₁ + (1 - ℓ/(L/2)) * c₂
                    @test interior(fine_1d_regular_c)[2] ≈ c₂
                end

                # Fine-graining
                regrid!(very_fine_1d_stretched_c, fine_1d_stretched_c)

                @allowscalar begin
                    @test interior(very_fine_1d_stretched_c)[1] ≈ c₁
                    @test interior(very_fine_1d_stretched_c)[2] ≈ (ℓ - 0.2)/0.4 * c₁ + (0.6 - ℓ)/0.4 * c₂
                    @test interior(very_fine_1d_stretched_c)[3] ≈ c₂
                end

                regrid!(super_fine_1d_stretched_c, fine_1d_stretched_c)

                @allowscalar begin
                    @test interior(super_fine_1d_stretched_c)[1] ≈ c₁
                    @test interior(super_fine_1d_stretched_c)[2] ≈ c₁
                    @test interior(super_fine_1d_stretched_c)[3] ≈ (ℓ - 0.3)/0.35 * c₁ + (0.65 - ℓ)/0.35 * c₂
                    @test interior(super_fine_1d_stretched_c)[4] ≈ c₂
                end

                regrid!(super_fine_1d_regular_c, fine_1d_stretched_c)

                @allowscalar begin
                    @test interior(super_fine_1d_regular_c)[1] ≈ c₁
                    @test interior(super_fine_1d_regular_c)[2] ≈ c₁
                    @test interior(super_fine_1d_regular_c)[3] ≈ (3 - ℓ/(L/5)) * c₂ + (-2 + ℓ/(L/5)) * c₁
                    @test interior(super_fine_1d_regular_c)[4] ≈ c₂
                    @test interior(super_fine_1d_regular_c)[5] ≈ c₂
                end

                #=
                # This test does not work, because we can only regrid in one direction.
                # To make this work, we have to transfer the reduced data to a "reduced" grid
                # (ie with one grid point in each reduced direction).

                # Fine-graining from reduction
                ind1 = dim == :x ? (1, :, :) : dim == :y ? (:, 1, :) : (:, :, 1)
                ind2 = dim == :x ? (2, :, :) : dim == :y ? (:, 2, :) : (:, :, 2)
                dims = dim == :x ? (2, 3) : dim == :y ? (1, 3) : (1, 2)

                Base.dotview(fine_stretched_c, ind1...) .= c₁
                Base.dotview(fine_stretched_c, ind2...) .= c₂

                fine_stretched_c_mean_xy = Field(Reduction(mean!, fine_stretched_c; dims))
                compute!(fine_stretched_c_mean_xy)

                @show size(fine_stretched_c_mean_xy.grid)
                @show size(super_fine_from_reduction_regular_c.grid)

                regrid!(super_fine_from_reduction_regular_c, fine_stretched_c_mean_xy)

                @allowscalar begin
                    @test interior(super_fine_from_reduction_regular_c)[1] ≈ c₁
                    @test interior(super_fine_from_reduction_regular_c)[2] ≈ c₁
                    @test interior(super_fine_from_reduction_regular_c)[3] ≈ (3 - ℓ/(L/5)) * c₂ + (-2 + ℓ/(L/5)) * c₁
                    @test interior(super_fine_from_reduction_regular_c)[4] ≈ c₂
                    @test interior(super_fine_from_reduction_regular_c)[5] ≈ c₂
                end
                =#
            end
        end
    end
end

overlap_length(left1, right1, left2, right2) = max(0, min(right1, right2) - max(left1, left2))

dimension_face_coordinates(grid::RectilinearGrid) =
    (xnodes(grid, Face()), ynodes(grid, Face()), znodes(grid, Face()))

dimension_face_coordinates(grid::LatitudeLongitudeGrid) =
    (λnodes(grid, Face()), sind.(φnodes(grid, Face())), znodes(grid, Face()))

collected_faces(grid) = Tuple(faces === nothing ? [0, 1] : collect(faces)
                              for faces in dimension_face_coordinates(grid))

# Reference multi-dimensional conservative regridding by brute-force cell-overlap integrals.
# Cell volumes factorize per dimension on RectilinearGrid (Δx Δy Δz) and LatitudeLongitudeGrid
# (R² Δλ Δsinφ Δz, constant factors cancel in the volume-weighted average).
function brute_force_regrid(target_grid, source_grid, c)
    target_faces = collected_faces(target_grid)
    source_faces = collected_faces(source_grid)

    Nx, Ny, Nz = size(target_grid)
    expected = zeros(Nx, Ny, Nz)

    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        target_intervals = ((target_faces[1][i], target_faces[1][i+1]),
                            (target_faces[2][j], target_faces[2][j+1]),
                            (target_faces[3][k], target_faces[3][k+1]))

        integral = 0.0
        for i′ in 1:size(source_grid, 1), j′ in 1:size(source_grid, 2), k′ in 1:size(source_grid, 3)
            volume = overlap_length(target_intervals[1]..., source_faces[1][i′], source_faces[1][i′+1]) *
                     overlap_length(target_intervals[2]..., source_faces[2][j′], source_faces[2][j′+1]) *
                     overlap_length(target_intervals[3]..., source_faces[3][k′], source_faces[3][k′+1])
            integral += c[i′, j′, k′] * volume
        end

        target_volume = prod(interval[2] - interval[1] for interval in target_intervals)
        expected[i, j, k] = integral / target_volume
    end

    return expected
end

total_integral(field) = @allowscalar compute!(Field(Integral(field)))[1, 1, 1]

@testset "Multi-dimensional regridding" begin
    @info "  Testing multi-dimensional regridding..."

    for arch in archs
        @testset "Multi-dimensional regridding on RectilinearGrid [$(typeof(arch))]" begin
            source_grid = RectilinearGrid(arch, size=(4, 4, 4),
                                          x=[0, 0.1, 0.3, 0.65, 1.1],
                                          y=[0, 0.2, 0.5, 0.9, 1.4],
                                          z=[-1, -0.7, -0.45, -0.2, 0],
                                          topology=(Bounded, Bounded, Bounded))

            target_grid = RectilinearGrid(arch, size=(2, 3, 2), x=(0, 1.1), y=(0, 1.4), z=(-1, 0),
                                          topology=(Bounded, Bounded, Bounded))

            c = reshape(collect(1.0:64.0) .+ sin.(1:64) ./ 2, 4, 4, 4)
            source_field = CenterField(source_grid)
            set!(source_field, c)
            target_field = CenterField(target_grid)
            regrid!(target_field, source_field)

            expected = brute_force_regrid(target_grid, source_grid, c)
            @test all(isapprox.(Array(interior(target_field)), expected, atol=1e-12))
            @test total_integral(target_field) ≈ total_integral(source_field)

            # Refinement in all three dimensions reproduces piecewise-constant data
            fine_grid = RectilinearGrid(arch, size=(8, 9, 11), x=(0, 1.1), y=(0, 1.4), z=(-1, 0),
                                        topology=(Bounded, Bounded, Bounded))
            fine_field = CenterField(fine_grid)
            regrid!(fine_field, target_field)

            expected = brute_force_regrid(fine_grid, target_grid, Array(interior(target_field)))
            @test all(isapprox.(Array(interior(fine_field)), expected, atol=1e-12))
            @test total_integral(fine_field) ≈ total_integral(target_field)
        end

        @testset "Multi-dimensional regridding with Periodic dimensions [$(typeof(arch))]" begin
            source_grid = RectilinearGrid(arch, size=(5, 4, 3),
                                          x=[0, 0.1, 0.3, 0.35, 0.7, 1], y=(0, 2), z=[-1, -0.5, -0.2, 0],
                                          topology=(Periodic, Bounded, Bounded))

            target_grid = RectilinearGrid(arch, size=(3, 2, 5), x=(0, 1), y=(0, 2), z=(-1, 0),
                                          topology=(Periodic, Bounded, Bounded))

            source_field = CenterField(source_grid)
            set!(source_field, 3.7)
            target_field = CenterField(target_grid)
            regrid!(target_field, source_field)

            @test all(isapprox.(Array(interior(target_field)), 3.7))
        end

        @testset "Multi-dimensional regridding on LatitudeLongitudeGrid [$(typeof(arch))]" begin
            source_grid = LatitudeLongitudeGrid(arch, size=(8, 8, 4),
                                                longitude=(0, 40), latitude=(10, 50), z=[-1, -0.6, -0.3, -0.1, 0],
                                                topology=(Bounded, Bounded, Bounded))

            target_grid = LatitudeLongitudeGrid(arch, size=(4, 5, 2),
                                                longitude=(0, 40), latitude=(10, 50), z=(-1, 0),
                                                topology=(Bounded, Bounded, Bounded))

            c = reshape(collect(1.0:256.0), 8, 8, 4) ./ 256 .+ 1
            source_field = CenterField(source_grid)
            set!(source_field, c)
            target_field = CenterField(target_grid)
            regrid!(target_field, source_field)

            expected = brute_force_regrid(target_grid, source_grid, c)
            @test all(isapprox.(Array(interior(target_field)), expected, atol=1e-12))
            @test total_integral(target_field) ≈ total_integral(source_field)

            constant_source_field = CenterField(source_grid)
            set!(constant_source_field, 2.5)
            constant_target_field = CenterField(target_grid)
            regrid!(constant_target_field, constant_source_field)
            @test all(isapprox.(Array(interior(constant_target_field)), 2.5))

            # Horizontal-only regridding of a two-dimensional field
            source_grid = LatitudeLongitudeGrid(arch, size=(6, 6), longitude=(-30, 30), latitude=(-20, 40),
                                                topology=(Bounded, Bounded, Flat))
            target_grid = LatitudeLongitudeGrid(arch, size=(3, 2), longitude=(-30, 30), latitude=(-20, 40),
                                                topology=(Bounded, Bounded, Flat))

            c = reshape(collect(1.0:36.0), 6, 6, 1)
            source_field = CenterField(source_grid)
            set!(source_field, c)
            target_field = CenterField(target_grid)
            regrid!(target_field, source_field)

            expected = brute_force_regrid(target_grid, source_grid, c)
            @test all(isapprox.(Array(interior(target_field)), expected, atol=1e-12))
        end

        @testset "Unsupported multi-dimensional regridding [$(typeof(arch))]" begin
            rectilinear_grid = RectilinearGrid(arch, size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1),
                                               topology=(Bounded, Bounded, Bounded))
            latlon_grid = LatitudeLongitudeGrid(arch, size=(2, 2, 2), longitude=(0, 1), latitude=(0, 1), z=(0, 1),
                                                topology=(Bounded, Bounded, Bounded))

            latlon_field = CenterField(latlon_grid)
            rectilinear_field = CenterField(rectilinear_grid)
            @test_throws ArgumentError regrid!(latlon_field, rectilinear_field)
        end
    end
end
