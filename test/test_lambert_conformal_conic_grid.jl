include("dependencies_for_runtests.jl")

using CUDA
using Oceananigans.OrthogonalSphericalShellGrids: LambertConformalConic,
    LambertConformalConicGrid, lcc_forward, lcc_inverse, lcc_scale_factor,
    lcc_xnode, lcc_ynode, spherical_distance, spherical_unit_vector,
    spherical_quadrilateral_area, fill_lcc_coordinates_and_metrics!
using Oceananigans.Grids: architecture, constructor_arguments, topology, halo_size,
                          with_halo, with_number_type, znodes
using Oceananigans.Fields: interior
using Oceananigans.Operators: intrinsic_vector, extrinsic_vector, rotation_angle
using Oceananigans.Architectures: on_architecture
using Adapt: adapt, adapt_structure

function unit_vector_from_degrees(λ, φ)
    λ = deg2rad(λ)
    φ = deg2rad(φ)
    return (cos(λ) * cos(φ), sin(λ) * cos(φ), sin(φ))
end

dot_tuple(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

function normalized_projected_tangent(a, b, n)
    tangent = (b[1] - a[1], b[2] - a[2], b[3] - a[3])
    normal_component = dot_tuple(tangent, n)
    tangent = (tangent[1] - normal_component * n[1],
               tangent[2] - normal_component * n[2],
               tangent[3] - normal_component * n[3])
    tangent_norm = sqrt(dot_tuple(tangent, tangent))
    return tangent ./ tangent_norm
end

lcc_coordinate_arrays(grid) =
    (grid.λᶜᶜᵃ, grid.λᶠᶜᵃ, grid.λᶜᶠᵃ, grid.λᶠᶠᵃ,
     grid.φᶜᶜᵃ, grid.φᶠᶜᵃ, grid.φᶜᶠᵃ, grid.φᶠᶠᵃ)

lcc_metric_arrays(grid) =
    (grid.Δxᶜᶜᵃ, grid.Δxᶠᶜᵃ, grid.Δxᶜᶠᵃ, grid.Δxᶠᶠᵃ,
     grid.Δyᶜᶜᵃ, grid.Δyᶠᶜᵃ, grid.Δyᶜᶠᵃ, grid.Δyᶠᶠᵃ,
     grid.Azᶜᶜᵃ, grid.Azᶠᶜᵃ, grid.Azᶜᶠᵃ, grid.Azᶠᶠᵃ)

function throws_argument_error_matching(f, pattern)
    err = try
        f()
        nothing
    catch err
        err
    end

    @test err isa ArgumentError

    if err isa ArgumentError
        @test occursin(pattern, sprint(showerror, err))
    end

    return nothing
end

@testset "LambertConformalConicGrid" begin
    @testset "projection math" begin
        for FT in float_types
            map = LambertConformalConic(FT;
                                        standard_parallels = (30, 60),
                                        central_longitude = -105,
                                        latitude_of_origin = 40,
                                        x₁ = -1e6,
                                        y₁ = -1e6,
                                        Δx = 10e3,
                                        Δy = 10e3)

            @test isbitstype(typeof(map))
            @test @inferred(adapt(identity, map)) isa typeof(map)
            @test adapt(identity, map) == map
            @test @inferred(adapt_structure(identity, map)) isa typeof(map)
            @test adapt_structure(identity, map) == map
            @test occursin("LambertConformalConic", sprint(show, map))
            @test occursin("standard_parallels", sprint(show, map))
            @test @inferred(lcc_forward(map, convert(FT, -100), convert(FT, 45))) isa Tuple{FT, FT}
            x, y = lcc_forward(map, convert(FT, -100), convert(FT, 45))
            @test @inferred(lcc_inverse(map, x, y)) isa Tuple{FT, FT}
            @test @inferred(lcc_scale_factor(map, convert(FT, 30))) isa FT
            @test @inferred(lcc_xnode(1, Center(), map)) isa FT
            @test @inferred(lcc_xnode(1, Face(), map)) isa FT
            @test @inferred(lcc_ynode(1, Center(), map)) isa FT
            @test @inferred(lcc_ynode(1, Face(), map)) isa FT
            @test @inferred(spherical_distance(convert(FT, -105),
                                               convert(FT, 40),
                                               convert(FT, -104),
                                               convert(FT, 41),
                                               map.radius)) isa FT
            @test @inferred(spherical_unit_vector(convert(FT, -105),
                                                 convert(FT, 40),
                                                 FT)) isa Tuple{FT, FT, FT}
            @test @inferred(spherical_quadrilateral_area(convert(FT, -105),
                                                         convert(FT, 40),
                                                         convert(FT, -104),
                                                         convert(FT, 40),
                                                         convert(FT, -104),
                                                         convert(FT, 41),
                                                         convert(FT, -105),
                                                         convert(FT, 41),
                                                         map.radius)) isa FT

            lcc_forward(map, convert(FT, -100), convert(FT, 45))
            lcc_inverse(map, x, y)
            lcc_scale_factor(map, convert(FT, 30))
            spherical_distance(convert(FT, -105),
                               convert(FT, 40),
                               convert(FT, -104),
                               convert(FT, 41),
                               map.radius)
            spherical_quadrilateral_area(convert(FT, -105),
                                         convert(FT, 40),
                                         convert(FT, -104),
                                         convert(FT, 40),
                                         convert(FT, -104),
                                         convert(FT, 41),
                                         convert(FT, -105),
                                         convert(FT, 41),
                                         map.radius)

            @test (@allocated lcc_forward(map, convert(FT, -100), convert(FT, 45))) == 0
            @test (@allocated lcc_inverse(map, x, y)) == 0
            @test (@allocated lcc_scale_factor(map, convert(FT, 30))) == 0
            @test (@allocated spherical_distance(convert(FT, -105),
                                                 convert(FT, 40),
                                                 convert(FT, -104),
                                                 convert(FT, 41),
                                                 map.radius)) == 0
            @test (@allocated spherical_quadrilateral_area(convert(FT, -105),
                                                           convert(FT, 40),
                                                           convert(FT, -104),
                                                           convert(FT, 40),
                                                           convert(FT, -104),
                                                           convert(FT, 41),
                                                           convert(FT, -105),
                                                           convert(FT, 41),
                                                           map.radius)) == 0

            offset_map = LambertConformalConic(FT;
                                               standard_parallels = (30, 60),
                                               central_longitude = -105,
                                               latitude_of_origin = 40,
                                               false_easting = 100,
                                               false_northing = -200,
                                               x₁ = -1e6,
                                               y₁ = -1e6,
                                               Δx = 10e3,
                                               Δy = 10e3)

            coordinate_tolerance = FT === Float64 ? 1e-10 : 1e-4
            scale_tolerance = FT === Float64 ? 1e-10 : 1e-5

            x_origin, y_origin = lcc_forward(offset_map, -105, 40)
            @test x_origin ≈ 100 atol=coordinate_tolerance
            @test y_origin ≈ -200 atol=coordinate_tolerance

            for λ in -115:5:-95, φ in 30:5:55
                x, y = lcc_forward(map, λ, φ)
                λ′, φ′ = lcc_inverse(map, x, y)

                @test λ′ ≈ λ atol=coordinate_tolerance
                @test φ′ ≈ φ atol=coordinate_tolerance
            end

            @test lcc_scale_factor(map, 30) ≈ 1 atol=scale_tolerance
            @test lcc_scale_factor(map, 60) ≈ 1 atol=scale_tolerance

            λ_apex, φ_apex = lcc_inverse(map, map.false_easting, map.false_northing + map.origin_radius)
            @test λ_apex ≈ -105 atol=coordinate_tolerance
            @test φ_apex ≈ 90 atol=coordinate_tolerance

            southern_map = LambertConformalConic(FT;
                                                 standard_parallels = (-60, -30),
                                                 central_longitude = 30,
                                                 latitude_of_origin = -40,
                                                 x₁ = -1e6,
                                                 y₁ = -1e6,
                                                 Δx = 10e3,
                                                 Δy = 10e3)

            λ_apex, φ_apex = lcc_inverse(southern_map,
                                         southern_map.false_easting,
                                         southern_map.false_northing + southern_map.origin_radius)

            @test λ_apex ≈ 30 atol=coordinate_tolerance
            @test φ_apex ≈ -90 atol=coordinate_tolerance

            for λ in 20:5:40, φ in -55:5:-30
                x, y = lcc_forward(southern_map, λ, φ)
                λ′, φ′ = lcc_inverse(southern_map, x, y)

                @test λ′ ≈ λ atol=coordinate_tolerance
                @test φ′ ≈ φ atol=coordinate_tolerance
            end

            tangent_map = LambertConformalConic(FT;
                                                standard_parallel = 45,
                                                central_longitude = -105,
                                                latitude_of_origin = 40,
                                                x₁ = -1e6,
                                                y₁ = -1e6,
                                                Δx = 10e3,
                                                Δy = 10e3)

            @test tangent_map.cone_constant ≈ sind(45) rtol=scale_tolerance

            numeric_parallels_map = LambertConformalConic(FT;
                                                          standard_parallels = 45,
                                                          central_longitude = -105,
                                                          latitude_of_origin = 40,
                                                          x₁ = -1e6,
                                                          y₁ = -1e6,
                                                          Δx = 10e3,
                                                          Δy = 10e3)

            @test numeric_parallels_map.standard_parallel_1 ≈ tangent_map.standard_parallel_1
            @test numeric_parallels_map.standard_parallel_2 ≈ tangent_map.standard_parallel_2
            @test numeric_parallels_map.cone_constant ≈ tangent_map.cone_constant
        end
    end

    @testset "polar stereographic limit" begin
        for FT in float_types
            tol = FT === Float64 ? 1e-12 : 1e-6

            north_polar = LambertConformalConic(FT;
                                                standard_parallel = 90,
                                                central_longitude = 0,
                                                latitude_of_origin = 90,
                                                x₁ = -1e6, y₁ = -1e6,
                                                Δx = 10e3, Δy = 10e3)

            @test north_polar.cone_constant ≈ +one(FT) atol = tol
            @test north_polar.scale_constant ≈ +convert(FT, 2) atol = tol
            @test isfinite(north_polar.origin_radius)
            @test isfinite(lcc_scale_factor(north_polar, FT(89)))
            @test lcc_scale_factor(north_polar, FT(90)) ≈ one(FT) atol = tol

            south_polar = LambertConformalConic(FT;
                                                standard_parallel = -90,
                                                central_longitude = 0,
                                                latitude_of_origin = -90,
                                                x₁ = -1e6, y₁ = -1e6,
                                                Δx = 10e3, Δy = 10e3)

            @test south_polar.cone_constant ≈ -one(FT) atol = tol
            @test south_polar.scale_constant ≈ -convert(FT, 2) atol = tol
            @test isfinite(south_polar.origin_radius)
            @test lcc_scale_factor(south_polar, FT(-90)) ≈ one(FT) atol = tol

            tuple_north = LambertConformalConic(FT;
                                                standard_parallels = (90, 90),
                                                central_longitude = 0,
                                                latitude_of_origin = 90,
                                                x₁ = -1e6, y₁ = -1e6,
                                                Δx = 10e3, Δy = 10e3)

            @test tuple_north.cone_constant ≈ +one(FT)
            @test tuple_north.scale_constant ≈ +convert(FT, 2)

            # Pole-centred grid round-trips cleanly
            grid = LambertConformalConicGrid(CPU(), FT;
                                             size = (16, 16, 1),
                                             center = (0, 90),
                                             spacing = 25e3,
                                             standard_parallel = 90,
                                             latitude_of_origin = 90,
                                             z = (-100, 0))

            @test grid isa LambertConformalConicGrid
            @test grid.conformal_mapping.cone_constant ≈ +one(FT)
            @test grid.conformal_mapping.scale_constant ≈ +convert(FT, 2)
            for array in lcc_coordinate_arrays(grid)
                @test all(isfinite, array)
            end
            for array in lcc_metric_arrays(grid)
                @test all(isfinite, array)
            end

            # Mixing a polar parallel with a non-polar one is rejected
            throws_argument_error_matching("polar stereographic limit") do
                LambertConformalConic(FT;
                                      standard_parallels = (90, 80),
                                      central_longitude = 0,
                                      latitude_of_origin = 90,
                                      x₁ = -1e6, y₁ = -1e6,
                                      Δx = 10e3, Δy = 10e3)
            end

            # Parallels at opposite poles is also rejected
            throws_argument_error_matching("polar stereographic limit") do
                LambertConformalConic(FT;
                                      standard_parallels = (90, -90),
                                      central_longitude = 0,
                                      latitude_of_origin = 0,
                                      x₁ = -1e6, y₁ = -1e6,
                                      Δx = 10e3, Δy = 10e3)
            end
        end

        # rotation_angle on a polar-centred LCC grid must span the full (-π, π]
        # range. Single-argument atan in the source would have clipped to
        # (-π/2, π/2] and sign-flipped half the grid.
        @testset "polar rotation_angle range" begin
            grid = LambertConformalConicGrid(CPU(), Float64;
                                             size = (16, 16, 1),
                                             center = (0, 90),
                                             spacing = 50e3,
                                             standard_parallel = 90,
                                             latitude_of_origin = 90,
                                             z = (-100, 0))

            Nx, Ny, _ = size(grid)
            θs = [rotation_angle(i, j, grid) for j in 1:Ny, i in 1:Nx]
            @test maximum(θs) > π/2 + 0.1
            @test minimum(θs) < -π/2 - 0.1
            @test all(isfinite, θs)
        end

        # Intrinsic/extrinsic vector conversion roundtrips to machine precision
        # everywhere on a polar-centred LCC grid, including cells whose rotation
        # angle falls outside (-π/2, π/2].
        @testset "polar intrinsic ↔ extrinsic roundtrip" begin
            grid = LambertConformalConicGrid(CPU(), Float64;
                                             size = (12, 12, 1),
                                             center = (0, 90),
                                             spacing = 50e3,
                                             standard_parallel = 90,
                                             latitude_of_origin = 90,
                                             z = (-100, 0))

            u_in, v_in = 1.234, -2.567
            for j in (3, 6, 9), i in (3, 6, 9)
                u_e, v_e = extrinsic_vector(i, j, 1, grid, u_in, v_in)
                u_back, v_back = intrinsic_vector(i, j, 1, grid, u_e, v_e)
                @test u_back ≈ u_in atol = 1e-10
                @test v_back ≈ v_in atol = 1e-10
            end
        end

        # Polar grid round-trips through every helper that downstream code uses
        # to derive related grids from an existing one.
        @testset "polar with_halo / similar / with_number_type / reconstruction" begin
            grid = LambertConformalConicGrid(CPU(), Float64;
                                             size = (16, 16, 1),
                                             center = (0, 90),
                                             spacing = 50e3,
                                             standard_parallel = 90,
                                             latitude_of_origin = 90,
                                             z = (-100, 0))

            @test grid.conformal_mapping.cone_constant ≈ 1.0
            @test grid.conformal_mapping.scale_constant ≈ 2.0

            grid_h = with_halo((5, 5, 5), grid)
            @test grid_h.conformal_mapping.cone_constant ≈ 1.0
            @test grid_h.conformal_mapping.scale_constant ≈ 2.0
            @test halo_size(grid_h) == (5, 5, 5)

            similar_grid = similar(grid)
            @test similar_grid.conformal_mapping.cone_constant ≈ 1.0
            @test similar_grid.conformal_mapping.scale_constant ≈ 2.0

            float32_grid = with_number_type(Float32, grid)
            @test float32_grid.conformal_mapping.cone_constant ≈ Float32(1)
            @test float32_grid.conformal_mapping.scale_constant ≈ Float32(2)

            args, kwargs = constructor_arguments(grid)
            reconstructed = LambertConformalConicGrid(args[:architecture],
                                                      args[:number_type]; kwargs...)
            @test reconstructed.conformal_mapping.cone_constant ≈ 1.0
            @test reconstructed.conformal_mapping.scale_constant ≈ 2.0
        end

        # Make sure a hydrostatic model can actually be integrated on a polar
        # grid (analogous to the midlatitude smoke test below).
        @testset "polar HFSM smoke test" begin
            grid = LambertConformalConicGrid(CPU(), Float64;
                                             size = (12, 12, 3),
                                             center = (0, 90),
                                             spacing = 25e3,
                                             standard_parallel = 90,
                                             latitude_of_origin = 90,
                                             z = (-100, 0),
                                             halo = (3, 3, 3))

            model = HydrostaticFreeSurfaceModel(grid;
                                                coriolis = HydrostaticSphericalCoriolis(),
                                                free_surface = SplitExplicitFreeSurface(grid;
                                                                                       substeps = 5))

            simulation = Simulation(model; Δt = 60, stop_iteration = 3)
            run!(simulation)

            @test isfinite(time(simulation))
            @test all(isfinite, interior(model.velocities.u))
            @test all(isfinite, interior(model.velocities.v))
            @test all(isfinite, interior(model.free_surface.displacement))
        end
    end

    @testset "constructors and validation" begin
        @test :LambertConformalConicGrid in names(Oceananigans)
        @test :LambertConformalConic in names(Oceananigans)
        @test :lcc_forward in names(Oceananigans)
        @test :lcc_inverse in names(Oceananigans)
        @test :lcc_scale_factor in names(Oceananigans)
        @test :LambertConformalConicGrid in names(Oceananigans.OrthogonalSphericalShellGrids)
        @test :LambertConformalConic in names(Oceananigans.OrthogonalSphericalShellGrids)
        @test :lcc_forward in names(Oceananigans.OrthogonalSphericalShellGrids)
        @test :lcc_inverse in names(Oceananigans.OrthogonalSphericalShellGrids)
        @test :lcc_scale_factor in names(Oceananigans.OrthogonalSphericalShellGrids)

        base_kwargs = (size = (16, 12, 4),
                       standard_parallels = (30, 60),
                       z = (-100, 0))

        grid = LambertConformalConicGrid(CPU(), Float64;
                                         center = (-105, 40),
                                         spacing = 20e3,
                                         base_kwargs...)

        @test grid isa LambertConformalConicGrid
        @test grid.conformal_mapping isa LambertConformalConic
        @test size(grid) == (16, 12, 4)
        @test topology(grid) == (Bounded, Bounded, Bounded)
        @test eltype(grid) == Float64
        @test grid.conformal_mapping.Δx ≈ 20e3
        @test grid.conformal_mapping.Δy ≈ 20e3
        @test grid.conformal_mapping.central_longitude ≈ deg2rad(-105)
        @test grid.conformal_mapping.latitude_of_origin ≈ deg2rad(40)

        x_center, y_center = lcc_forward(grid.conformal_mapping, -105, 40)
        x_domain_center = grid.conformal_mapping.x₁ + size(grid, 1) * grid.conformal_mapping.Δx / 2
        y_domain_center = grid.conformal_mapping.y₁ + size(grid, 2) * grid.conformal_mapping.Δy / 2

        @test x_domain_center ≈ x_center
        @test y_domain_center ≈ y_center

        grid = LambertConformalConicGrid(; center = (-105, 40),
                                           spacing = 20e3,
                                           base_kwargs...)

        @test grid isa LambertConformalConicGrid
        @test architecture(grid) == CPU()
        @test eltype(grid) == Oceananigans.defaults.FloatType

        grid = LambertConformalConicGrid(Float32;
                                         center = (-105, 40),
                                         spacing = 20e3,
                                         base_kwargs...)

        @test grid isa LambertConformalConicGrid
        @test architecture(grid) == CPU()
        @test eltype(grid) == Float32

        grid = LambertConformalConicGrid(CPU(), Float32;
                                         center = (-105, 40),
                                         extent = (320e3, 240e3),
                                         base_kwargs...)

        @test grid isa LambertConformalConicGrid
        @test eltype(grid) == Float32
        @test grid.conformal_mapping.Δx ≈ Float32(320e3 / 16)
        @test grid.conformal_mapping.Δy ≈ Float32(240e3 / 12)

        grid = LambertConformalConicGrid(CPU(), Float64;
                                         x = (-160e3, 160e3),
                                         y = (-120e3, 120e3),
                                         central_longitude = -105,
                                         latitude_of_origin = 40,
                                         base_kwargs...)

        @test grid isa LambertConformalConicGrid
        @test grid.conformal_mapping.x₁ ≈ -160e3
        @test grid.conformal_mapping.y₁ ≈ -120e3
        @test grid.conformal_mapping.Δx ≈ 320e3 / 16
        @test grid.conformal_mapping.Δy ≈ 240e3 / 12

        offset_grid = LambertConformalConicGrid(CPU(), Float64;
                                                size = (8, 6, 2),
                                                x = (450e3, 530e3),
                                                y = (-220e3, -160e3),
                                                standard_parallels = (33, 45),
                                                central_longitude = -97,
                                                latitude_of_origin = 40,
                                                false_easting = 500e3,
                                                false_northing = -200e3,
                                                z = (-10, 0),
                                                halo = (2, 2, 2))

        @test offset_grid.conformal_mapping.false_easting ≈ 500e3
        @test offset_grid.conformal_mapping.false_northing ≈ -200e3
        @test offset_grid.conformal_mapping.x₁ ≈ 450e3
        @test offset_grid.conformal_mapping.y₁ ≈ -220e3
        @test offset_grid.conformal_mapping.Δx ≈ 80e3 / 8
        @test offset_grid.conformal_mapping.Δy ≈ 60e3 / 6

        x_origin, y_origin = lcc_forward(offset_grid.conformal_mapping, -97, 40)

        @test x_origin ≈ 500e3
        @test y_origin ≈ -200e3

        args, kwargs = constructor_arguments(offset_grid)
        reconstructed_offset_grid = LambertConformalConicGrid(args[:architecture], args[:number_type]; kwargs...)
        offset_grid_h4 = with_halo((4, 4, 4), offset_grid)
        similar_offset_grid = similar(offset_grid)
        float32_offset_grid = with_number_type(Float32, offset_grid)

        @test reconstructed_offset_grid.conformal_mapping == offset_grid.conformal_mapping
        @test offset_grid_h4.conformal_mapping.false_easting ≈ offset_grid.conformal_mapping.false_easting
        @test offset_grid_h4.conformal_mapping.false_northing ≈ offset_grid.conformal_mapping.false_northing
        @test offset_grid_h4.conformal_mapping.x₁ ≈ offset_grid.conformal_mapping.x₁
        @test offset_grid_h4.conformal_mapping.y₁ ≈ offset_grid.conformal_mapping.y₁
        @test similar_offset_grid.conformal_mapping == offset_grid.conformal_mapping
        @test eltype(float32_offset_grid) == Float32
        @test float32_offset_grid.conformal_mapping.false_easting ≈ offset_grid.conformal_mapping.false_easting
        @test float32_offset_grid.conformal_mapping.false_northing ≈ offset_grid.conformal_mapping.false_northing
        @test float32_offset_grid.conformal_mapping.x₁ ≈ offset_grid.conformal_mapping.x₁
        @test float32_offset_grid.conformal_mapping.y₁ ≈ offset_grid.conformal_mapping.y₁

        grid = LambertConformalConicGrid(CPU(), Float64;
                                         center = (-105, 40),
                                         spacing = (20e3, 30e3),
                                         standard_parallels = 45,
                                         size = (16, 12, 4),
                                         z = (-100, 0))

        @test grid.conformal_mapping.standard_parallel_1 ≈ grid.conformal_mapping.standard_parallel_2

        polar_origin_grid = LambertConformalConicGrid(CPU(), Float64;
                                                      center = (0, 89),
                                                      spacing = 20e3,
                                                      standard_parallels = (80, 85),
                                                      central_longitude = 0,
                                                      latitude_of_origin = 90,
                                                      size = (8, 8, 1),
                                                      z = (-100, 0))

        @test isfinite(polar_origin_grid.conformal_mapping.origin_radius)

        flat_grid = LambertConformalConicGrid(CPU(), Float64;
                                              size = (8, 6),
                                              center = (-105, 40),
                                              spacing = 20e3,
                                              standard_parallels = (30, 60),
                                              topology = (Bounded, Bounded, Flat),
                                              z = nothing)

        flat_grid_h2 = with_halo((2, 2), flat_grid)
        flat_grid_h2_from_inflated_halo = with_halo((2, 2, 0), flat_grid)
        similar_flat_grid = similar(flat_grid)
        float32_flat_grid = with_number_type(Float32, flat_grid)
        args, kwargs = constructor_arguments(flat_grid)
        reconstructed_flat_grid = LambertConformalConicGrid(args[:architecture], args[:number_type]; kwargs...)

        @test flat_grid isa LambertConformalConicGrid
        @test size(flat_grid) == (8, 6, 1)
        @test topology(flat_grid) == (Bounded, Bounded, Flat)
        @test halo_size(flat_grid) == (3, 3, 0)
        @test flat_grid.conformal_mapping.Δx ≈ 20e3
        @test flat_grid.conformal_mapping.Δy ≈ 20e3
        @test topology(flat_grid_h2) == topology(flat_grid)
        @test halo_size(flat_grid_h2) == (2, 2, 0)
        @test topology(flat_grid_h2_from_inflated_halo) == topology(flat_grid)
        @test halo_size(flat_grid_h2_from_inflated_halo) == (2, 2, 0)
        @test flat_grid_h2_from_inflated_halo.conformal_mapping == flat_grid_h2.conformal_mapping
        @test similar_flat_grid isa LambertConformalConicGrid
        @test topology(similar_flat_grid) == topology(flat_grid)
        @test halo_size(similar_flat_grid) == halo_size(flat_grid)
        @test similar_flat_grid.conformal_mapping == flat_grid.conformal_mapping
        @test float32_flat_grid isa LambertConformalConicGrid
        @test eltype(float32_flat_grid) == Float32
        @test topology(float32_flat_grid) == topology(flat_grid)
        @test halo_size(float32_flat_grid) == halo_size(flat_grid)
        @test reconstructed_flat_grid.conformal_mapping == flat_grid.conformal_mapping
        @test topology(reconstructed_flat_grid) == topology(flat_grid)
        @test halo_size(reconstructed_flat_grid) == halo_size(flat_grid)

        throws_argument_error_matching("Specify exactly one domain mode") do
            LambertConformalConicGrid(; central_longitude = -105,
                                      latitude_of_origin = 40,
                                      base_kwargs...)
        end

        throws_argument_error_matching("Specify exactly one domain mode") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      x = (-1, 1), y = (-1, 1),
                                      spacing = 20e3,
                                      base_kwargs...)
        end

        throws_argument_error_matching("Specify exactly one domain mode") do
            LambertConformalConicGrid(; x = (-1, 1), y = (-1, 1),
                                      extent = (320e3, 240e3),
                                      central_longitude = -105,
                                      latitude_of_origin = 40,
                                      base_kwargs...)
        end

        throws_argument_error_matching("Specify exactly one domain mode") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      extent = (320e3, 240e3),
                                      spacing = 20e3,
                                      base_kwargs...)
        end

        throws_argument_error_matching("Specify exactly one domain mode") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      x = (-1, 1), y = (-1, 1),
                                      base_kwargs...)
        end

        throws_argument_error_matching("LambertConformalConicGrid requires both x and y") do
            LambertConformalConicGrid(; x = (-1, 1),
                                      central_longitude = -105,
                                      latitude_of_origin = 40,
                                      base_kwargs...)
        end

        throws_argument_error_matching("spacing entries must be positive") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = -20e3,
                                      base_kwargs...)
        end

        throws_argument_error_matching("spacing entries must be finite") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = Inf,
                                      base_kwargs...)
        end

        throws_argument_error_matching("extent entries must be positive") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      extent = (-320e3, 240e3),
                                      base_kwargs...)
        end

        throws_argument_error_matching("spacing must be a number or a 2-tuple convertible") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = (20e3, "bad"),
                                      base_kwargs...)
        end

        throws_argument_error_matching("x must be an increasing interval") do
            LambertConformalConicGrid(; x = (1, -1),
                                      y = (-1, 1),
                                      central_longitude = -105,
                                      latitude_of_origin = 40,
                                      base_kwargs...)
        end

        throws_argument_error_matching("y must be an increasing interval") do
            LambertConformalConicGrid(; x = (-1, 1),
                                      y = (1, -1),
                                      central_longitude = -105,
                                      latitude_of_origin = 40,
                                      base_kwargs...)
        end

        throws_argument_error_matching("center entries must be convertible") do
            LambertConformalConicGrid(; center = (-105, "bad"),
                                      spacing = 20e3,
                                      base_kwargs...)
        end

        throws_argument_error_matching("radius must be positive") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      radius = -1,
                                      base_kwargs...)
        end

        throws_argument_error_matching("central_longitude must be finite") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      central_longitude = Inf,
                                      base_kwargs...)
        end

        throws_argument_error_matching("x entries must be finite") do
            LambertConformalConicGrid(; x = (0, Inf),
                                      y = (-1, 1),
                                      central_longitude = -105,
                                      latitude_of_origin = 40,
                                      base_kwargs...)
        end

        throws_argument_error_matching("center latitude must lie between") do
            LambertConformalConicGrid(; center = (-105, 91),
                                      spacing = 20e3,
                                      latitude_of_origin = 40,
                                      base_kwargs...)
        end

        throws_argument_error_matching("latitude_of_origin must lie between") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      standard_parallels = (30, 60),
                                      latitude_of_origin = 91,
                                      size = (16, 12, 4),
                                      z = (-100, 0))
        end

        throws_argument_error_matching("central_longitude is required") do
            LambertConformalConicGrid(; x = (-1, 1),
                                      y = (-1, 1),
                                      latitude_of_origin = 40,
                                      base_kwargs...)
        end

        throws_argument_error_matching("latitude_of_origin is required") do
            LambertConformalConicGrid(; x = (-1, 1),
                                      y = (-1, 1),
                                      central_longitude = -105,
                                      base_kwargs...)
        end

        throws_argument_error_matching("standard parallels cannot be symmetric") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      standard_parallels = (30, -30),
                                      size = (16, 12, 4),
                                      z = (-100, 0))
        end

        throws_argument_error_matching("Specify either standard_parallel or standard_parallels") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      standard_parallel = 45,
                                      standard_parallels = (30, 60),
                                      size = (16, 12, 4),
                                      z = (-100, 0))
        end

        throws_argument_error_matching("standard_parallels must be a number or a 2-tuple") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      standard_parallels = [30, 60],
                                      size = (16, 12, 4),
                                      z = (-100, 0))
        end

        throws_argument_error_matching("standard parallels must be convertible") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      standard_parallel = "bad",
                                      size = (16, 12, 4),
                                      z = (-100, 0))
        end

        throws_argument_error_matching("latitude_of_origin must be convertible") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      standard_parallel = 45,
                                      latitude_of_origin = "bad",
                                      size = (16, 12, 4),
                                      z = (-100, 0))
        end

        throws_argument_error_matching("standard parallels cannot be symmetric") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      standard_parallel = 0,
                                      size = (16, 12, 4),
                                      z = (-100, 0))
        end

        throws_argument_error_matching("requires Bounded topology in x") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      topology = (Periodic, Bounded, Bounded),
                                      base_kwargs...)
        end

        throws_argument_error_matching("requires Bounded topology in y") do
            LambertConformalConicGrid(; center = (-105, 40),
                                      spacing = 20e3,
                                      topology = (Bounded, Periodic, Bounded),
                                      base_kwargs...)
        end

        apex_map = LambertConformalConic(Float64;
                                         standard_parallel = 45,
                                         central_longitude = 0,
                                         latitude_of_origin = 45,
                                         x₁ = 0, y₁ = 0,
                                         Δx = 1, Δy = 1)

        apex_y = apex_map.false_northing + apex_map.origin_radius

        @test_logs (:warn, r"cone apex / pole on a grid node") LambertConformalConicGrid(CPU(), Float64;
                                                                                         size = (2, 2, 1),
                                                                                         x = (-1, 1),
                                                                                         y = (apex_y - 1, apex_y + 1),
                                                                                         standard_parallel = 45,
                                                                                         central_longitude = 0,
                                                                                         latitude_of_origin = 45,
                                                                                         halo = (1, 1, 1),
                                                                                         z = (-1, 0))

        @test_logs (:warn, r"cone apex") LambertConformalConicGrid(CPU(), Float64;
                                                                   size = (2, 2, 1),
                                                                   x = (-1.25, 0.75),
                                                                   y = (apex_y - 1.25, apex_y + 0.75),
                                                                   standard_parallel = 45,
                                                                   central_longitude = 0,
                                                                   latitude_of_origin = 45,
                                                                   halo = (1, 1, 1),
                                                                   z = (-1, 0))
    end

    @testset "coordinates, metrics, and with_halo" begin
        grid = LambertConformalConicGrid(CPU(), Float64;
                                         size = (48, 40, 3),
                                         center = (-105, 40),
                                         spacing = 2e3,
                                         standard_parallels = (30, 60),
                                         z = (-100, 0),
                                         halo = (4, 4, 4))

        map = grid.conformal_mapping

        locations = ((Center(), Center()), (Face(), Center()), (Center(), Face()), (Face(), Face()))
        λ_arrays = (grid.λᶜᶜᵃ, grid.λᶠᶜᵃ, grid.λᶜᶠᵃ, grid.λᶠᶠᵃ)
        φ_arrays = (grid.φᶜᶜᵃ, grid.φᶠᶜᵃ, grid.φᶜᶠᵃ, grid.φᶠᶠᵃ)

        for ((ℓx, ℓy), λ_array, φ_array) in zip(locations, λ_arrays, φ_arrays)
            for i in (-3, 1, 20, 48), j in (-3, 1, 17, 40)
                x = lcc_xnode(i, ℓx, map)
                y = lcc_ynode(j, ℓy, map)
                λ, φ = lcc_inverse(map, x, y)

                @test λ_array[i, j] ≈ λ atol=1e-10
                @test φ_array[i, j] ≈ φ atol=1e-10
            end
        end

        Nx, Ny, _ = size(grid)
        Hx, Hy, _ = halo_size(grid)
        operator_range = (-Hx:Nx+Hx+1, -Hy:Ny+Hy+1)

        for metric in (grid.Δxᶜᶜᵃ, grid.Δxᶠᶜᵃ, grid.Δxᶜᶠᵃ, grid.Δxᶠᶠᵃ,
                       grid.Δyᶜᶜᵃ, grid.Δyᶠᶜᵃ, grid.Δyᶜᶠᵃ, grid.Δyᶠᶠᵃ,
                       grid.Azᶜᶜᵃ, grid.Azᶠᶜᵃ, grid.Azᶜᶠᵃ, grid.Azᶠᶠᵃ)
            operator_metric = metric[operator_range...]
            @test all(isfinite, operator_metric)
            @test minimum(operator_metric) > 0
        end

        i, j = 24, 20

        expected_Azᶜᶜᵃ = spherical_quadrilateral_area(grid.λᶠᶠᵃ[i,   j  ], grid.φᶠᶠᵃ[i,   j  ],
                                                      grid.λᶠᶠᵃ[i+1, j  ], grid.φᶠᶠᵃ[i+1, j  ],
                                                      grid.λᶠᶠᵃ[i+1, j+1], grid.φᶠᶠᵃ[i+1, j+1],
                                                      grid.λᶠᶠᵃ[i,   j+1], grid.φᶠᶠᵃ[i,   j+1],
                                                      grid.radius)

        expected_Azᶠᶠᵃ = spherical_quadrilateral_area(grid.λᶜᶜᵃ[i-1, j-1], grid.φᶜᶜᵃ[i-1, j-1],
                                                      grid.λᶜᶜᵃ[i,   j-1], grid.φᶜᶜᵃ[i,   j-1],
                                                      grid.λᶜᶜᵃ[i,   j  ], grid.φᶜᶜᵃ[i,   j  ],
                                                      grid.λᶜᶜᵃ[i-1, j  ], grid.φᶜᶜᵃ[i-1, j  ],
                                                      grid.radius)

        @test grid.Azᶜᶜᵃ[i, j] ≈ expected_Azᶜᶜᵃ
        @test grid.Azᶠᶠᵃ[i, j] ≈ expected_Azᶠᶠᵃ
        @test grid.Azᶠᶜᵃ[i, j] ≈ grid.Δyᶠᶜᵃ[i, j] * grid.Δxᶠᶜᵃ[i, j]
        @test grid.Azᶜᶠᵃ[i, j] ≈ grid.Δyᶜᶠᵃ[i, j] * grid.Δxᶜᶠᵃ[i, j]

        φ = grid.φᶜᶜᵃ[i, j]
        k = lcc_scale_factor(map, φ)
        @test grid.Δxᶜᶜᵃ[i, j] ≈ map.Δx / k rtol=1e-3
        @test grid.Δyᶜᶜᵃ[i, j] ≈ map.Δy / k rtol=1e-3

        n = unit_vector_from_degrees(grid.λᶜᶜᵃ[i, j], grid.φᶜᶜᵃ[i, j])
        a = unit_vector_from_degrees(grid.λᶠᶠᵃ[i, j], grid.φᶠᶠᵃ[i, j])
        b = unit_vector_from_degrees(grid.λᶠᶠᵃ[i+1, j], grid.φᶠᶠᵃ[i+1, j])
        d = unit_vector_from_degrees(grid.λᶠᶠᵃ[i, j+1], grid.φᶠᶠᵃ[i, j+1])

        t̂x = normalized_projected_tangent(a, b, n)
        t̂y = normalized_projected_tangent(a, d, n)

        @test abs(dot_tuple(t̂x, t̂y)) < 1e-3

        grid_h7 = with_halo((7, 7, 7), grid)

        @test halo_size(grid_h7) == (7, 7, 7)
        @test topology(grid_h7) == topology(grid)
        @test znodes(grid_h7, Face()) == znodes(grid, Face())
        @test grid_h7.conformal_mapping.x₁ ≈ grid.conformal_mapping.x₁
        @test grid_h7.conformal_mapping.y₁ ≈ grid.conformal_mapping.y₁
        @test grid_h7.conformal_mapping.x₁ + size(grid_h7, 1) * grid_h7.conformal_mapping.Δx ≈
              grid.conformal_mapping.x₁ + size(grid, 1) * grid.conformal_mapping.Δx
        @test grid_h7.conformal_mapping.y₁ + size(grid_h7, 2) * grid_h7.conformal_mapping.Δy ≈
              grid.conformal_mapping.y₁ + size(grid, 2) * grid.conformal_mapping.Δy
        @test grid_h7.conformal_mapping.Δx ≈ grid.conformal_mapping.Δx
        @test grid_h7.conformal_mapping.Δy ≈ grid.conformal_mapping.Δy
        @test grid_h7.radius ≈ grid.radius
        @test grid_h7.conformal_mapping.radius ≈ grid.conformal_mapping.radius
        @test grid_h7.conformal_mapping.central_longitude ≈ grid.conformal_mapping.central_longitude
        @test grid_h7.conformal_mapping.latitude_of_origin ≈ grid.conformal_mapping.latitude_of_origin
        @test grid_h7.conformal_mapping.standard_parallel_1 ≈ grid.conformal_mapping.standard_parallel_1
        @test grid_h7.conformal_mapping.standard_parallel_2 ≈ grid.conformal_mapping.standard_parallel_2
        @test grid_h7.conformal_mapping.false_easting ≈ grid.conformal_mapping.false_easting
        @test grid_h7.conformal_mapping.false_northing ≈ grid.conformal_mapping.false_northing

        for (array_h7, array) in zip(lcc_coordinate_arrays(grid_h7),
                                     lcc_coordinate_arrays(grid))
            @test array_h7[24, 20] ≈ array[24, 20]
        end

        for (array_h7, array) in zip(lcc_metric_arrays(grid_h7),
                                     lcc_metric_arrays(grid))
            @test array_h7[24, 20] ≈ array[24, 20]
        end

        args, kwargs = constructor_arguments(grid)
        reconstructed_grid = LambertConformalConicGrid(args[:architecture], args[:number_type]; kwargs...)

        @test reconstructed_grid isa LambertConformalConicGrid
        @test reconstructed_grid.conformal_mapping == grid.conformal_mapping
        @test topology(reconstructed_grid) == topology(grid)
        @test halo_size(reconstructed_grid) == halo_size(grid)
        @test znodes(reconstructed_grid, Face()) == znodes(grid, Face())

        for (reconstructed_array, array) in zip(lcc_coordinate_arrays(reconstructed_grid),
                                               lcc_coordinate_arrays(grid))
            @test reconstructed_array[24, 20] ≈ array[24, 20]
        end

        similar_grid = similar(grid)

        @test similar_grid isa LambertConformalConicGrid
        @test eltype(similar_grid) == eltype(grid)
        @test similar_grid.conformal_mapping == grid.conformal_mapping
        @test topology(similar_grid) == topology(grid)
        @test halo_size(similar_grid) == halo_size(grid)
        @test znodes(similar_grid, Face()) == znodes(grid, Face())

        float32_grid = with_number_type(Float32, grid)

        @test float32_grid isa LambertConformalConicGrid
        @test eltype(float32_grid) == Float32
        @test topology(float32_grid) == topology(grid)
        @test halo_size(float32_grid) == halo_size(grid)
        @test znodes(float32_grid, Face()) ≈ znodes(grid, Face())
        @test float32_grid.conformal_mapping.x₁ ≈ grid.conformal_mapping.x₁
        @test float32_grid.conformal_mapping.y₁ ≈ grid.conformal_mapping.y₁
        @test float32_grid.conformal_mapping.Δx ≈ grid.conformal_mapping.Δx
        @test float32_grid.conformal_mapping.Δy ≈ grid.conformal_mapping.Δy
    end

    @testset "Float32 coordinate and metric arrays" begin
        grid = LambertConformalConicGrid(CPU(), Float32;
                                         size = (16, 12, 1),
                                         center = (-105, 40),
                                         spacing = 5e3,
                                         standard_parallels = (30, 60),
                                         z = (-1, 0),
                                         halo = (3, 3, 3))

        map = grid.conformal_mapping
        locations = ((Center(), Center()), (Face(), Center()), (Center(), Face()), (Face(), Face()))
        λ_arrays = (grid.λᶜᶜᵃ, grid.λᶠᶜᵃ, grid.λᶜᶠᵃ, grid.λᶠᶠᵃ)
        φ_arrays = (grid.φᶜᶜᵃ, grid.φᶠᶜᵃ, grid.φᶜᶠᵃ, grid.φᶠᶠᵃ)

        for ((ℓx, ℓy), λ_array, φ_array) in zip(locations, λ_arrays, φ_arrays)
            for i in (-2, 1, 8, 16), j in (-2, 1, 6, 12)
                x = lcc_xnode(i, ℓx, map)
                y = lcc_ynode(j, ℓy, map)
                λ, φ = lcc_inverse(map, x, y)

                @test λ_array[i, j] ≈ λ atol=1e-4
                @test φ_array[i, j] ≈ φ atol=1e-4
            end
        end

        Nx, Ny, _ = size(grid)
        Hx, Hy, _ = halo_size(grid)
        operator_range = (-Hx:Nx+Hx+1, -Hy:Ny+Hy+1)

        for metric in lcc_metric_arrays(grid)
            operator_metric = metric[operator_range...]
            @test all(isfinite, operator_metric)
            @test minimum(operator_metric) > 0
        end

        grid_h5 = with_halo((5, 5, 5), grid)

        @test eltype(grid_h5) === Float32
        @test halo_size(grid_h5) == (5, 5, 5)
        @test topology(grid_h5) == topology(grid)
        @test znodes(grid_h5, Face()) == znodes(grid, Face())
        @test grid_h5.conformal_mapping == grid.conformal_mapping

        for (array_h5, array) in zip(lcc_coordinate_arrays(grid_h5),
                                     lcc_coordinate_arrays(grid))
            @test array_h5[8, 6] ≈ array[8, 6]
        end

        for (array_h5, array) in zip(lcc_metric_arrays(grid_h5),
                                     lcc_metric_arrays(grid))
            @test array_h5[8, 6] ≈ array[8, 6]
        end
    end

    @testset "vector rotation" begin
        grid = LambertConformalConicGrid(CPU(), Float64;
                                         size = (24, 20, 1),
                                         center = (-105, 40),
                                         spacing = 5e3,
                                         standard_parallels = (30, 60),
                                         z = (-1, 0))

        i, j, k = 12, 10, 1
        θ = rotation_angle(i, j, grid)
        @test isfinite(θ)

        uᵢ, vᵢ = intrinsic_vector(i, j, k, grid, 1, -2)
        uₑ, vₑ = extrinsic_vector(i, j, k, grid, uᵢ, vᵢ)

        @test isfinite(uᵢ)
        @test isfinite(vᵢ)
        @test uₑ ≈ 1
        @test vₑ ≈ -2
    end

    @testset "hydrostatic model smoke test" begin
        grid = LambertConformalConicGrid(CPU(), Float64;
                                         size = (12, 12, 3),
                                         center = (0, 85),
                                         spacing = 25e3,
                                         standard_parallels = (80, 85),
                                         latitude_of_origin = 85,
                                         central_longitude = 0,
                                         z = (-100, 0),
                                         halo = (3, 3, 3))

        model = HydrostaticFreeSurfaceModel(grid;
                                            coriolis = HydrostaticSphericalCoriolis(),
                                            free_surface = SplitExplicitFreeSurface(grid; substeps = 5),
                                            tracers = ())

        simulation = Simulation(model; Δt = 10, stop_iteration = 3)
        run!(simulation)

        @test isfinite(time(simulation))
        @test all(isfinite, interior(model.free_surface.displacement))
        @test all(isfinite, interior(model.velocities.u))
        @test all(isfinite, interior(model.velocities.v))
    end

    @testset "architecture construction and transfer" begin
        for FT in float_types
            grid_cpu = LambertConformalConicGrid(CPU(), FT;
                                                 size = (12, 10, 2),
                                                 center = (-105, 40),
                                                 spacing = 20e3,
                                                 standard_parallels = (30, 60),
                                                 z = (-100, 0))

            @test Oceananigans.OrthogonalSphericalShellGrids.fill_lcc_coordinates_and_metrics!(grid_cpu) === nothing

            coordinate_atol = FT === Float64 ? 1e-10 : 1e-6
            coordinate_rtol = FT === Float64 ? 1e-12 : 1e-6
            # GPU-vs-CPU metric comparisons use loose rtol because spherical_quadrilateral_area
            # (atan + dot/cross of unit vectors) accumulates FMA/transcendental differences
            # of ~2e-3 relative on Float32 GPU.
            metric_tolerance = FT === Float64 ? 1e-10 : 5e-3

            grid_cpu_again = on_architecture(CPU(), grid_cpu)

            @test on_architecture(CPU(), grid_cpu.conformal_mapping) == grid_cpu.conformal_mapping
            @test grid_cpu_again.conformal_mapping == grid_cpu.conformal_mapping

            for (transferred_array, cpu_array) in
                zip(lcc_coordinate_arrays(grid_cpu_again),
                    lcc_coordinate_arrays(grid_cpu))
                @test Array(parent(transferred_array)) == Array(parent(cpu_array))
            end

            for (transferred_array, cpu_array) in
                zip(lcc_metric_arrays(grid_cpu_again),
                    lcc_metric_arrays(grid_cpu))
                @test Array(parent(transferred_array)) == Array(parent(cpu_array))
            end

            for arch in archs
                arch isa GPU || continue

                grid_gpu_direct = LambertConformalConicGrid(arch, FT;
                                                            size = (12, 10, 2),
                                                            center = (-105, 40),
                                                            spacing = 20e3,
                                                            standard_parallels = (30, 60),
                                                            z = (-100, 0))

                # Direct construction already launches these kernels. Refill once
                # explicitly so the GPU gate covers LCC coordinate and metric
                # generation outside constructor plumbing too.
                @test fill_lcc_coordinates_and_metrics!(grid_gpu_direct) === nothing

                grid_gpu = on_architecture(arch, grid_cpu)
                grid_direct_back = on_architecture(CPU(), grid_gpu_direct)
                grid_back = on_architecture(CPU(), grid_gpu)

                @test on_architecture(arch, grid_cpu.conformal_mapping) == grid_cpu.conformal_mapping
                @test architecture(grid_gpu_direct) == arch
                @test architecture(grid_gpu) == arch
                @test grid_direct_back.conformal_mapping isa LambertConformalConic
                @test grid_back.conformal_mapping isa LambertConformalConic
                @test grid_direct_back.conformal_mapping == grid_cpu.conformal_mapping
                @test grid_back.conformal_mapping == grid_cpu.conformal_mapping

                for array in (lcc_coordinate_arrays(grid_gpu_direct)...,
                              lcc_metric_arrays(grid_gpu_direct)...)
                    @test architecture(parent(array)) == arch
                end

                for array in (lcc_coordinate_arrays(grid_gpu)...,
                              lcc_metric_arrays(grid_gpu)...)
                    @test architecture(parent(array)) == arch
                end

                for (direct_array, transferred_array, cpu_array) in
                    zip(lcc_coordinate_arrays(grid_direct_back),
                        lcc_coordinate_arrays(grid_back),
                        lcc_coordinate_arrays(grid_cpu))
                    @test isapprox(Array(parent(direct_array)),
                                   Array(parent(cpu_array));
                                   atol = coordinate_atol,
                                   rtol = coordinate_rtol)

                    @test isapprox(Array(parent(transferred_array)),
                                   Array(parent(cpu_array));
                                   atol = coordinate_atol,
                                   rtol = coordinate_rtol)
                end

                for (direct_array, transferred_array, cpu_array) in
                    zip(lcc_metric_arrays(grid_direct_back),
                        lcc_metric_arrays(grid_back),
                        lcc_metric_arrays(grid_cpu))
                    @test isapprox(Array(parent(direct_array)),
                                   Array(parent(cpu_array));
                                   rtol = metric_tolerance)

                    @test isapprox(Array(parent(transferred_array)),
                                   Array(parent(cpu_array));
                                   rtol = metric_tolerance)
                end

                grid_cpu_h7 = with_halo((7, 7, 7), grid_cpu)
                grid_gpu_h7 = with_halo((7, 7, 7), grid_gpu_direct)
                grid_h7_back = on_architecture(CPU(), grid_gpu_h7)

                @test architecture(grid_gpu_h7) == arch
                @test halo_size(grid_gpu_h7) == (7, 7, 7)
                @test halo_size(grid_h7_back) == halo_size(grid_cpu_h7)
                @test topology(grid_h7_back) == topology(grid_cpu_h7)
                @test znodes(grid_h7_back, Face()) == znodes(grid_cpu_h7, Face())
                @test grid_h7_back.conformal_mapping.x₁ ≈ grid_cpu_h7.conformal_mapping.x₁
                @test grid_h7_back.conformal_mapping.y₁ ≈ grid_cpu_h7.conformal_mapping.y₁
                @test grid_h7_back.conformal_mapping.Δx ≈ grid_cpu_h7.conformal_mapping.Δx
                @test grid_h7_back.conformal_mapping.Δy ≈ grid_cpu_h7.conformal_mapping.Δy
                @test grid_h7_back.conformal_mapping.radius ≈ grid_cpu_h7.conformal_mapping.radius
                @test grid_h7_back.conformal_mapping.central_longitude ≈
                      grid_cpu_h7.conformal_mapping.central_longitude
                @test grid_h7_back.conformal_mapping.latitude_of_origin ≈
                      grid_cpu_h7.conformal_mapping.latitude_of_origin
                @test grid_h7_back.conformal_mapping.standard_parallel_1 ≈
                      grid_cpu_h7.conformal_mapping.standard_parallel_1
                @test grid_h7_back.conformal_mapping.standard_parallel_2 ≈
                      grid_cpu_h7.conformal_mapping.standard_parallel_2
                @test grid_h7_back.conformal_mapping.false_easting ≈
                      grid_cpu_h7.conformal_mapping.false_easting
                @test grid_h7_back.conformal_mapping.false_northing ≈
                      grid_cpu_h7.conformal_mapping.false_northing

                for array in (lcc_coordinate_arrays(grid_gpu_h7)...,
                              lcc_metric_arrays(grid_gpu_h7)...)
                    @test architecture(parent(array)) == arch
                end

                for (gpu_h7_array, cpu_h7_array) in
                    zip(lcc_coordinate_arrays(grid_h7_back),
                        lcc_coordinate_arrays(grid_cpu_h7))
                    @test isapprox(Array(parent(gpu_h7_array)),
                                   Array(parent(cpu_h7_array));
                                   atol = coordinate_atol,
                                   rtol = coordinate_rtol)
                end

                for (gpu_h7_array, cpu_h7_array) in
                    zip(lcc_metric_arrays(grid_h7_back),
                        lcc_metric_arrays(grid_cpu_h7))
                    @test isapprox(Array(parent(gpu_h7_array)),
                                   Array(parent(cpu_h7_array));
                                   rtol = metric_tolerance)
                end

                flat_grid_gpu = LambertConformalConicGrid(arch, FT;
                                                          size = (8, 6),
                                                          center = (-105, 40),
                                                          spacing = 20e3,
                                                          standard_parallels = (30, 60),
                                                          z = nothing,
                                                          topology = (Bounded, Bounded, Flat))

                flat_grid_gpu_h2 = with_halo((2, 2, 0), flat_grid_gpu)
                flat_grid_back = on_architecture(CPU(), flat_grid_gpu)
                flat_grid_h2_back = on_architecture(CPU(), flat_grid_gpu_h2)

                @test architecture(flat_grid_gpu) == arch
                @test architecture(flat_grid_gpu_h2) == arch
                @test topology(flat_grid_back) == (Bounded, Bounded, Flat)
                @test topology(flat_grid_h2_back) == (Bounded, Bounded, Flat)
                @test halo_size(flat_grid_back) == (3, 3, 0)
                @test halo_size(flat_grid_h2_back) == (2, 2, 0)
                @test flat_grid_h2_back.conformal_mapping == flat_grid_back.conformal_mapping

                for array in (lcc_coordinate_arrays(flat_grid_gpu)...,
                              lcc_metric_arrays(flat_grid_gpu)...,
                              lcc_coordinate_arrays(flat_grid_gpu_h2)...,
                              lcc_metric_arrays(flat_grid_gpu_h2)...)
                    @test architecture(parent(array)) == arch
                end

                model = HydrostaticFreeSurfaceModel(grid_gpu_direct;
                                                    coriolis = HydrostaticSphericalCoriolis(),
                                                    free_surface = SplitExplicitFreeSurface(grid_gpu_direct;
                                                                                           substeps = 5),
                                                    tracers = ())

                simulation = Simulation(model; Δt = 10, stop_iteration = 3)
                run!(simulation)

                η = on_architecture(CPU(), model.free_surface.displacement)
                u = on_architecture(CPU(), model.velocities.u)
                v = on_architecture(CPU(), model.velocities.v)

                @test isfinite(time(simulation))
                @test all(isfinite, interior(η))
                @test all(isfinite, interior(u))
                @test all(isfinite, interior(v))
            end
        end
    end
end
