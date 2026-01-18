using Test
using Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.Grids: nodes

const CAIROMAKIE_AVAILABLE = let loaded = false
    try
        @eval begin
            using CairoMakie
        end
        CairoMakie.activate!()  # use headless backend for CI
        loaded = true
    catch err
        @info "Skipping OceananigansMakieExt tests because CairoMakie is unavailable." exception = (err, catch_backtrace())
        loaded = false
    end
    loaded
end

sequential_data(sz::NTuple{N, Int}) where N = reshape(Float64.(1:prod(sz)), sz...)

@testset "OceananigansMakieExt" begin
    if !CAIROMAKIE_AVAILABLE
        @test true
        return
    end

    @testset "2D heatmap conversion" begin
        grid = RectilinearGrid(size=(4, 3), x=(0, 1), y=(0, 1), topology=(Periodic, Periodic, Flat))
        field = CenterField(grid)
        set!(field, sequential_data(size(field)))

        slice = view(field, :, :, 1)
        plot = heatmap(slice)
        heatmap_plot = plot.plot
        converted = heatmap_plot.converted[]

        @test heatmap_plot isa CairoMakie.Heatmap
        @test length(converted[1]) == size(slice, 1) + 1
        @test length(converted[2]) == size(slice, 2) + 1

        expected_plane = Float32.(Array(interior(slice, :, :, 1)))
        @test converted[3] == expected_plane
    end

    @testset "1D horizontal line conversion" begin
        grid = RectilinearGrid(size=(4, 1, 1), x=(0, 1), y=(0, 1), z=(-1, 0))
        field = CenterField(grid)
        set!(field, sequential_data(size(field)))

        slice = view(field, :, 1, 1)
        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(fig[1, 1])
        line_plot = lines!(ax, slice)

        points = first(line_plot.converted[])
        x_coords = [p[1] for p in points]
        y_coords = [p[2] for p in points]

        expected_x = collect(nodes(slice)[1])
        expected_y = vec(Array(interior(slice)))

        @test x_coords == expected_x
        @test y_coords == expected_y
    end

    @testset "1D vertical line conversion" begin
        grid = RectilinearGrid(size=(1, 1, 4), x=(0, 1), y=(0, 1), z=(-1, 0))
        field = CenterField(grid)
        set!(field, sequential_data(size(field)))

        slice = view(field, 1, 1, :)
        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(fig[1, 1])
        line_plot = lines!(ax, slice)

        points = first(line_plot.converted[])
        x_coords = [p[1] for p in points]
        y_coords = [p[2] for p in points]

        expected_y = collect(nodes(slice)[3])
        expected_x = vec(Array(interior(slice)))

        @test x_coords == expected_x
        @test y_coords == expected_y
    end

    @testset "1D line with field first and coordinate array second" begin
        # Test lines!(ax, field, coordinate_array) syntax
        grid = RectilinearGrid(size=8, z=(0, 1), topology=(Flat, Flat, Bounded))
        z = znodes(grid, Center()) ./ 1e3
        T = CenterField(grid)
        set!(T, sequential_data(size(T)))

        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis(fig[1, 1])
        line_plot = lines!(ax, T, z)

        points = first(line_plot.converted[])
        x_coords = [p[1] for p in points]
        y_coords = [p[2] for p in points]

        expected_x = vec(Array(interior(T)))
        expected_y = collect(z)

        @test x_coords == expected_x
        @test y_coords == expected_y
    end

    @testset "spherical_coordinates" begin
        ext = Base.get_extension(Oceananigans, :OceananigansMakieExt)
        spherical_coordinates = ext.spherical_coordinates

        # Test scalar conversion
        x, y, z = spherical_coordinates(0.0, 0.0)
        @test x ≈ 1.0
        @test y ≈ 0.0
        @test z ≈ 0.0

        # Test at north pole
        x, y, z = spherical_coordinates(0.0, 90.0)
        @test x ≈ 0.0 atol=1e-10
        @test y ≈ 0.0 atol=1e-10
        @test z ≈ 1.0

        # Test array conversion
        λ = [0.0, 90.0, 180.0]
        φ = [0.0, 0.0, 0.0]
        x, y, z = spherical_coordinates(λ, φ)
        @test x ≈ [1.0, 0.0, -1.0] atol=1e-10
        @test y ≈ [0.0, 1.0, 0.0] atol=1e-10
        @test z ≈ [0.0, 0.0, 0.0] atol=1e-10
    end

    @testset "surface! on LatitudeLongitudeGrid" begin
        grid = LatitudeLongitudeGrid(size=(8, 6, 1),
                                     longitude=(0, 360),
                                     latitude=(-60, 60),
                                     z=(0, 1))
        T = CenterField(grid)
        set!(T, (λ, φ, z) -> cosd(φ))

        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis3(fig[1, 1]; aspect=:data)

        # This should work via the extension
        plt = surface!(ax, T; colormap=:viridis)
        @test plt isa CairoMakie.Surface
    end

    @testset "surface! on TripolarGrid" begin
        grid = TripolarGrid(size=(8, 6, 1), z=(0, 1))
        T = CenterField(grid)
        set!(T, (λ, φ, z) -> cosd(φ))

        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis3(fig[1, 1]; aspect=:data)

        plt = surface!(ax, T; colormap=:viridis)
        @test plt isa CairoMakie.Surface
    end

    @testset "surface! with Observable on spherical grid" begin
        grid = LatitudeLongitudeGrid(size=(8, 6, 1),
                                     longitude=(0, 360),
                                     latitude=(-60, 60),
                                     z=(0, 1))
        T1 = CenterField(grid)
        T2 = CenterField(grid)
        set!(T1, (λ, φ, z) -> cosd(φ))
        set!(T2, (λ, φ, z) -> sind(λ))

        fields = [T1, T2]
        n = CairoMakie.Observable(1)
        T_obs = CairoMakie.@lift fields[$n]

        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis3(fig[1, 1]; aspect=:data)

        # This should work with Observable fields
        plt = surface!(ax, T_obs; colormap=:viridis)
        @test plt isa CairoMakie.Surface

        # Update observable and verify no errors
        n[] = 2
        @test true  # If we get here, the update worked
    end

    @testset "geo_surface!" begin
        ext = Base.get_extension(Oceananigans, :OceananigansMakieExt)
        geo_surface! = ext.geo_surface!

        grid = LatitudeLongitudeGrid(size=(8, 6, 1),
                                     longitude=(0, 360),
                                     latitude=(-60, 60),
                                     z=(0, 1))
        T = CenterField(grid)
        set!(T, (λ, φ, z) -> cosd(φ))

        fig = CairoMakie.Figure()
        ax = CairoMakie.Axis3(fig[1, 1]; aspect=:data)

        plt = geo_surface!(ax, T; colormap=:thermal)
        @test plt isa CairoMakie.Surface
    end
end
