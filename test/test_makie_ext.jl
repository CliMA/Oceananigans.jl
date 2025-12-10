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
end
