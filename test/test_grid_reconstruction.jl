include("dependencies_for_runtests.jl")

using Oceananigans.Grids: constructor_arguments, halo_size
using NCDatasets
using Oceananigans.OutputWriters: write_grid_reconstruction_data!, materialize_from_netcdf, reconstruct_grid_from_netcdf

#####
##### Grid reconstruction tests using constructor_arguments
#####

function test_regular_rectilinear_grid_reconstruction(arch, FT)
    original_grid = RectilinearGrid(arch, FT,
                                    size = (4, 6, 8),
                                    extent = (2π, 3π, 4π),
                                    topology = (Periodic, Bounded, Bounded),
                                    halo = (2, 3, 2))

    # Get constructor arguments
    args, kwargs = constructor_arguments(original_grid)

    # Reconstruct the grid
    reconstructed_grid = RectilinearGrid(args[:architecture], args[:number_type]; kwargs...)

    # Test that key properties match
    @test reconstructed_grid == original_grid # tests grid type, topology and face locations
    @test size(reconstructed_grid) == size(original_grid)
    @test halo_size(reconstructed_grid) == halo_size(original_grid)
    @test eltype(reconstructed_grid) == eltype(original_grid)

    # Test coordinate spacings (for regular grids these should be constant numbers)
    @test reconstructed_grid.Δxᶠᵃᵃ == original_grid.Δxᶠᵃᵃ
    @test reconstructed_grid.Δyᵃᶠᵃ == original_grid.Δyᵃᶠᵃ
    @test reconstructed_grid.z.Δᵃᵃᶠ == original_grid.z.Δᵃᵃᶠ

    # Test face and center coordinates match
    @test all(reconstructed_grid.xᶠᵃᵃ == original_grid.xᶠᵃᵃ)
    @test all(reconstructed_grid.xᶜᵃᵃ == original_grid.xᶜᵃᵃ)
    @test all(reconstructed_grid.yᵃᶠᵃ == original_grid.yᵃᶠᵃ)
    @test all(reconstructed_grid.yᵃᶜᵃ == original_grid.yᵃᶜᵃ)
    @test all(reconstructed_grid.z.cᵃᵃᶠ == original_grid.z.cᵃᵃᶠ)
    @test all(reconstructed_grid.z.cᵃᵃᶜ == original_grid.z.cᵃᵃᶜ)

    return nothing
end

function test_stretched_rectilinear_grid_reconstruction(arch, FT)
    N = 8
    x_faces = collect(range(0, 1, length=N+1))
    y_faces = [0.0, 0.1, 0.3, 0.6, 1.0]  # Irregular spacing
    z_func(k) = -1.0 + (k-1)/N  # Function-based coordinate

    original_grid = RectilinearGrid(arch, FT,
                                    size = (N, 4, N),
                                    x = x_faces,
                                    y = y_faces,
                                    z = z_func,
                                    topology = (Bounded, Bounded, Bounded),
                                    halo = (1, 1, 1))

    # Get constructor arguments
    args, kwargs = constructor_arguments(original_grid)

    # Reconstruct the grid
    reconstructed_grid = RectilinearGrid(args[:architecture], args[:number_type]; kwargs...)

    # Test that key properties match
    @test reconstructed_grid == original_grid # tests grid type, topology and face locations
    @test size(reconstructed_grid) == size(original_grid)
    @test halo_size(reconstructed_grid) == halo_size(original_grid)
    @test eltype(reconstructed_grid) == eltype(original_grid)

    # Test coordinate spacings (for regular grids these should be constant numbers)
    @test reconstructed_grid.Δxᶠᵃᵃ == original_grid.Δxᶠᵃᵃ
    @test reconstructed_grid.Δyᵃᶠᵃ == original_grid.Δyᵃᶠᵃ
    @test reconstructed_grid.z.Δᵃᵃᶠ == original_grid.z.Δᵃᵃᶠ

    # Test coordinate arrays match (these should be arrays for stretched grids)
    @test all(reconstructed_grid.xᶠᵃᵃ == original_grid.xᶠᵃᵃ)
    @test all(reconstructed_grid.xᶜᵃᵃ == original_grid.xᶜᵃᵃ)
    @test all(reconstructed_grid.yᵃᶠᵃ == original_grid.yᵃᶠᵃ)
    @test all(reconstructed_grid.yᵃᶜᵃ == original_grid.yᵃᶜᵃ)
    @test all(reconstructed_grid.z.cᵃᵃᶠ == original_grid.z.cᵃᵃᶠ)
    @test all(reconstructed_grid.z.cᵃᵃᶜ == original_grid.z.cᵃᵃᶜ)

    # Test vertical coordinate
    @test all(reconstructed_grid.z.cᵃᵃᶠ == original_grid.z.cᵃᵃᶠ)
    @test all(reconstructed_grid.z.cᵃᵃᶜ == original_grid.z.cᵃᵃᶜ)

    return nothing
end

function test_flat_dimension_grid_reconstruction(arch, FT)
    original_grid = RectilinearGrid(arch, FT,
                                    size = (8, 16),
                                    x = (0, 2),
                                    z = (-1, 0),
                                    topology = (Periodic, Flat, Bounded),
                                    halo = (2, 2))

    # Get constructor arguments
    args, kwargs = constructor_arguments(original_grid)

    # Reconstruct the grid
    reconstructed_grid = RectilinearGrid(args[:architecture], args[:number_type]; kwargs...)

    # Test that key properties match
    @test reconstructed_grid == original_grid # tests grid type, topology and face locations
    @test size(reconstructed_grid) == size(original_grid)
    @test halo_size(reconstructed_grid) == halo_size(original_grid)
    @test eltype(reconstructed_grid) == eltype(original_grid)

    # Test coordinate spacings (for regular grids these should be constant numbers)
    @test reconstructed_grid.Δxᶠᵃᵃ == original_grid.Δxᶠᵃᵃ
    @test reconstructed_grid.Δyᵃᶠᵃ == original_grid.Δyᵃᶠᵃ
    @test reconstructed_grid.z.Δᵃᵃᶠ == original_grid.z.Δᵃᵃᶠ

    # Test coordinates match
    @test all(reconstructed_grid.xᶠᵃᵃ == original_grid.xᶠᵃᵃ)
    @test all(reconstructed_grid.xᶜᵃᵃ == original_grid.xᶜᵃᵃ)
    @test all(reconstructed_grid.yᵃᶠᵃ == original_grid.yᵃᶠᵃ)
    @test all(reconstructed_grid.yᵃᶜᵃ == original_grid.yᵃᶜᵃ)
    @test all(reconstructed_grid.z.cᵃᵃᶠ == original_grid.z.cᵃᵃᶠ)
    @test all(reconstructed_grid.z.cᵃᵃᶜ == original_grid.z.cᵃᵃᶜ)

    return nothing
end

function test_different_topologies_grid_reconstruction(arch, FT)
    topologies = [(Periodic, Periodic, Bounded),
                  (Bounded, Bounded, Bounded),
                  (Periodic, Bounded, Bounded),
                  (Bounded, Periodic, Bounded)]

    for topo in topologies
        original_grid = RectilinearGrid(arch, FT,
                                        size = (4, 4, 4),
                                        extent = (1, 1, 1),
                                        topology = topo,
                                        halo = (2, 2, 2))

        # Get constructor arguments
        args, kwargs = constructor_arguments(original_grid)

        # Reconstruct the grid
        reconstructed_grid = RectilinearGrid(args[:architecture], args[:number_type]; kwargs...)

        # Test that key properties match
        @test reconstructed_grid == original_grid # tests grid type, topology and face locations
        @test size(reconstructed_grid) == size(original_grid)
        @test halo_size(reconstructed_grid) == halo_size(original_grid)
        @test eltype(reconstructed_grid) == eltype(original_grid)

        # Test coordinate spacings (for regular grids these should be constant numbers)
        @test reconstructed_grid.Δxᶠᵃᵃ == original_grid.Δxᶠᵃᵃ
        @test reconstructed_grid.Δyᵃᶠᵃ == original_grid.Δyᵃᶠᵃ
        @test reconstructed_grid.z.Δᵃᵃᶠ == original_grid.z.Δᵃᵃᶠ

        # Test coordinates match
        @test all(reconstructed_grid.xᶠᵃᵃ == original_grid.xᶠᵃᵃ)
        @test all(reconstructed_grid.xᶜᵃᵃ == original_grid.xᶜᵃᵃ)
        @test all(reconstructed_grid.yᵃᶠᵃ == original_grid.yᵃᶠᵃ)
        @test all(reconstructed_grid.yᵃᶜᵃ == original_grid.yᵃᶜᵃ)
        @test all(reconstructed_grid.z.cᵃᵃᶠ == original_grid.z.cᵃᵃᶠ)
        @test all(reconstructed_grid.z.cᵃᵃᶜ == original_grid.z.cᵃᵃᶜ)
    end

    return nothing
end

function test_grid_equality_after_reconstruction(arch, FT)
    # Test that reconstructed grids are functionally equivalent
    original_grid = RectilinearGrid(arch, FT,
                                    size = (8, 8, 8),
                                    extent = (1, 2, 3),
                                    topology = (Periodic, Periodic, Bounded),
                                    halo = (3, 3, 3))

    # Get constructor arguments and reconstruct
    args, kwargs = constructor_arguments(original_grid)
    reconstructed_grid = RectilinearGrid(args[:architecture], args[:number_type]; kwargs...)

    # Test grid equality
    @test original_grid == reconstructed_grid

    # Test that we can create equivalent fields on both grids
    original_field = CenterField(original_grid)
    reconstructed_field = CenterField(reconstructed_grid)

    # Fields should have the same size and location
    @test size(original_field) == size(reconstructed_field)
    @test eltype(original_field) == eltype(reconstructed_field)

    return nothing
end

function test_latitude_longitude_grid_reconstruction(original_grid)
    args, kwargs = constructor_arguments(original_grid)

    # Reconstruct the grid
    reconstructed_grid = LatitudeLongitudeGrid(args[:architecture], args[:number_type]; kwargs...)

    # Test that key properties match
    @test reconstructed_grid == original_grid # tests grid type, topology and face locations
    @test size(reconstructed_grid) == size(original_grid)
    @test halo_size(reconstructed_grid) == halo_size(original_grid)
    @test eltype(reconstructed_grid) == eltype(original_grid)

    # Test radius
    @test reconstructed_grid.radius == original_grid.radius

    # Test coordinate spacings (for regular grids these should be constant numbers)
    @test reconstructed_grid.Δλᶠᵃᵃ == original_grid.Δλᶠᵃᵃ
    @test reconstructed_grid.Δφᵃᶠᵃ == original_grid.Δφᵃᶠᵃ
    @test reconstructed_grid.Δxᶠᶠᵃ == original_grid.Δxᶠᶠᵃ
    @test reconstructed_grid.Δyᶜᶠᵃ == original_grid.Δyᶜᶠᵃ
    @test reconstructed_grid.z.Δᵃᵃᶠ == original_grid.z.Δᵃᵃᶠ

    # Test coordinate arrays match
    @test all(reconstructed_grid.λᶠᵃᵃ == original_grid.λᶠᵃᵃ)
    @test all(reconstructed_grid.λᶜᵃᵃ == original_grid.λᶜᵃᵃ)
    @test all(reconstructed_grid.φᵃᶠᵃ == original_grid.φᵃᶠᵃ)
    @test all(reconstructed_grid.φᵃᶜᵃ == original_grid.φᵃᶜᵃ)
    @test all(reconstructed_grid.z.cᵃᵃᶠ == original_grid.z.cᵃᵃᶠ)
    @test all(reconstructed_grid.z.cᵃᵃᶜ == original_grid.z.cᵃᵃᶜ)

    return nothing
end

function test_immersed_grid_reconstruction(original_grid)
    args, kwargs = constructor_arguments(original_grid)

    @test :immersed_boundary_type in keys(args)
    @test :architecture in keys(args)
    @test :number_type in keys(args)

    # Reconstruct the immersed boundary and then the grid
    original_ib = original_grid.immersed_boundary
    if original_ib isa GridFittedBottom
        reconstructed_ib = GridFittedBottom(args[:bottom_height], args[:immersed_condition])
    elseif original_ib isa PartialCellBottom
        reconstructed_ib = PartialCellBottom(args[:bottom_height], args[:minimum_fractional_cell_height])
    elseif original_ib isa GridFittedBoundary
        reconstructed_ib = GridFittedBoundary(args[:mask])
    end
    reconstructed_underlying_grid = RectilinearGrid(args[:architecture], args[:number_type]; kwargs...)
    reconstructed_grid = ImmersedBoundaryGrid(reconstructed_underlying_grid, reconstructed_ib)

    # Test that key properties match
    @test reconstructed_grid.underlying_grid == original_grid.underlying_grid # tests grid type, topology and face locations
    @test reconstructed_grid == original_grid
    @test size(reconstructed_grid) == size(original_grid)
    @test halo_size(reconstructed_grid) == halo_size(original_grid)
    @test eltype(reconstructed_grid) == eltype(original_grid)

    return nothing
end

function test_netcdf_grid_reconstruction(original_grid)
    # Create NetCDF dataset and write grid reconstruction data
    filename = "test_netcdf_grid_reconstruction.nc"
    ds = NCDataset(filename, "c")
    write_grid_reconstruction_data!(ds, original_grid)
    close(ds)

    # Read back the grid reconstruction metadata
    reconstructed_grid = reconstruct_grid_from_netcdf(filename)

    # Test that key properties match
    @test reconstructed_grid == original_grid # tests grid type, topology and face locations
    @test size(reconstructed_grid) == size(original_grid)
    @test halo_size(reconstructed_grid) == halo_size(original_grid)
    @test eltype(reconstructed_grid) == eltype(original_grid)

    rm(filename)
    return nothing
end

#####
##### Run tests
#####

N = 6

@testset "Grid constructor_arguments and reconstruction tests" begin
    @info "Testing grid constructor_arguments function and reconstruction..."

    for arch in archs, FT in float_types

        @testset "RectilinearGrid reconstruction tests [$FT, $(typeof(arch))]" begin
            @info "  Testing RectilinearGrid reconstruction [$FT, $(typeof(arch))]..."

            test_regular_rectilinear_grid_reconstruction(arch, FT)
            test_stretched_rectilinear_grid_reconstruction(arch, FT)
            test_flat_dimension_grid_reconstruction(arch, FT)
            test_different_topologies_grid_reconstruction(arch, FT)
            test_grid_equality_after_reconstruction(arch, FT)
        end

        regular_rectilinear_grid = RectilinearGrid(arch, FT, size=(N, N, N),
                                                   extent = (1, 1, 1),
                                                   topology = (Periodic, Bounded, Bounded),
                                                   halo=(2, 2, 2))

        regular_latlon_grid = LatitudeLongitudeGrid(arch, FT, size=(N, N, N),
                                                   longitude = (0, 1),
                                                   latitude = (0, 1),
                                                   z = (-1, 0),
                                                   topology = (Periodic, Bounded, Bounded),
                                                   halo = (2, 2, 2))

        stretched_rectilinear_grid = RectilinearGrid(arch, FT, size=(N, N, N),
                                                     x = collect(range(0, 1, length=N+1)),
                                                     y = collect(range(0, 1, length=N+1)),
                                                     z = k -> -1 + (k-1)/N,
                                                     topology = (Bounded, Bounded, Bounded),
                                                     halo=(1, 1, 1))

        stretched_latlon_grid = LatitudeLongitudeGrid(arch, FT, size=(N, N, N),
                                                      longitude = collect(range(0, 1, length=N+1)),
                                                      latitude = collect(range(0, 1, length=N+1)),
                                                      z = k -> -1 + (k-1)/N,
                                                      topology = (Periodic, Bounded, Bounded),
                                                      halo = (1, 1, 1))

        bfboundary = GridFittedBoundary((x, y, z) -> z < -1/2)
        gfboundary_rectilinear_grid = ImmersedBoundaryGrid(regular_rectilinear_grid, bfboundary)
        gfboundary_latlon_grid = ImmersedBoundaryGrid(regular_latlon_grid, bfboundary)

        gfbottom = GridFittedBottom(-1/2)
        gfbottom_rectilinear_grid = ImmersedBoundaryGrid(regular_rectilinear_grid, gfbottom)
        gfbottom_latlon_grid = ImmersedBoundaryGrid(regular_latlon_grid, gfbottom)

        pcbottom = PartialCellBottom(-1/2)
        pcbottom_rectilinear_grid = ImmersedBoundaryGrid(regular_rectilinear_grid, pcbottom)
        pcbottom_latlon_grid = ImmersedBoundaryGrid(regular_latlon_grid, pcbottom)

        @testset "ImmersedBoundaryGrid reconstruction tests [$FT, $(typeof(arch))]" begin
            @info "  Testing ImmersedBoundaryGrid reconstruction [$FT, $(typeof(arch))]..."
            test_immersed_grid_reconstruction(gfboundary_rectilinear_grid)
            test_immersed_grid_reconstruction(gfbottom_rectilinear_grid)
            test_immersed_grid_reconstruction(pcbottom_rectilinear_grid)
        end

        @testset "LatitudeLongitudeGrid reconstruction tests [$FT, $(typeof(arch))]" begin
            @info "  Testing LatitudeLongitudeGrid reconstruction [$FT, $(typeof(arch))]..."
            test_latitude_longitude_grid_reconstruction(regular_latlon_grid)
            test_latitude_longitude_grid_reconstruction(stretched_latlon_grid)
        end

        @testset "NetCDF grid reconstruction tests [$FT, $(typeof(arch))]" begin
            @info "  Testing NetCDF grid reconstruction [$FT, $(typeof(arch))]..."
            test_netcdf_grid_reconstruction(regular_rectilinear_grid)
            test_netcdf_grid_reconstruction(regular_latlon_grid)
            test_netcdf_grid_reconstruction(stretched_rectilinear_grid)
            test_netcdf_grid_reconstruction(stretched_latlon_grid)

            # TODO: Make the functionality below work
            # test_netcdf_grid_reconstruction(gfboundary_rectilinear_grid)
            # test_netcdf_grid_reconstruction(gfboundary_latlon_grid)

            # test_netcdf_grid_reconstruction(gfbottom_rectilinear_grid)
            # test_netcdf_grid_reconstruction(gfbottom_latlon_grid)

            # test_netcdf_grid_reconstruction(pcbottom_rectilinear_grid)
            # test_netcdf_grid_reconstruction(pcbottom_latlon_grid)
        end
    end
end
