include("dependencies_for_runtests.jl")

using Oceananigans.Grids: constructor_arguments, halo_size

#####
##### Grid reconstruction tests using constructor_arguments
#####

function test_regular_rectilinear_grid_reconstruction(arch, FT)
    # Test regular rectilinear grid reconstruction
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
    # Test grid with stretched coordinates
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
    # Test grid with flat dimensions
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
    # Test various topology combinations
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

function test_latitude_longitude_grid_reconstruction(arch, FT)
    # Test LatitudeLongitudeGrid reconstruction
    original_grid = LatitudeLongitudeGrid(arch, FT,
                                          size = (36, 24, 16),
                                          longitude = (-180, 180),
                                          latitude = (-80, 80),
                                          z = (-1000, 0),
                                          topology = (Periodic, Bounded, Bounded),
                                          halo = (2, 2, 2))

    # Get constructor arguments
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

#####
##### Run tests
#####

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

        @testset "LatitudeLongitudeGrid reconstruction tests [$FT, $(typeof(arch))]" begin
            @info "  Testing LatitudeLongitudeGrid reconstruction [$FT, $(typeof(arch))]..."
            
            test_latitude_longitude_grid_reconstruction(arch, FT)
        end
    end
end
