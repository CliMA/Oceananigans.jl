include("dependencies_for_runtests.jl")

using Oceananigans.Grids: total_extent, xspacings, yspacings, zspacings, xnode, ynode, znode
using Oceananigans.Operators: Δx, Δy, Δz, volume
using Oceananigans.ImmersedBoundaries: GridFittedBottom, PartialCellBottom, GridFittedBoundary, _immersed_cell, CenterImmersedCondition, InterfaceImmersedCondition

#####
##### Basic immersed boundary grid construction tests
#####

function test_immersed_boundary_grid_construction(FT, arch, boundary_type)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 4), extent=(1, 1, 1))

    bottom_function(x, y) = 0.2 * sin(2π * x) * cos(2π * y)
    ib = boundary_type(bottom_function)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    # Basic grid properties
    @test architecture(ibg) === arch
    @test eltype(ibg) === FT
    @test size(ibg) == size(underlying_grid)
    @test halo_size(ibg) == halo_size(underlying_grid)
    @test topology(ibg) == topology(underlying_grid)

    # Test that immersed boundary was materialized
    @test ibg.immersed_boundary.bottom_height isa Field
    @test eltype(ibg.immersed_boundary.bottom_height) === FT

    # Test summary function
    @test summary(ibg) isa String
    @test summary(ibg.immersed_boundary) isa String

    return nothing
end

function test_immersed_boundary_grid_with_array_bottom(FT, arch, boundary_type)
    underlying_grid = RectilinearGrid(arch, FT, size=(3, 3, 4), extent=(1, 1, 1))

    Nx, Ny = size(underlying_grid)[1:2]
    bottom_array = zeros(FT, Nx, Ny) .+ 0.3

    ib = boundary_type(bottom_array)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    @test architecture(ibg) === arch
    @test eltype(ibg) === FT
    @test size(ibg) == size(underlying_grid)

    return nothing
end

function test_immersed_boundary_grid_with_constant_bottom(FT, arch, boundary_type)
    underlying_grid = RectilinearGrid(arch, FT, size=(3, 3, 4), extent=(1, 1, 1))

    ib = boundary_type(-0.4)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    @test architecture(ibg) === arch
    @test eltype(ibg) === FT
    @test size(ibg) == size(underlying_grid)

    return nothing
end

function test_immersed_boundary_grid_with_stretched_grid(FT, arch, boundary_type)
    # Test with vertically stretched grid
    z_faces = collect(range(0, 1, length=6).^2)
    underlying_grid = RectilinearGrid(arch, FT, size=(3, 3, 5), x=(0, 1), y=(0, 1), z=z_faces)

    bottom_function(x, y) = 0.3
    ib = boundary_type(bottom_function)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    @test architecture(ibg) === arch
    @test eltype(ibg) === FT
    @test size(ibg) == size(underlying_grid)

    # Test that spacings are preserved
    @test all(zspacings(ibg, Center()) .== zspacings(underlying_grid, Center()))

    return nothing
end

#####
##### Grid fitting and immersed cell detection tests
#####

function test_grid_fitted_bottom_cell_detection(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 8), extent=(1, 1, 1))

    # Create a bottom at z = -0.5
    bottom_height = -0.5
    ib = GridFittedBottom(bottom_height)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    # Test that cells below the bottom are immersed
    for i in 1:4, j in 1:4
        for k in 1:4  # Lower half of domain
            z_center = znode(i, j, k, ibg, Center(), Center(), Center())
            expected_immersed = z_center ≤ bottom_height
            @test _immersed_cell(i, j, k, ibg.underlying_grid, ibg.immersed_boundary) == expected_immersed
        end
    end

    return nothing
end

function test_partial_cell_bottom_cell_detection(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 8), extent=(1, 1, 1))

    # Create a bottom at z = -0.5 with minimum fractional cell height
    bottom_height = -0.5
    ϵ = 0.2
    ib = PartialCellBottom(bottom_height; minimum_fractional_cell_height=ϵ)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    # Test immersed cell detection
    for i in 1:4, j in 1:4, k in 1:8
        # The partial cell criterion is different - need to check grid spacing
        immersed = _immersed_cell(i, j, k, ibg.underlying_grid, ibg.immersed_boundary)
        @test immersed isa Bool
    end

    return nothing
end

function test_bottom_height_field_consistency(FT, arch, boundary_type)
    underlying_grid = RectilinearGrid(arch, FT, size=(6, 6, 8), extent=(1, 1, 1))

    # Test that bottom height is correctly positioned on grid faces
    bottom_function(x, y) = -0.25 + 0.1 * sin(2π * x) * cos(2π * y)

    if boundary_type === GridFittedBottom
        ib = GridFittedBottom(bottom_function)
        ibg = ImmersedBoundaryGrid(underlying_grid, ib)

        # For GridFittedBottom, the bottom should be snapped to grid faces
        bottom_height = interior(ibg.immersed_boundary.bottom_height)
        zfaces = znodes(ibg, Face())

        for i in 1:size(ibg, 1), j in 1:size(ibg, 2)
            @test bottom_height[i, j, 1] ∈ zfaces
        end
    elseif boundary_type === PartialCellBottom
        ib = PartialCellBottom(bottom_function; minimum_fractional_cell_height=0.2)
        ibg = ImmersedBoundaryGrid(underlying_grid, ib)

        # For PartialCellBottom, bottom height should be clamped to domain
        bottom_height = interior(ibg.immersed_boundary.bottom_height)
        domain_bottom = minimum(znodes(ibg, Face()))
        domain_top = maximum(znodes(ibg, Face()))

        for i in 1:size(ibg, 1), j in 1:size(ibg, 2)
            @test domain_bottom ≤ bottom_height[i, j, 1] ≤ domain_top
        end
    end

    return nothing
end

#####
##### Grid spacing tests for PartialCellBottom
#####

function test_partial_cell_bottom_grid_spacings(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 8), extent=(1, 1, 1))

    bottom_height = -0.3
    ϵ = 0.2
    ib = PartialCellBottom(bottom_height; minimum_fractional_cell_height=ϵ)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    # Test that grid spacings are reasonable
    for i in 1:4, j in 1:4, k in 1:8
        Δz_partial = Δz(i, j, k, ibg, Center(), Center(), Center())
        Δz_original = Δz(i, j, k, underlying_grid, Center(), Center(), Center())

        @test Δz_partial > 0
        @test Δz_partial ≤ Δz_original  # Partial cells should be smaller or equal
    end

    return nothing
end

#####
##### Active cell and volume tests
#####

function test_immersed_volume_calculation(FT, arch, boundary_type)
    underlying_grid = RectilinearGrid(arch, FT, size=(10, 10, 10), extent=(1, 1, 1))

    # Create a bottom that removes exactly half the domain
    bottom_height = -0.5
    ib = boundary_type(bottom_height)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    # Test with a field to measure active volume
    c = CenterField(ibg)
    fill!(c, 1)

    # The dot product gives the number of active cells (for unit field values)
    active_volume = dot(c, c)
    total_cells = prod(size(underlying_grid))

    @test active_volume ≤ total_cells
    @test active_volume > 0

    # For this specific case, should be approximately half
    if boundary_type === GridFittedBottom
        @test abs(active_volume - total_cells/2) ≤ size(underlying_grid, 3)
    end

    return nothing
end

#####
##### GridFittedBoundary tests
#####

function test_grid_fitted_boundary_with_function(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 4), extent=(2, 2, 1))

    # Create a spherical immersed region
    mask_function(x, y, z) = x^2 + y^2 + z^2 ≤ 0.25
    ib = GridFittedBoundary(mask_function)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    # Basic grid properties
    @test architecture(ibg) === arch
    @test eltype(ibg) === FT
    @test size(ibg) == size(underlying_grid)
    @test halo_size(ibg) == halo_size(underlying_grid)
    @test topology(ibg) == topology(underlying_grid)

    # Test that immersed boundary was materialized
    @test ibg.immersed_boundary.mask isa Field
    @test eltype(ibg.immersed_boundary.mask) === Bool

    # Test immersed cell detection
    for i in 1:4, j in 1:4, k in 1:4
        x, y, z = xnode(i, j, k, ibg, Center(), Center(), Center()),
                  ynode(i, j, k, ibg, Center(), Center(), Center()),
                  znode(i, j, k, ibg, Center(), Center(), Center())
        expected_immersed = mask_function(x, y, z)
        @test _immersed_cell(i, j, k, ibg.underlying_grid, ibg.immersed_boundary) == expected_immersed
    end

    return nothing
end

function test_grid_fitted_boundary_with_array(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))

    # Create a 3D mask array
    Nx, Ny, Nz = size(underlying_grid)
    mask_array = zeros(Bool, Nx, Ny, Nz)

    # Mark center cell as immersed
    mask_array[2, 2, 2] = true

    ib = GridFittedBoundary(mask_array)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    @test architecture(ibg) === arch
    @test eltype(ibg) === FT
    @test size(ibg) == size(underlying_grid)

    # Test that the center cell is immersed
    @test _immersed_cell(2, 2, 2, ibg.underlying_grid, ibg.immersed_boundary) == true

    # Test that corner cells are not immersed
    @test _immersed_cell(1, 1, 1, ibg.underlying_grid, ibg.immersed_boundary) == false
    @test _immersed_cell(3, 3, 3, ibg.underlying_grid, ibg.immersed_boundary) == false

    return nothing
end

#####
##### Show function tests
#####

function test_immersed_boundary_grid_show(FT, arch, boundary_type)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 4), extent=(1, 1, 1))

    bottom_function(x, y) = 0.2
    ib = boundary_type(bottom_function)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    @test try
        show(ibg); println()
        show(ibg.immersed_boundary); println()
        true
    catch err
        println("error in show functions")
        println(sprint(showerror, err))
        false
    end

    return nothing
end

#####
##### Error condition tests
#####

function test_immersed_boundary_grid_errors(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 4), extent=(1, 1, 1))

    @test_nowarn PartialCellBottom(0.0; minimum_fractional_cell_height=0.1)
    @test_nowarn PartialCellBottom(0.0; minimum_fractional_cell_height=1.0)

    @test_nowarn ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(0.0))
    @test_nowarn ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(0.0))

    return nothing
end

#####
##### Flat topology tests
#####

function test_immersed_boundary_grid_flat_topologies(FT, arch, boundary_type)
    # Test with different flat topologies
    topologies = [
        (Flat, Periodic, Bounded),
        (Periodic, Flat, Bounded),
        (Flat, Flat, Bounded)
    ]

    for topo in topologies
        if topo == (Flat, Periodic, Bounded) || topo == (Periodic, Flat, Bounded)
            grid_size = (5, 5)
            extent = (1, 1)
        elseif topo == (Flat, Flat, Bounded)
            grid_size = 5
            extent = 1
        end

        underlying_grid = RectilinearGrid(arch, FT, topology=topo, size=grid_size, extent=extent)
        bottom_height = -0.3
        ib = boundary_type(bottom_height)
        ibg = ImmersedBoundaryGrid(underlying_grid, ib)

        @test architecture(ibg) === arch
        @test eltype(ibg) === FT
        @test topology(ibg) == topo
    end

    return nothing
end

#####
##### Node and spacing consistency tests
#####

function test_immersed_boundary_grid_nodes_and_spacings(FT, arch, boundary_type)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 6), extent=(2, 2, 1))

    bottom_function(x, y) = 0.1 * sin(π * x) * cos(π * y)

    if boundary_type === GridFittedBottom
        ib = GridFittedBottom(bottom_function)
    elseif boundary_type === PartialCellBottom
        ib = PartialCellBottom(bottom_function; minimum_fractional_cell_height=0.2)
    end

    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    # Test that node functions work
    @test xnode(2, 1, 1, ibg, Center(), Center(), Center()) isa FT
    @test ynode(1, 2, 1, ibg, Center(), Center(), Center()) isa FT
    @test znode(1, 1, 2, ibg, Center(), Center(), Center()) isa FT

    # Test spacing functions
    @test Δx(1, 1, 1, ibg, Face(), Center(), Center()) isa FT
    @test Δy(1, 1, 1, ibg, Center(), Face(), Center()) isa FT
    @test Δz(1, 1, 1, ibg, Center(), Center(), Face()) isa FT

    # Test that horizontal spacings are preserved from underlying grid
    @test Δx(2, 2, 2, ibg, Face(), Center(), Center()) == Δx(2, 2, 2, underlying_grid, Face(), Center(), Center())
    @test Δy(2, 2, 2, ibg, Center(), Face(), Center()) == Δy(2, 2, 2, underlying_grid, Center(), Face(), Center())

    return nothing
end

#####
##### Main test sets
#####

@testset "Immersed Boundary Grids" begin
    @info "Testing immersed boundary grids..."

    @testset "Basic construction" begin
        @info "  Testing basic immersed boundary grid construction..."

        for arch in archs, FT in float_types
            for boundary_type in (GridFittedBottom, PartialCellBottom)
                @testset "Construction [$FT, $(typeof(arch)), $boundary_type]" begin
                    test_immersed_boundary_grid_construction(FT, arch, boundary_type)
                    test_immersed_boundary_grid_with_array_bottom(FT, arch, boundary_type)
                    test_immersed_boundary_grid_with_constant_bottom(FT, arch, boundary_type)
                    test_immersed_boundary_grid_with_stretched_grid(FT, arch, boundary_type)
                end
            end
        end
    end

    @testset "Grid fitting and cell detection" begin
        @info "  Testing grid fitting and immersed cell detection..."

        for arch in archs, FT in float_types
            @testset "Cell detection [$FT, $(typeof(arch))]" begin
                test_grid_fitted_bottom_cell_detection(FT, arch)
                test_partial_cell_bottom_cell_detection(FT, arch)

                for boundary_type in (GridFittedBottom, PartialCellBottom)
                    test_bottom_height_field_consistency(FT, arch, boundary_type)
                end
            end
        end
    end

    @testset "Grid spacings and metrics" begin
        @info "  Testing grid spacings and metrics..."
        for arch in archs, FT in float_types
            @testset "Spacings [$FT, $(typeof(arch))]" begin
                test_partial_cell_bottom_grid_spacings(FT, arch)

                for boundary_type in (GridFittedBottom, PartialCellBottom)
                    test_immersed_volume_calculation(FT, arch, boundary_type)
                    test_immersed_boundary_grid_nodes_and_spacings(FT, arch, boundary_type)
                end
            end
        end
    end

    @testset "GridFittedBoundary" begin
        @info "  Testing GridFittedBoundary..."

        for arch in archs, FT in float_types
            @testset "GridFittedBoundary [$FT, $(typeof(arch))]" begin
                test_grid_fitted_boundary_with_function(FT, arch)
                test_grid_fitted_boundary_with_array(FT, arch)
            end
        end
    end

    @testset "Show functions" begin
        @info "  Testing show functions..."
        for arch in archs, FT in float_types
            for boundary_type in (GridFittedBottom, PartialCellBottom)
                @testset "Show [$FT, $(typeof(arch)), $boundary_type]" begin
                    test_immersed_boundary_grid_show(FT, arch, boundary_type)
                end
            end
        end
    end

    @testset "Error conditions" begin
        @info "  Testing error conditions..."

        for arch in archs, FT in float_types
            @testset "Errors [$FT, $(typeof(arch))]" begin
                test_immersed_boundary_grid_errors(FT, arch)
            end
        end
    end

    @testset "Flat topologies" begin
        @info "  Testing flat topologies..."

        for arch in archs, FT in float_types
            for boundary_type in (GridFittedBottom, PartialCellBottom)
                @testset "Flat [$FT, $(typeof(arch)), $boundary_type]" begin
                    test_immersed_boundary_grid_flat_topologies(FT, arch, boundary_type)
                end
            end
        end
    end
end
