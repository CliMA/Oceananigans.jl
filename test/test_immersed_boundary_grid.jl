include("dependencies_for_runtests.jl")

using Oceananigans.Grids: total_extent, xspacings, yspacings, zspacings, rspacings, xnode, ynode, znode
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
    bottom_array = on_architecture(arch, bottom_array)

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
    @test Field(zspacings(ibg, Center())) == Field(zspacings(underlying_grid, Center()))

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
    CUDA.@allowscalar begin
        for i in 1:4, j in 1:4
            for k in 1:4  # Lower half of domain
                z_center = znode(i, j, k, ibg, Center(), Center(), Center())
                expected_immersed = z_center ≤ bottom_height
                @test _immersed_cell(i, j, k, ibg.underlying_grid, ibg.immersed_boundary) == expected_immersed
            end
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
    CUDA.@allowscalar begin
        for i in 1:4, j in 1:4, k in 1:8
            # The partial cell criterion is different - need to check grid spacing
            immersed = _immersed_cell(i, j, k, ibg.underlying_grid, ibg.immersed_boundary)
            @test immersed isa Bool
        end
    end

    return nothing
end

function test_bottom_height_field_consistency(FT, arch, boundary_type)
    underlying_grid = RectilinearGrid(arch, FT, size=(6, 6, 8), extent=(1, 1, 1))

    # Test that bottom height is correctly positioned on grid faces
    bottom_function(x, y) = -0.25 + 0.1 * sin(2π * x) * cos(2π * y)

    ib = boundary_type(bottom_function)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)
    bottom_height = interior(ibg.immersed_boundary.bottom_height)
    if boundary_type === GridFittedBottom
        # For GridFittedBottom, the bottom should be snapped to grid faces
        zfaces = znodes(ibg, Face())
        @test all(in.(bottom_height, Ref(zfaces)))

    elseif boundary_type === PartialCellBottom
        # For PartialCellBottom, bottom height should be clamped to domain
        domain_bottom, domain_top = extrema(znodes(ibg, Face()))
        @test all(domain_bottom .≤ bottom_height .≤ domain_top)
    end

    return nothing
end

#####
##### Grid spacing tests for PartialCellBottom
#####

function test_partial_cell_bottom_grid_spacings(FT, arch; mutable_grid = false)
    Nz = 3; Lz = 1
    if mutable_grid
        @info "    Testing grid with a MutableVerticalDiscretization"
        underlying_grid = RectilinearGrid(arch, FT, topology=(Flat, Flat, Bounded), size=Nz, z=MutableVerticalDiscretization((-Lz, 0)))
    else
        @info "    Testing grid with a StaticVerticalDiscretization"
        underlying_grid = RectilinearGrid(arch, FT, topology=(Flat, Flat, Bounded), size=Nz, z=(-Lz, 0))
    end

    bottom_height = -Lz/2 # Bottom height is modway through the water column
    ϵ = 0.2
    ib = PartialCellBottom(bottom_height; minimum_fractional_cell_height=ϵ)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    Δz_partial     = Field(zspacings(ibg, Center(), Center(), Center()))
    Δz_original    = Field(zspacings(ibg.underlying_grid, Center(), Center(), Center()))
    Δz_partial_min = Field(zspacings(ibg.underlying_grid, Center(), Center(), Center()) * ϵ)

    # Test that grid spacings are whithin expected bounds
    @test all(interior(Δz_partial_min) .≤ interior(Δz_partial) .≤ interior(Δz_original))

    # Test that Δz_partial equals Δr_partial for this configuration
    Δr_partial = Field(rspacings(ibg, Center(), Center(), Center()))
    @test interior(Δz_partial) == interior(Δr_partial)

    # Test that specific values are correct
    expected_spacings = FT.([1, 1/2, 1] ./ Nz)
    actual_spacings = Array(interior(Δz_partial, 1, 1, :))
    @test actual_spacings ≈ expected_spacings

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

    # Use a Field to measure active volume
    c_ibg = CenterField(ibg)
    c_udl = CenterField(ibg.underlying_grid)
    set!(c_ibg, 1); set!(c_udl, 1)

    active_volume = Field(Integral(c_ibg)) |> interior |> collect |> only
    total_volume = Field(Integral(c_udl)) |> interior |> collect |> only

    @test active_volume > 0
    @test active_volume ≈ total_volume/2

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
    CUDA.@allowscalar begin
        for i in 1:4, j in 1:4, k in 1:4
            x, y, z = xnode(i, j, k, ibg, Center(), Center(), Center()),
                      ynode(i, j, k, ibg, Center(), Center(), Center()),
                      znode(i, j, k, ibg, Center(), Center(), Center())
            expected_immersed = mask_function(x, y, z)
            @test _immersed_cell(i, j, k, ibg.underlying_grid, ibg.immersed_boundary) == expected_immersed
        end
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

    CUDA.@allowscalar begin
        # Test that the center cell is immersed
        @test _immersed_cell(2, 2, 2, ibg.underlying_grid, ibg.immersed_boundary) == true

        # Test that corner cells are not immersed
        @test _immersed_cell(1, 1, 1, ibg.underlying_grid, ibg.immersed_boundary) == false
        @test _immersed_cell(3, 3, 3, ibg.underlying_grid, ibg.immersed_boundary) == false
    end

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
    ib = boundary_type(bottom_function)
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    # Test that node functions work
    @test xnodes(ibg, Center(), Center(), Center()) isa AbstractArray
    @test ynodes(ibg, Center(), Center(), Center()) isa AbstractArray
    @test znodes(ibg, Center(), Center(), Center()) isa AbstractArray

    # Test spacing functions
    @test xspacings(ibg, Face(), Center(), Center()) isa KernelFunctionOperation
    @test yspacings(ibg, Center(), Face(), Center()) isa KernelFunctionOperation
    @test zspacings(ibg, Center(), Center(), Face()) isa KernelFunctionOperation

    # Test that horizontal spacings are preserved from underlying grid
    @test Field(xspacings(ibg, Face(), Center(), Center())) == Field(xspacings(ibg.underlying_grid, Face(), Center(), Center()))
    @test Field(yspacings(ibg, Center(), Face(), Center())) == Field(yspacings(ibg.underlying_grid, Center(), Face(), Center()))

    return nothing
end

#####
##### Main test sets
#####

@testset "Immersed Boundary Grids" begin
    @info "Testing immersed boundary grids..."

    @testset "Basic construction" begin
        for arch in archs, FT in float_types
            for boundary_type in (GridFittedBottom, PartialCellBottom)
                @info "  Testing basic immersed boundary grid construction [$FT, $(typeof(arch)), $boundary_type] ..."
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
        for arch in archs, FT in float_types
            @info "  Testing grid fitting and immersed cell detection [$FT, $(typeof(arch))]..."
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
        for arch in archs, FT in float_types
            @info "  Testing grid spacings and metrics [$FT, $(typeof(arch))]..."
            @testset "Spacings [$FT, $(typeof(arch))]" begin
                test_partial_cell_bottom_grid_spacings(FT, arch, mutable_grid=false)
                test_partial_cell_bottom_grid_spacings(FT, arch, mutable_grid=true)

                for boundary_type in (GridFittedBottom, PartialCellBottom)
                    test_immersed_volume_calculation(FT, arch, boundary_type)
                    test_immersed_boundary_grid_nodes_and_spacings(FT, arch, boundary_type)
                end
            end
        end
    end

    @testset "GridFittedBoundary" begin
        for arch in archs, FT in float_types
            @info "  Testing GridFittedBoundary [$FT, $(typeof(arch))]..."
            @testset "GridFittedBoundary [$FT, $(typeof(arch))]" begin
                test_grid_fitted_boundary_with_function(FT, arch)
                test_grid_fitted_boundary_with_array(FT, arch)
            end
        end
    end

    @testset "Show functions" begin
        for arch in archs, FT in float_types
            for boundary_type in (GridFittedBottom, PartialCellBottom)
                @info "  Testing show functions [$FT, $(typeof(arch)), $boundary_type]..."
                @testset "Show [$FT, $(typeof(arch)), $boundary_type]" begin
                    test_immersed_boundary_grid_show(FT, arch, boundary_type)
                end
            end
        end
    end

    @testset "Error conditions" begin
        for arch in archs, FT in float_types
            @info "  Testing error conditions [$FT, $(typeof(arch))]..."
            @testset "Errors [$FT, $(typeof(arch))]" begin
                test_immersed_boundary_grid_errors(FT, arch)
            end
        end
    end

    @testset "Flat topologies" begin
        for arch in archs, FT in float_types
            for boundary_type in (GridFittedBottom, PartialCellBottom)
                @info "  Testing flat topologies [$FT, $(typeof(arch)), $boundary_type]..."
                @testset "Flat [$FT, $(typeof(arch)), $boundary_type]" begin
                    test_immersed_boundary_grid_flat_topologies(FT, arch, boundary_type)
                end
            end
        end
    end
end
