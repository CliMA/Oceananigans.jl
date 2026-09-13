include("dependencies_for_runtests.jl")

using Oceananigans.Grids: znodes, znode, rspacings, zspacings, constructor_arguments,
                          static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ,
                          peripheral_node
using Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶠᶜᶜ, Δrᶜᶠᶜ, Δrᶠᶠᶜ, Δzᶜᶜᶜ, Δzᶠᶜᶜ
using Oceananigans.ImmersedBoundaries: ShavedCellBottom, PartialCellBottom, GridFittedBottom,
                                       _immersed_cell, bottom_height_interior, corner_bottom_height_field

const c = Center()
const f = Face()

#####
##### Geometry
#####

function test_shaved_cell_reduces_to_partial_cells(FT, arch)
    Nz = 8
    Lz = 1
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, Nz), extent=(1, 1, Lz))

    # A bottom that is flat, and aligned with an interface, leaves nothing to shave.
    aligned_bottom = -Lz/2

    fitted = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(aligned_bottom))
    partial = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(aligned_bottom))
    shaved = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(aligned_bottom))

    @allowscalar begin
        for i in 1:4, j in 1:4, k in 1:Nz
            @test Δrᶜᶜᶜ(i, j, k, shaved) == Δrᶜᶜᶜ(i, j, k, partial) == Δrᶜᶜᶜ(i, j, k, fitted)
            @test Δrᶠᶜᶜ(i, j, k, shaved) == Δrᶠᶜᶜ(i, j, k, partial)
            @test Δrᶜᶠᶜ(i, j, k, shaved) == Δrᶜᶠᶜ(i, j, k, partial)
        end

        # A bottom halfway through a cell gives the same cell volumes as partial cells, since the
        # trapezoid mean of a flat surface is the surface itself.
        unaligned_bottom = -Lz/2 - Lz/(2Nz)
        partial = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(unaligned_bottom))
        shaved = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(unaligned_bottom))

        for i in 1:4, j in 1:4, k in 1:Nz
            @test Δrᶜᶜᶜ(i, j, k, shaved) ≈ Δrᶜᶜᶜ(i, j, k, partial)
        end
    end

    return nothing
end

function test_shaved_cell_trapezoid_volumes(FT, arch)
    Nx = 16
    Nz = 8
    Lx = 1
    Lz = 1
    underlying_grid = RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded),
                                      size=(Nx, Nz), halo=(4, 4), x=(0, Lx), z=(-Lz, 0))

    # A linear bottom, whose cell means are exactly its values at the cell centers.
    slope(x) = -Lz + Lz/2 * x / Lx + Lz/8
    ibg = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(slope, minimum_fractional_cell_height=0))

    xᶜ = xnodes(underlying_grid, c)
    xᶠ = xnodes(underlying_grid, f)
    zᶠ = znodes(underlying_grid, f)
    Δz = Lz / Nz

    @allowscalar begin
        # The column depth is the depth of the analytic bottom, both at centers and at faces.
        for i in 2:Nx-1
            @test static_column_depthᶜᶜᵃ(i, 1, ibg) ≈ -slope(xᶜ[i])
        end

        # A lateral face is cut where the bottom crosses it, not where the neighboring cell ends.
        for i in 2:Nx, k in 1:Nz
            peripheral_node(i, 1, k, ibg, f, c, c) && continue
            expected = clamp(zᶠ[k+1] - slope(xᶠ[i]), 0, Δz)
            @test Δrᶠᶜᶜ(i, 1, k, ibg) ≈ expected
        end

        # Which is strictly more open than the partial cell face over a slope like this one.
        partial = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(slope, minimum_fractional_cell_height=0))
        shaved_throat = sum(Δrᶠᶜᶜ(i, 1, k, ibg) for i in 2:Nx, k in 1:Nz if !peripheral_node(i, 1, k, ibg, f, c, c))
        partial_throat = sum(Δrᶠᶜᶜ(i, 1, k, partial) for i in 2:Nx, k in 1:Nz if !peripheral_node(i, 1, k, partial, f, c, c))
        @test shaved_throat > partial_throat
    end

    return nothing
end

function test_shaved_cell_column_depth_consistency(FT, arch)
    Nx = 16
    Nz = 8
    underlying_grid = RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded),
                                      size=(Nx, Nz), halo=(4, 4), x=(0, 1), z=(-1, 0))

    # A bumpy bottom, so that the wet columns have unequal depths.
    bumpy(x) = -0.9 + 0.4 * sin(3π * x) + 0.1 * cos(11π * x)
    ibg = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(bumpy))

    # The barotropic mode integrates the same water column that the shaved face heights describe.
    @allowscalar begin
        for i in 2:Nx-1
            Σᶜ = sum(Δrᶜᶜᶜ(i, 1, k, ibg) for k in 1:Nz if !peripheral_node(i, 1, k, ibg, c, c, c); init=zero(FT))
            @test Σᶜ ≈ static_column_depthᶜᶜᵃ(i, 1, ibg)

            Σᶠ = sum(Δrᶠᶜᶜ(i, 1, k, ibg) for k in 1:Nz if !peripheral_node(i, 1, k, ibg, f, c, c); init=zero(FT))
            @test Σᶠ ≈ static_column_depthᶠᶜᵃ(i, 1, ibg)
        end
    end

    return nothing
end

function test_shaved_cell_minimum_height(FT, arch)
    Nx = 16
    Nz = 8
    Lz = 1
    underlying_grid = RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded),
                                      size=(Nx, Nz), halo=(4, 4), x=(0, 1), z=(-Lz, 0))

    Δz = Lz / Nz

    for ϵ in (0.1, 0.2, 0.5)
        bumpy(x) = -0.9 + 0.4 * sin(3π * x) + 0.1 * cos(11π * x)
        ibg = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(bumpy, minimum_fractional_cell_height=ϵ))

        @allowscalar begin
            for i in 2:Nx-1, k in 1:Nz
                peripheral_node(i, 1, k, ibg, c, c, c) && continue
                @test Δrᶜᶜᶜ(i, 1, k, ibg) ≥ ϵ * Δz - 10eps(FT)
                @test Δrᶜᶜᶜ(i, 1, k, ibg) ≤ Δz + 10eps(FT)
            end

            for i in 2:Nx, k in 1:Nz
                peripheral_node(i, 1, k, ibg, f, c, c) && continue
                @test Δrᶠᶜᶜ(i, 1, k, ibg) ≥ ϵ * Δz - 10eps(FT)
                @test Δrᶠᶜᶜ(i, 1, k, ibg) ≤ Δz + 10eps(FT)
            end
        end
    end

    return nothing
end

function test_shaved_cell_flat_topologies(FT, arch)
    Nz = 8
    bumpy(ξ) = -0.9 + 0.4 * sin(3π * ξ)

    # In a Flat direction the staggered metrics collapse onto the centered ones.
    yflat = RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded), size=(16, Nz), halo=(4, 4), x=(0, 1), z=(-1, 0))
    yflat_ibg = ImmersedBoundaryGrid(yflat, ShavedCellBottom(bumpy))

    xflat = RectilinearGrid(arch, FT, topology=(Flat, Bounded, Bounded), size=(16, Nz), halo=(4, 4), y=(0, 1), z=(-1, 0))
    xflat_ibg = ImmersedBoundaryGrid(xflat, ShavedCellBottom(bumpy))

    @allowscalar begin
        for i in 1:16, k in 1:Nz
            @test Δrᶜᶠᶜ(i, 1, k, yflat_ibg) == Δrᶜᶜᶜ(i, 1, k, yflat_ibg)
            @test Δrᶠᶠᶜ(i, 1, k, yflat_ibg) == Δrᶠᶜᶜ(i, 1, k, yflat_ibg)
            @test Δrᶠᶜᶜ(1, i, k, xflat_ibg) == Δrᶜᶜᶜ(1, i, k, xflat_ibg)
            @test Δrᶠᶠᶜ(1, i, k, xflat_ibg) == Δrᶜᶠᶜ(1, i, k, xflat_ibg)
        end

        for i in 1:16
            @test static_column_depthᶜᶠᵃ(i, 1, yflat_ibg) == static_column_depthᶜᶜᵃ(i, 1, yflat_ibg)
            @test static_column_depthᶠᶜᵃ(1, i, xflat_ibg) == static_column_depthᶜᶜᵃ(1, i, xflat_ibg)
        end
    end

    return nothing
end

#####
##### Construction, reconstruction and comparison
#####

function test_shaved_cell_bottom_height_sources(FT, arch)
    Nx = Ny = 6
    Nz = 8
    underlying_grid = RectilinearGrid(arch, FT, size=(Nx, Ny, Nz), halo=(4, 4, 4), extent=(1, 1, 1))

    bumpy(x, y) = -0.5 + 0.1 * sin(2π * x) * cos(2π * y)

    from_function = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(bumpy))

    # A Field sampled at cell centers is interpolated to the corners, which differs from sampling the
    # function there, but must still give a valid and clamped bottom.
    bottom_field = Field{Center, Center, Nothing}(underlying_grid)
    set!(bottom_field, bumpy)
    from_field = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(bottom_field))

    domain_bottom, domain_top = extrema(znodes(underlying_grid, f))
    for ibg in (from_function, from_field)
        heights = bottom_height_interior(ibg.immersed_boundary.bottom_height)
        @test all(domain_bottom .≤ Array(heights) .≤ domain_top)
    end

    # A corner Field is the primary geometry, so it round-trips exactly.
    corners = corner_bottom_height_field(from_function)
    from_corners = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(corners))
    @test from_corners.immersed_boundary == from_function.immersed_boundary

    return nothing
end

function test_shaved_cell_reconstruction(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(6, 6, 8), halo=(4, 4, 4), extent=(1, 1, 1))
    bumpy(x, y) = -0.5 + 0.1 * sin(2π * x) * cos(2π * y)

    original = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(bumpy, minimum_fractional_cell_height=0.3))

    grid_args, grid_kwargs, immersed_args = constructor_arguments(original)
    @test :bottom_height in keys(immersed_args)
    @test :minimum_fractional_cell_height in keys(immersed_args)

    reconstructed_underlying_grid = RectilinearGrid(values(grid_args)...; grid_kwargs...)
    reconstructed = ImmersedBoundaryGrid(reconstructed_underlying_grid,
                                         ShavedCellBottom(immersed_args[:bottom_height],
                                                          immersed_args[:minimum_fractional_cell_height]))

    @test reconstructed == original
    @test size(reconstructed) == size(original)
    @test eltype(reconstructed) == eltype(original)

    return nothing
end

function test_shaved_cell_equality(FT, arch)
    ib1 = ShavedCellBottom(-0.5; minimum_fractional_cell_height=0.2)
    ib2 = ShavedCellBottom(-0.5; minimum_fractional_cell_height=0.2)
    ib3 = ShavedCellBottom(-0.4; minimum_fractional_cell_height=0.2)
    ib4 = ShavedCellBottom(-0.5; minimum_fractional_cell_height=0.1)

    @test ib1 == ib2
    @test ib1 != ib3
    @test ib1 != ib4

    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 8), extent=(1, 1, 1))
    @test ImmersedBoundaryGrid(underlying_grid, ib1) == ImmersedBoundaryGrid(underlying_grid, ib2)
    @test ImmersedBoundaryGrid(underlying_grid, ib1) != ImmersedBoundaryGrid(underlying_grid, ib3)

    return nothing
end

function test_shaved_cell_architecture_change(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(6, 6, 8), halo=(4, 4, 4), extent=(1, 1, 1))
    bumpy(x, y) = -0.5 + 0.1 * sin(2π * x) * cos(2π * y)
    ibg = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(bumpy))

    on_cpu = on_architecture(CPU(), ibg)
    @test on_cpu.immersed_boundary == on_architecture(CPU(), ibg.immersed_boundary)
    @test on_architecture(arch, on_cpu) == ibg

    return nothing
end

function test_shaved_cell_show(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(6, 6, 8), extent=(1, 1, 1))
    ibg = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(-0.5))

    @test occursin("ShavedCellBottom", summary(ibg.immersed_boundary))
    @test occursin("minimum_fractional_cell_height", sprint(show, ibg.immersed_boundary))

    return nothing
end

#####
##### Time stepping
#####

function test_shaved_cell_time_stepping(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded),
                                      size=(32, 16), halo=(5, 5), x=(0, 1000), z=(-100, 0))

    slope(x) = -100 + 80 * clamp((x - 200) / 400, 0, 1)
    ibg = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(slope))

    model = HydrostaticFreeSurfaceModel(ibg; tracers=:b, buoyancy=BuoyancyTracer(),
                                        momentum_advection=WENO(), tracer_advection=WENO())

    set!(model, b = (x, z) -> 1e-5 * z - 1e-4 * exp(-(x - 100)^2 / 1e4))

    for _ in 1:20
        time_step!(model, 10)
    end

    @test !any(isnan, Array(interior(model.tracers.b)))
    @test !any(isnan, Array(interior(model.velocities.u)))
    @test !any(isnan, Array(interior(model.velocities.w)))

    return nothing
end

#####
##### Run
#####

@testset "ShavedCellBottom" begin
    @info "Testing ShavedCellBottom..."

    @testset "Geometry" begin
        for arch in archs, FT in float_types
            @info "  Testing shaved cell geometry [$FT, $(typeof(arch))]..."
            test_shaved_cell_reduces_to_partial_cells(FT, arch)
            test_shaved_cell_trapezoid_volumes(FT, arch)
            test_shaved_cell_column_depth_consistency(FT, arch)
            test_shaved_cell_minimum_height(FT, arch)
            test_shaved_cell_flat_topologies(FT, arch)
        end
    end

    @testset "Construction and reconstruction" begin
        for arch in archs, FT in float_types
            @info "  Testing shaved cell construction [$FT, $(typeof(arch))]..."
            test_shaved_cell_bottom_height_sources(FT, arch)
            test_shaved_cell_reconstruction(FT, arch)
            test_shaved_cell_equality(FT, arch)
            test_shaved_cell_architecture_change(FT, arch)
            test_shaved_cell_show(FT, arch)
        end
    end

    @testset "Time stepping" begin
        for arch in archs, FT in float_types
            @info "  Testing shaved cell time stepping [$FT, $(typeof(arch))]..."
            test_shaved_cell_time_stepping(FT, arch)
        end
    end
end
