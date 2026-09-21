include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: znodes, peripheral_node, with_halo,
                          static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ
using Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶠᶜᶜ, Δrᶜᶠᶜ, Δrᶠᶠᶜ
using Oceananigans.ImmersedBoundaries: immersed_cell

const c = Center()
const f = Face()

bumpy(ξ) = -0.9 + 0.4 * sin(3π * ξ) + 0.1 * cos(11π * ξ)
slice_grid(FT, arch; Nx=16, Nz=8) = RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded),
                                                    size=(Nx, Nz), halo=(4, 4), x=(0, 1), z=(-1, 0))

function test_shaved_cell_reduces_to_partial_cells(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 8), extent=(1, 1, 1))
    Nx, Ny, Nz = size(underlying_grid)

    # A flat bottom leaves nothing to shave, whether or not it falls on an interface.
    for bottom_height in (-1/2, -1/2 - 1/2Nz)
        partial = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height))
        shaved = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(bottom_height))

        @allowscalar for i in 1:Nx, j in 1:Ny, k in 1:Nz
            @test Δrᶜᶜᶜ(i, j, k, shaved) ≈ Δrᶜᶜᶜ(i, j, k, partial)
            @test Δrᶠᶜᶜ(i, j, k, shaved) ≈ Δrᶠᶜᶜ(i, j, k, partial)
            @test Δrᶜᶠᶜ(i, j, k, shaved) ≈ Δrᶜᶠᶜ(i, j, k, partial)
        end
    end

    return nothing
end

function test_shaved_cell_land_stays_out_of_the_sea(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(6, 6, 8), extent=(1, 1, 1))
    Nx, Ny, Nz = size(underlying_grid)

    island_bottom_height(land_height, sea_height) = [2 ≤ i ≤ 4 && 3 ≤ j ≤ 4 ? land_height : sea_height for i in 1:Nx, j in 1:Ny]

    # An island on a flat sea floor stays dry, and the height given to the land does not reach the sea.
    for sea_height in (-1/2, -1/2 - 1/2Nz)
        partial = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(island_bottom_height(100, sea_height)))
        shaved = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(island_bottom_height(100, sea_height)))
        flush_shaved = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(island_bottom_height(0, sea_height)))

        @test shaved.immersed_boundary == flush_shaved.immersed_boundary

        @allowscalar for i in 1:Nx, j in 1:Ny, k in 1:Nz
            @test immersed_cell(i, j, k, shaved) == immersed_cell(i, j, k, partial)
        end
    end

    return nothing
end

function test_shaved_cell_geometry(FT, arch)
    Nx, Nz, Lz = 16, 8, 1
    underlying_grid = slice_grid(FT, arch; Nx, Nz)
    Δz = Lz / Nz

    slope(x) = -Lz + Lz/2 * x + Lz/8
    ibg = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(slope, minimum_fractional_cell_height=0))

    xᶠ = xnodes(underlying_grid, f)
    zᶠ = znodes(underlying_grid, f)

    @allowscalar begin
        # A lateral face is cut where the bottom crosses it, not where the neighboring cell ends.
        for i in 2:Nx, k in 1:Nz
            peripheral_node(i, 1, k, ibg, f, c, c) && continue
            @test Δrᶠᶜᶜ(i, 1, k, ibg) ≈ clamp(zᶠ[k+1] - slope(xᶠ[i]), 0, Δz)
        end

        # Which opens a wider throat over a slope than partial cells do.
        partial = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(slope, minimum_fractional_cell_height=0))
        throat(grid) = sum(Δrᶠᶜᶜ(i, 1, k, grid) for i in 2:Nx, k in 1:Nz if !peripheral_node(i, 1, k, grid, f, c, c))
        @test throat(ibg) > throat(partial)
    end

    return nothing
end

function test_shaved_cells_are_bounded_by_their_faces(FT, arch)
    Nx, Nz, Lz = 16, 8, 1
    Δz = Lz / Nz
    ϵ = 0.2
    ibg = ImmersedBoundaryGrid(slice_grid(FT, arch; Nx, Nz), ShavedCellBottom(bumpy, minimum_fractional_cell_height=ϵ))

    opening(i, k) = peripheral_node(i, 1, k, ibg, f, c, c) ? zero(FT) : Δrᶠᶜᶜ(i, 1, k, ibg)

    @allowscalar for i in 2:Nx-1
        kᵇ = findfirst(k -> !immersed_cell(i, 1, k, ibg), 1:Nz)
        isnothing(kᵇ) && continue
        @test Δrᶜᶜᶜ(i, 1, kᵇ, ibg) ≈ (opening(i, kᵇ) + opening(i+1, kᵇ)) / 2
        @test Δrᶜᶜᶜ(i, 1, kᵇ, ibg) ≥ ϵ * Δz - 10eps(FT)
    end

    return nothing
end

function test_shaved_cell_column_depth_consistency(FT, arch)
    Nx, Nz, Lz = 16, 8, 1
    Δz = Lz / Nz
    ϵ = 0.2
    ibg = ImmersedBoundaryGrid(slice_grid(FT, arch; Nx, Nz), ShavedCellBottom(bumpy, minimum_fractional_cell_height=ϵ))

    # The barotropic mode integrates the water column that the shaved face heights describe.
    @allowscalar for i in 2:Nx-1
        Σᶜ = sum(Δrᶜᶜᶜ(i, 1, k, ibg) for k in 1:Nz if !peripheral_node(i, 1, k, ibg, c, c, c); init=zero(FT))
        @test Σᶜ ≈ static_column_depthᶜᶜᵃ(i, 1, ibg)

        Σᶠ = sum(Δrᶠᶜᶜ(i, 1, k, ibg) for k in 1:Nz if !peripheral_node(i, 1, k, ibg, f, c, c); init=zero(FT))
        @test Σᶠ ≈ static_column_depthᶠᶜᵃ(i, 1, ibg)
    end

    @allowscalar for i in 2:Nx, k in 1:Nz
        peripheral_node(i, 1, k, ibg, f, c, c) && continue
        @test ϵ * Δz - 10eps(FT) ≤ Δrᶠᶜᶜ(i, 1, k, ibg) ≤ Δz + 10eps(FT)
    end

    return nothing
end

function test_shaved_cell_flat_topologies(FT, arch)
    yflat = ImmersedBoundaryGrid(slice_grid(FT, arch), ShavedCellBottom(bumpy))

    xflat_grid = RectilinearGrid(arch, FT, topology=(Flat, Bounded, Bounded), size=(16, 8), halo=(4, 4), y=(0, 1), z=(-1, 0))
    xflat = ImmersedBoundaryGrid(xflat_grid, ShavedCellBottom(bumpy))
    N, _, Nz = size(xflat_grid)

    # In a Flat direction the staggered metrics collapse onto the centered ones.
    @allowscalar for i in 1:N, k in 1:Nz
        @test Δrᶜᶠᶜ(i, 1, k, yflat) == Δrᶜᶜᶜ(i, 1, k, yflat)
        @test Δrᶠᶠᶜ(i, 1, k, yflat) == Δrᶠᶜᶜ(i, 1, k, yflat)
        @test Δrᶠᶜᶜ(1, i, k, xflat) == Δrᶜᶜᶜ(1, i, k, xflat)
        @test Δrᶠᶠᶜ(1, i, k, xflat) == Δrᶜᶠᶜ(1, i, k, xflat)
    end

    @allowscalar for i in 1:N
        @test static_column_depthᶜᶠᵃ(i, 1, yflat) == static_column_depthᶜᶜᵃ(i, 1, yflat)
        @test static_column_depthᶠᶜᵃ(1, i, xflat) == static_column_depthᶜᶜᵃ(1, i, xflat)
    end

    return nothing
end

function test_shaved_cell_with_halo(FT, arch)
    underlying_grid = LatitudeLongitudeGrid(arch, FT, size=(32, 24, 6), halo=(3, 3, 3), longitude=(0, 360), latitude=(-60, 60), z=(-1000, 0))
    bottom(λ, φ) = -600 - 300 * sin(2π * λ / 360) * cos(2π * φ / 120)
    shaved_cell_bottom = ShavedCellBottom(bottom, minimum_fractional_cell_height=0.5)

    grid = ImmersedBoundaryGrid(underlying_grid, shaved_cell_bottom)

    for new_halo in ((3, 18, 3), (5, 5, 3))
        extended_grid = with_halo(new_halo, grid)
        rebuilt_grid = ImmersedBoundaryGrid(with_halo(new_halo, underlying_grid), shaved_cell_bottom)
        @test extended_grid.immersed_boundary == rebuilt_grid.immersed_boundary
        @test extended_grid.immersed_boundary == grid.immersed_boundary
        @test with_halo((3, 3, 3), extended_grid).immersed_boundary == grid.immersed_boundary
    end

    return nothing
end

function test_shaved_cell_time_stepping(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded),
                                      size=(32, 16), halo=(5, 5), x=(0, 1000), z=(-100, 0))

    slope(x) = -100 + 80 * clamp((x - 200) / 400, 0, 1)
    ibg = ImmersedBoundaryGrid(underlying_grid, ShavedCellBottom(slope))

    model = HydrostaticFreeSurfaceModel(ibg; tracers=(:b, :q), buoyancy=BuoyancyTracer(),
                                        momentum_advection=WENO(), tracer_advection=WENO())

    set!(model, b = (x, z) -> 1e-5 * z - 1e-4 * exp(-(x - 100)^2 / 1e4), q = 1)

    for _ in 1:20
        time_step!(model, 10)
    end

    @test !any(isnan, Array(interior(model.tracers.b)))
    @test !any(isnan, Array(interior(model.velocities.u)))

    # A uniform tracer stays uniform only if the face areas and the cell volumes agree.
    Nx, _, Nz = size(ibg)
    q = Array(interior(model.tracers.q, :, 1, :))
    wet = @allowscalar [!immersed_cell(i, 1, k, ibg) for i in 1:Nx, k in 1:Nz]
    @test maximum(abs, q[wet] .- 1) < 100eps(FT)

    return nothing
end

@testset "ShavedCellBottom" begin
    @info "Testing ShavedCellBottom..."

    for arch in archs, FT in float_types
        @info "  Testing ShavedCellBottom [$FT, $(typeof(arch))]..."
        test_shaved_cell_reduces_to_partial_cells(FT, arch)
        test_shaved_cell_land_stays_out_of_the_sea(FT, arch)
        test_shaved_cell_geometry(FT, arch)
        test_shaved_cells_are_bounded_by_their_faces(FT, arch)
        test_shaved_cell_column_depth_consistency(FT, arch)
        test_shaved_cell_flat_topologies(FT, arch)
        test_shaved_cell_with_halo(FT, arch)
        test_shaved_cell_time_stepping(FT, arch)
    end
end
