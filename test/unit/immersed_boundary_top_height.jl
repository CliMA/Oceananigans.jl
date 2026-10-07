include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: znodes, static_column_depthᶜᶜᵃ
using Oceananigans.ImmersedBoundaries: GridFittedBottom, PartialCellBottom, immersed_cell, _immersed_cell,
                                       mask_immersed_field!, bottom_height_interior
using Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶜᶜᶠ

center_values(op, grid, args...) =
    Array(interior(compute!(Field(KernelFunctionOperation{Center, Center, Center}(op, grid, args...)))))

face_values(op, grid, args...) =
    Array(interior(compute!(Field(KernelFunctionOperation{Center, Center, Face}(op, grid, args...)))))

column_depthᶜᶜᵃ(i, j, k, grid) = static_column_depthᶜᶜᵃ(i, j, grid)

underlying_immersed_cell(i, j, k, ibg) = _immersed_cell(i, j, k, ibg.underlying_grid, ibg.immersed_boundary)

wet_cells(grid) = dropdims(sum(1 .- center_values(immersed_cell, grid), dims=3), dims=3)

function test_top_height_construction(FT, arch, Boundary)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 8), extent=(1, 1, 1))

    bottom(x, y) = -1 + 0.1 * sin(2π * x) * cos(2π * y)
    top(x, y) = -0.5 + 0.4 * x

    ibg = ImmersedBoundaryGrid(underlying_grid, Boundary(bottom; top_height=top))
    ib = ibg.immersed_boundary

    @test architecture(ibg) === arch
    @test eltype(ibg) === FT
    @test size(ibg) == size(underlying_grid)
    @test eltype(ib.bottom_height) === FT
    @test eltype(ib.top_height) === FT
    @test all(Array(bottom_height_interior(ib.top_height)) .> Array(bottom_height_interior(ib.bottom_height)))

    @test isnothing(Boundary(bottom).top_height)
    @test isnothing(ImmersedBoundaryGrid(underlying_grid, Boundary(bottom)).immersed_boundary.top_height)

    Nx, Ny = size(underlying_grid)[1:2]
    bottom_array = on_architecture(arch, fill(FT(-0.8), Nx, Ny))
    top_array    = on_architecture(arch, fill(FT(-0.2), Nx, Ny))
    @test ImmersedBoundaryGrid(underlying_grid, Boundary(bottom_array; top_height=top_array)) isa ImmersedBoundaryGrid
    @test ImmersedBoundaryGrid(underlying_grid, Boundary(-0.8; top_height=-0.2)) isa ImmersedBoundaryGrid

    return nothing
end

function test_grid_fitted_top_immersed_cell_pattern(FT, arch)
    Nz = 10
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, Nz), extent=(1, 1, 1))

    # The bottom snaps to -0.8 and the top to -0.3
    ibg = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-0.83; top_height=-0.27))
    ib = ibg.immersed_boundary

    @test all(Array(bottom_height_interior(ib.bottom_height)) .≈ FT(-0.8))
    @test all(Array(bottom_height_interior(ib.top_height)) .≈ FT(-0.3))

    immersed = center_values(immersed_cell, ibg) .== 1
    @test immersed == (center_values(underlying_immersed_cell, ibg) .== 1)
    @test all(immersed[:, :, 1:2])
    @test all(immersed[:, :, 8:10])
    @test !any(immersed[:, :, 3:7])

    @test all(center_values(column_depthᶜᶜᵃ, ibg) .≈ FT(0.5))

    return nothing
end

function test_grid_fitted_top_column_closure(FT, arch)
    Nx, Nz = 5, 10
    underlying_grid = RectilinearGrid(arch, FT, topology=(Periodic, Flat, Bounded), size=(Nx, Nz), extent=(1, 1))

    bottom = fill(FT(-0.53), Nx)                 # snaps to -0.5
    top    = FT[0, -0.27, -0.37, -0.57, -0.97]   # open, 2 wet cells, 1 wet cell, below bottom, below domain

    ib = GridFittedBottom(on_architecture(arch, bottom); top_height=on_architecture(arch, top))
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    @test wet_cells(ibg)[:, 1] == [5, 2, 1, 0, 0]
    @test center_values(column_depthᶜᶜᵃ, ibg)[:, 1, 1] ≈ FT[0.5, 0.2, 0.1, 0, 0]

    return nothing
end

function test_partial_cell_top_immersed_cell_pattern(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 10), extent=(1, 1, 1))

    # Partial cells of height 0.05 at the bottom (k = 2) and the top (k = 8)
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(-0.85; top_height=-0.25))

    immersed = center_values(immersed_cell, ibg) .== 1
    @test immersed == (center_values(underlying_immersed_cell, ibg) .== 1)
    @test all(immersed[:, :, 1])
    @test all(immersed[:, :, 9:10])
    @test !any(immersed[:, :, 2:8])

    Δr = center_values(Δrᶜᶜᶜ, ibg)
    @test all(Δr[:, :, 2] .≈ FT(0.05))
    @test all(Δr[:, :, 8] .≈ FT(0.05))
    @test all(Δr[:, :, 3:7] .≈ FT(0.1))

    # Face spacings adjacent to the partial cells: half the partial cell plus half the full neighbor
    Δrᶠ = face_values(Δrᶜᶜᶠ, ibg)
    @test all(Δrᶠ[:, :, 3] .≈ FT(0.075))
    @test all(Δrᶠ[:, :, 8] .≈ FT(0.075))
    @test all(Δrᶠ[:, :, 4:7] .≈ FT(0.1))

    @test all(center_values(column_depthᶜᶜᵃ, ibg) .≈ FT(0.6))

    return nothing
end

function test_partial_cell_top_minimum_fractional_cell_height(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 10), extent=(1, 1, 1))
    ϵ = 0.2

    # Fractions of 0.15 and 0.05 below ϵ are both raised to ϵ; no cell is removed
    for (zb, zt) in ((-0.815, -0.285), (-0.805, -0.295))
        ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(zb; top_height=zt, minimum_fractional_cell_height=ϵ))
        immersed = center_values(immersed_cell, ibg) .== 1
        Δr = center_values(Δrᶜᶜᶜ, ibg)
        @test all(immersed[:, :, 1])
        @test all(immersed[:, :, 9:10])
        @test !any(immersed[:, :, 2:8])
        @test all(Δr[:, :, 2] .≈ FT(ϵ * 0.1))
        @test all(Δr[:, :, 8] .≈ FT(ϵ * 0.1))
    end

    return nothing
end

function test_partial_cell_top_column_closure(FT, arch)
    Nx, Nz = 4, 10
    underlying_grid = RectilinearGrid(arch, FT, topology=(Periodic, Flat, Bounded), size=(Nx, Nz), extent=(1, 1))

    # open, top and bottom in the same cell, thin column raised to two ϵ cells, top below bottom
    bottom = FT[-0.85, -0.55, -0.501, -0.5]
    top    = FT[0, -0.52, -0.499, -0.6]

    ib = PartialCellBottom(on_architecture(arch, bottom); top_height=on_architecture(arch, top))
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    @test wet_cells(ibg)[:, 1] == [9, 1, 2, 0]

    Δr = center_values(Δrᶜᶜᶜ, ibg)
    @test Δr[2, 1, 5] ≈ FT(0.03)
    @test Δr[3, 1, 5] ≈ FT(0.02)
    @test Δr[3, 1, 6] ≈ FT(0.02)

    @test center_values(column_depthᶜᶜᵃ, ibg)[:, 1, 1] ≈ FT[0.85, 0.03, 0.04, 0]

    return nothing
end

function test_top_height_volume(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(8, 8, 8), extent=(1, 1, 1))

    # Neither -0.7 nor -0.3 is a cell face
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(-0.7; top_height=-0.3))

    c_ibg = CenterField(ibg)
    c_udl = CenterField(underlying_grid)
    set!(c_ibg, 1)
    set!(c_udl, 1)

    active_volume = Field(Integral(c_ibg)) |> interior |> Array |> only
    total_volume  = Field(Integral(c_udl)) |> interior |> Array |> only

    @test active_volume ≈ 2 * total_volume / 5

    return nothing
end

function test_open_top_matches_bottom_only(FT, arch, Boundary)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 10), extent=(1, 1, 1))

    bottom(x, y) = -0.83 + 0.1 * x
    bottom_only = ImmersedBoundaryGrid(underlying_grid, Boundary(bottom))
    open_top    = ImmersedBoundaryGrid(underlying_grid, Boundary(bottom; top_height=0))

    @test center_values(immersed_cell, open_top) == center_values(immersed_cell, bottom_only)
    @test center_values(Δrᶜᶜᶜ, open_top) == center_values(Δrᶜᶜᶜ, bottom_only)
    @test face_values(Δrᶜᶜᶠ, open_top) == face_values(Δrᶜᶜᶠ, bottom_only)
    @test center_values(column_depthᶜᶜᵃ, open_top) == center_values(column_depthᶜᶜᵃ, bottom_only)

    return nothing
end

function test_top_height_reduced_field_masking(FT, arch, Boundary)
    underlying_grid = RectilinearGrid(arch, FT, topology=(Periodic, Flat, Bounded), size=(3, 10), extent=(1, 1))

    bottom = fill(FT(-0.85), 3)
    top    = FT[0, -0.45, -0.95] # open, wet beneath the top, land
    ib = Boundary(on_architecture(arch, bottom); top_height=on_architecture(arch, top))
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    η = Field{Center, Center, Nothing}(ibg)
    set!(η, 1)
    mask_immersed_field!(η)

    @test Array(interior(η))[:, 1, 1] == FT[1, 1, 0]

    return nothing
end

function test_top_height_equality_and_show(FT, arch, Boundary)
    ib = Boundary(-0.8; top_height=-0.2)
    @test ib == Boundary(-0.8; top_height=-0.2)
    @test ib != Boundary(-0.8; top_height=-0.3)
    @test ib != Boundary(-0.7; top_height=-0.2)
    @test ib != Boundary(-0.8)

    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 4), extent=(1, 1, 1))
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)
    @test ibg.immersed_boundary == ImmersedBoundaryGrid(underlying_grid, Boundary(-0.8; top_height=-0.2)).immersed_boundary
    @test Oceananigans.Grids.with_halo((4, 4, 4), ibg).immersed_boundary == ibg.immersed_boundary

    @test occursin("zt", summary(ibg.immersed_boundary))
    @test occursin("top_height", sprint(show, ibg.immersed_boundary))
    @test !occursin("zt", summary(ImmersedBoundaryGrid(underlying_grid, Boundary(-0.8)).immersed_boundary))
    @test sprint(show, ibg) isa String

    return nothing
end

function test_no_bottom_height(FT, arch, Boundary)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 10), extent=(1, 1, 1))

    top(x, y) = -0.27 - 0.4 * x
    no_bottom   = ImmersedBoundaryGrid(underlying_grid, Boundary(; top_height=top))
    flat_bottom = ImmersedBoundaryGrid(underlying_grid, Boundary(-1; top_height=top))

    @test Array(bottom_height_interior(no_bottom.immersed_boundary.bottom_height)) ==
          Array(bottom_height_interior(flat_bottom.immersed_boundary.bottom_height))
    @test center_values(immersed_cell, no_bottom) == center_values(immersed_cell, flat_bottom)
    @test center_values(Δrᶜᶜᶜ, no_bottom) == center_values(Δrᶜᶜᶜ, flat_bottom)
    @test center_values(column_depthᶜᶜᵃ, no_bottom) == center_values(column_depthᶜᶜᵃ, flat_bottom)

    no_boundary = ImmersedBoundaryGrid(underlying_grid, Boundary())
    @test !any(center_values(immersed_cell, no_boundary) .== 1)
    @test occursin("nothing", summary(Boundary(; top_height=top)))

    return nothing
end

@testset "Immersed boundary top height" begin
    for arch in archs, FT in float_types
        A = typeof(arch)
        for Boundary in (GridFittedBottom, PartialCellBottom)
            @testset "$Boundary construction [$FT, $A]"          test_top_height_construction(FT, arch, Boundary)
            @testset "$Boundary open top [$FT, $A]"              test_open_top_matches_bottom_only(FT, arch, Boundary)
            @testset "$Boundary reduced field masking [$FT, $A]" test_top_height_reduced_field_masking(FT, arch, Boundary)
            @testset "$Boundary equality and show [$FT, $A]"     test_top_height_equality_and_show(FT, arch, Boundary)
            @testset "$Boundary without bottom [$FT, $A]"        test_no_bottom_height(FT, arch, Boundary)
        end
        @testset "GridFittedBottom top pattern [$FT, $A]"                     test_grid_fitted_top_immersed_cell_pattern(FT, arch)
        @testset "GridFittedBottom column closure [$FT, $A]"                  test_grid_fitted_top_column_closure(FT, arch)
        @testset "PartialCellBottom top pattern [$FT, $A]"                    test_partial_cell_top_immersed_cell_pattern(FT, arch)
        @testset "PartialCellBottom minimum fractional cell height [$FT, $A]" test_partial_cell_top_minimum_fractional_cell_height(FT, arch)
        @testset "PartialCellBottom column closure [$FT, $A]"                 test_partial_cell_top_column_closure(FT, arch)
        @testset "PartialCellBottom volume [$FT, $A]"                         test_top_height_volume(FT, arch)
    end
end
