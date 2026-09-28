include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: znodes, static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ, static_column_depthᶠᶠᵃ
using Oceananigans.ImmersedBoundaries: GridFittedBottom, GridFittedCavity, immersed_cell, bottom_height_interior, mask_immersed_field!

cavity_values(op, grid, args...) =
    Array(interior(compute!(Field(KernelFunctionOperation{Center, Center, Center}(op, grid, args...)))))

column_depthᶜᶜᵃ(i, j, k, grid) = static_column_depthᶜᶜᵃ(i, j, grid)
column_depthᶠᶜᵃ(i, j, k, grid) = static_column_depthᶠᶜᵃ(i, j, grid)
column_depthᶜᶠᵃ(i, j, k, grid) = static_column_depthᶜᶠᵃ(i, j, grid)
column_depthᶠᶠᵃ(i, j, k, grid) = static_column_depthᶠᶠᵃ(i, j, grid)

wet_cells(grid) = dropdims(sum(1 .- cavity_values(immersed_cell, grid), dims=3), dims=3)

function test_cavity_grid_construction(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 8), extent=(1, 1, 1))

    bottom(x, y) = -1 + 0.1 * sin(2π * x) * cos(2π * y)
    ceiling(x, y) = -0.5 + 0.4 * x

    ibg = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(bottom, ceiling))
    ib = ibg.immersed_boundary

    @test architecture(ibg) === arch
    @test eltype(ibg) === FT
    @test size(ibg) == size(underlying_grid)
    @test halo_size(ibg) == halo_size(underlying_grid)
    @test topology(ibg) == topology(underlying_grid)
    @test eltype(ib.bottom_height) === FT
    @test eltype(ib.ceiling_height) === FT

    # Both heights are snapped to cell faces
    zfaces = znodes(ibg, Face())
    @test all(in.(Array(bottom_height_interior(ib.bottom_height)), Ref(zfaces)))
    @test all(in.(Array(bottom_height_interior(ib.ceiling_height)), Ref(zfaces)))

    @test summary(ibg) isa String
    @test summary(ib) isa String

    Nx, Ny = size(underlying_grid)[1:2]
    bottom_array  = on_architecture(arch, fill(FT(-0.8), Nx, Ny))
    ceiling_array = on_architecture(arch, fill(FT(-0.2), Nx, Ny))
    @test ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(bottom_array, ceiling_array)) isa ImmersedBoundaryGrid
    @test ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-0.8, -0.2)) isa ImmersedBoundaryGrid

    return nothing
end

function test_cavity_immersed_cell_pattern(FT, arch)
    Nz = 10
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, Nz), extent=(1, 1, 1))

    zb, zd = -0.85, -0.25
    ibg = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(zb, zd))

    zᶜ = reshape(Array(znodes(ibg, Center())), 1, 1, Nz)
    expected = (zᶜ .≤ zb) .| (zᶜ .≥ zd)
    @test cavity_values(immersed_cell, ibg) == repeat(expected, 4, 4, 1)

    # A ceiling at the top of the grid is a GridFittedBottom
    ibg_cavity = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(zb, 0))
    ibg_bottom = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(zb))
    @test cavity_values(immersed_cell, ibg_cavity) == cavity_values(immersed_cell, ibg_bottom)
    @test cavity_values(column_depthᶜᶜᵃ, ibg_cavity) == cavity_values(column_depthᶜᶜᵃ, ibg_bottom)

    return nothing
end

function test_cavity_column_depth(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 10), extent=(1, 1, 1))

    # The bottom snaps to -0.8 and the ceiling to -0.3
    ibg = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-0.85, -0.25))

    for depth in (column_depthᶜᶜᵃ, column_depthᶠᶜᵃ, column_depthᶜᶠᵃ, column_depthᶠᶠᵃ)
        @test all(cavity_values(depth, ibg) .≈ FT(0.5))
    end

    return nothing
end

function test_cavity_minimum_thickness(FT, arch)
    Nx, Nz = 5, 10
    underlying_grid = RectilinearGrid(arch, FT, topology=(Periodic, Flat, Bounded), size=(Nx, Nz), extent=(1, 1))

    bottom  = fill(FT(-0.55), Nx)                # snaps to -0.5
    ceiling = FT[0, -0.25, -0.35, -0.55, -0.95]  # open, 2 wet cells, 1 wet cell, below bottom, below domain

    ib = GridFittedCavity(on_architecture(arch, bottom), on_architecture(arch, ceiling))
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    @test wet_cells(ibg)[:, 1] == [5, 2, 0, 0, 0]
    @test cavity_values(column_depthᶜᶜᵃ, ibg)[:, 1, 1] ≈ FT[0.5, 0.2, 0, 0, 0]

    # Without a ceiling a single wet cell survives, as for GridFittedBottom
    shallow = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(on_architecture(arch, fill(FT(-0.15), Nx)),
                                                                     on_architecture(arch, zeros(FT, Nx))))
    @test all(wet_cells(shallow) .== 1)

    return nothing
end

function test_cavity_volume(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(8, 8, 8), extent=(1, 1, 1))
    ibg = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-0.75, -0.25))

    c_ibg = CenterField(ibg)
    c_udl = CenterField(underlying_grid)
    set!(c_ibg, 1)
    set!(c_udl, 1)

    active_volume = Field(Integral(c_ibg)) |> interior |> Array |> only
    total_volume  = Field(Integral(c_udl)) |> interior |> Array |> only

    @test active_volume ≈ total_volume / 2

    return nothing
end

function test_cavity_reduced_field_masking(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, topology=(Periodic, Flat, Bounded), size=(3, 10), extent=(1, 1))

    bottom  = fill(FT(-0.85), 3)
    ceiling = FT[0, -0.45, -0.95] # open ocean, wet cavity, land
    ib = GridFittedCavity(on_architecture(arch, bottom), on_architecture(arch, ceiling))
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    η = Field{Center, Center, Nothing}(ibg)
    set!(η, 1)
    mask_immersed_field!(η)

    @test Array(interior(η))[:, 1, 1] == FT[1, 1, 0]

    return nothing
end

function test_cavity_equality_and_show(FT, arch)
    @test GridFittedCavity(-0.8, -0.2) == GridFittedCavity(-0.8, -0.2)
    @test GridFittedCavity(-0.8, -0.2) != GridFittedCavity(-0.8, -0.3)
    @test GridFittedCavity(-0.8, -0.2) != GridFittedCavity(-0.7, -0.2)

    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 4), extent=(1, 1, 1))
    ibg = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-0.8, -0.2))
    @test ibg.immersed_boundary == ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-0.8, -0.2)).immersed_boundary

    @test sprint(show, ibg) isa String
    @test sprint(show, ibg.immersed_boundary) isa String

    @test isnothing(GridFittedCavity(-0.8, -0.2).ice_load)
    @test GridFittedCavity(-0.8, -0.2) != GridFittedCavity(-0.8, -0.2; ice_load=1)

    ice_load(x, y) = x
    loaded = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-0.8, -0.2; ice_load))
    @test Array(bottom_height_interior(loaded.immersed_boundary.ice_load))[:, 1, 1] ≈ FT[0.125, 0.375, 0.625, 0.875]
    @test loaded.immersed_boundary != ibg.immersed_boundary
    @test Oceananigans.Grids.with_halo((4, 4, 4), loaded).immersed_boundary == loaded.immersed_boundary
    @test sprint(show, loaded.immersed_boundary) isa String

    return nothing
end

function test_cavity_model_at_rest(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, topology=(Periodic, Flat, Bounded), size=(8, 10), extent=(1, 1))

    ceiling(x) = min(-0.6 + x, 0)
    ibg = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-0.9, ceiling))
    model = HydrostaticFreeSurfaceModel(ibg; buoyancy=nothing, tracers=())

    for _ in 1:3
        time_step!(model, 1e-3)
    end

    @test maximum(abs, interior(model.velocities.u)) ≤ 100 * eps(FT)
    @test maximum(abs, interior(model.velocities.w)) ≤ 100 * eps(FT)
    @test maximum(abs, interior(model.free_surface.displacement)) ≤ 100 * eps(FT)

    return nothing
end

@testset "GridFittedCavity" begin
    for arch in archs, FT in float_types
        @info "  Testing GridFittedCavity [$FT, $(typeof(arch))]..."
        @testset "Construction [$FT, $(typeof(arch))]"             test_cavity_grid_construction(FT, arch)
        @testset "Immersed cell pattern [$FT, $(typeof(arch))]"    test_cavity_immersed_cell_pattern(FT, arch)
        @testset "Column depth [$FT, $(typeof(arch))]"             test_cavity_column_depth(FT, arch)
        @testset "Minimum cavity thickness [$FT, $(typeof(arch))]" test_cavity_minimum_thickness(FT, arch)
        @testset "Cavity volume [$FT, $(typeof(arch))]"            test_cavity_volume(FT, arch)
        @testset "Reduced field masking [$FT, $(typeof(arch))]"    test_cavity_reduced_field_masking(FT, arch)
        @testset "Equality and show [$FT, $(typeof(arch))]"        test_cavity_equality_and_show(FT, arch)
        @testset "Model at rest [$FT, $(typeof(arch))]"            test_cavity_model_at_rest(FT, arch)
    end
end
