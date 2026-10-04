include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.ImmersedBoundaries: PartialCellBottomAndTop, TopLoad, immersed_cell, _immersed_cell, mask_immersed_field!, bottom_height_interior
using Oceananigans.Models: top_load_potential
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!
using Oceananigans.BuoyancyFormulations: materialize_buoyancy
using Oceananigans.Operators: Δrᶜᶜᶜ

bottom_and_top_values(op, grid, args...) =
    Array(interior(compute!(Field(KernelFunctionOperation{Center, Center, Center}(op, grid, args...)))))

underlying_immersed_cell(i, j, k, ibg) = _immersed_cell(i, j, k, ibg.underlying_grid, ibg.immersed_boundary)

wet_cells(grid) = dropdims(sum(1 .- bottom_and_top_values(immersed_cell, grid), dims=3), dims=3)

function test_partial_cell_bottom_and_top_grid_construction(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 8), extent=(1, 1, 1))

    bottom(x, y) = -1 + 0.1 * sin(2π * x) * cos(2π * y)
    top(x, y) = -0.5 + 0.4 * x

    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(bottom, top))
    ib = ibg.immersed_boundary

    @test architecture(ibg) === arch
    @test eltype(ibg) === FT
    @test size(ibg) == size(underlying_grid)
    @test halo_size(ibg) == halo_size(underlying_grid)
    @test topology(ibg) == topology(underlying_grid)
    @test eltype(ib.bottom_height) === FT
    @test eltype(ib.top_height) === FT
    @test ib.minimum_fractional_cell_height isa FT
    @test ib.minimum_cell_height isa FT

    @test summary(ibg) isa String
    @test summary(ib) isa String

    Nx, Ny = size(underlying_grid)[1:2]
    bottom_array  = on_architecture(arch, fill(FT(-0.8), Nx, Ny))
    top_array = on_architecture(arch, fill(FT(-0.2), Nx, Ny))
    @test ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(bottom_array, top_array)) isa ImmersedBoundaryGrid
    @test ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.8, -0.2; minimum_fractional_cell_height=0.1)) isa ImmersedBoundaryGrid

    return nothing
end

function test_partial_cell_bottom_and_top_immersed_cell_pattern(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 10), extent=(1, 1, 1))

    # Partial cells of height 0.05 at the bottom (k = 2) and the top (k = 8)
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.85, -0.25))

    immersed = bottom_and_top_values(immersed_cell, ibg) .== 1
    @test immersed == (bottom_and_top_values(underlying_immersed_cell, ibg) .== 1)
    @test all(immersed[:, :, 1])
    @test all(immersed[:, :, 9:10])
    @test !any(immersed[:, :, 2:8])

    Δr = bottom_and_top_values(Δrᶜᶜᶜ, ibg)
    @test all(Δr[:, :, 2] .≈ FT(0.05))
    @test all(Δr[:, :, 8] .≈ FT(0.05))
    @test all(Δr[:, :, 3:7] .≈ FT(0.1))

    return nothing
end

function test_partial_cell_bottom_and_top_minimum_thickness(FT, arch)
    Nz = 10
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, Nz), extent=(1, 1, 1))
    ϵ = 0.2

    # Fractions of 0.15, between ϵ / 2 and ϵ, are raised to ϵ
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.815, -0.285; minimum_fractional_cell_height=ϵ))
    immersed = bottom_and_top_values(immersed_cell, ibg) .== 1
    Δr = bottom_and_top_values(Δrᶜᶜᶜ, ibg)
    @test !any(immersed[:, :, 2:8])
    @test all(Δr[:, :, 2] .≈ FT(ϵ * 0.1))
    @test all(Δr[:, :, 8] .≈ FT(ϵ * 0.1))

    # Fractions of 0.05, below ϵ / 2, are removed
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.805, -0.295; minimum_fractional_cell_height=ϵ))
    immersed = bottom_and_top_values(immersed_cell, ibg) .== 1
    Δr = bottom_and_top_values(Δrᶜᶜᶜ, ibg)
    @test all(immersed[:, :, 2])
    @test all(immersed[:, :, 8])
    @test !any(immersed[:, :, 3:7])
    @test all(Δr[:, :, 3] .≈ FT(0.1))
    @test all(Δr[:, :, 7] .≈ FT(0.1))

    # A minimum_cell_height larger than ϵ Δz sets the floor
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.818, -0.282; minimum_cell_height=0.03))
    Δr = bottom_and_top_values(Δrᶜᶜᶜ, ibg)
    @test all(Δr[:, :, 2] .≈ FT(0.03))
    @test all(Δr[:, :, 8] .≈ FT(0.03))

    # A column thinner than one floored cell closes
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.501, -0.499))
    @test all(wet_cells(ibg) .== 0)

    return nothing
end

function test_partial_cell_bottom_and_top_volume(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(8, 8, 8), extent=(1, 1, 1))

    # Neither -0.7 nor -0.3 is a cell face
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.7, -0.3))

    c_ibg = CenterField(ibg)
    c_udl = CenterField(underlying_grid)
    set!(c_ibg, 1)
    set!(c_udl, 1)

    active_volume = Field(Integral(c_ibg)) |> interior |> Array |> only
    total_volume  = Field(Integral(c_udl)) |> interior |> Array |> only

    @test active_volume ≈ 2 * total_volume / 5

    return nothing
end

function test_partial_cell_bottom_and_top_reduced_field_masking(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, topology=(Periodic, Flat, Bounded), size=(3, 10), extent=(1, 1))

    bottom  = fill(FT(-0.85), 3)
    top = FT[0, -0.45, -0.95] # open, wet beneath the top, land
    ib = PartialCellBottomAndTop(on_architecture(arch, bottom), on_architecture(arch, top))
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    η = Field{Center, Center, Nothing}(ibg)
    set!(η, 1)
    mask_immersed_field!(η)

    @test Array(interior(η))[:, 1, 1] == FT[1, 1, 0]

    return nothing
end

function test_partial_cell_top_load_potential(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(5, 3, 4), x=(0, 5), y=(0, 1), z=(-1, 0),
                                      topology=(Bounded, Periodic, Bounded))

    # Open, face-aligned top, closed, open with a raised bottom, partial top
    bottom(x, y) = 3 ≤ x < 4 ? -0.6 : -1
    top(x, y) = x < 1 ? 0 :
                    x < 2 ? -0.5 :
                    x < 3 ? -0.99 :
                    x < 4 ? 0 : -0.6

    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(bottom, top))

    Φ = Array(interior(top_load_potential(ibg, BuoyancyTracer(), (; b = (x, y, z) -> z))))

    @test all(Φ[1, :, 1] .== 0)
    @test all(Φ[3, :, 1] .== 0)
    @test all(Φ[4, :, 1] .== 0)

    # -Σ b Δz over the top-covered whole cells and the top-covered part of k = 2
    @test all(Φ[2, :, 1] .≈ FT(0.125))
    @test all(Φ[5, :, 1] .≈ FT(0.1875))

    Φ² = Array(interior(top_load_potential(ibg, BuoyancyTracer(), (; b = (x, y, z) -> 2z))))
    @test all(isapprox.(Φ², 2 .* Φ; atol=100 * eps(FT)))

    T(x, y, z) = 20 + 2z
    S(x, y, z) = 35
    Φˢʷ = Array(interior(top_load_potential(ibg, SeawaterBuoyancy(FT), (; T, S))))
    @test all(Φˢʷ[[1, 3, 4], :, 1] .== 0)
    @test all(abs.(Φˢʷ[[2, 5], :, 1]) .> 0)

    # ∂p/∂z = b, so the pressure anomaly increases downward where b < 0
    b = CenterField(ibg)
    set!(b, (x, y, z) -> z)
    pHY′ = CenterField(ibg)
    update_hydrostatic_pressure!(pHY′, arch, ibg, materialize_buoyancy(BuoyancyTracer(), ibg), (; b))
    p = Array(interior(pHY′))
    @test all(diff(p[1, 1, :]) .< 0)
    @test all(diff(p[2, 1, 1:2]) .< 0)

    return nothing
end

function test_partial_cell_bottom_and_top_equality_and_show(FT, arch)
    ib = PartialCellBottomAndTop(-0.8, -0.2)
    @test ib == PartialCellBottomAndTop(-0.8, -0.2)
    @test ib != PartialCellBottomAndTop(-0.8, -0.3)
    @test ib != PartialCellBottomAndTop(-0.7, -0.2)
    @test ib != PartialCellBottomAndTop(-0.8, -0.2; minimum_fractional_cell_height=0.1)
    @test ib != PartialCellBottomAndTop(-0.8, -0.2; minimum_cell_height=0.01)
    @test isnothing(ib.top_load)
    @test ib != PartialCellBottomAndTop(-0.8, -0.2; top_load=1)

    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 4), extent=(1, 1, 1))
    ibg = ImmersedBoundaryGrid(underlying_grid, ib)

    @test sprint(show, ibg) isa String
    @test sprint(show, ibg.immersed_boundary) isa String

    return nothing
end

function test_partial_cell_bottom_and_top_model_at_rest(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, topology=(Periodic, Flat, Bounded), size=(8, 10), extent=(1, 1))

    top(x) = min(-0.6 + x, 0)
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.9, top))
    model = HydrostaticFreeSurfaceModel(ibg; buoyancy=nothing, tracers=())

    for _ in 1:3
        time_step!(model, 1e-3)
    end

    @test maximum(abs, interior(model.velocities.u)) ≤ 100 * eps(FT)
    @test maximum(abs, interior(model.velocities.w)) ≤ 100 * eps(FT)
    @test maximum(abs, interior(model.free_surface.displacement)) ≤ 100 * eps(FT)

    return nothing
end

function rest_state_max_velocity(ibg; Δt=0.01, Nt=10)
    model = HydrostaticFreeSurfaceModel(ibg; buoyancy=BuoyancyTracer(), tracers=:b)
    set!(model, b=(x, z) -> z / 2)

    for _ in 1:Nt
        time_step!(model, Δt)
    end

    max_u = maximum(abs, interior(model.velocities.u))
    max_w = maximum(abs, interior(model.velocities.w))
    max_η = maximum(abs, interior(model.free_surface.displacement))

    return max_u, max_w, max_η
end

function test_partial_cell_bottom_and_top_rest_state(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(16, 20), x=(0, 1), z=(-1, 0),
                                      topology=(Bounded, Flat, Bounded))

    # Partial cells at the bottom everywhere and along the sloping top
    top(x) = min(0, -0.93 + 1.7x)
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.98, top))

    max_u, _, _ = rest_state_max_velocity(ibg)
    @test max_u > 1e-3

    b(x, z) = z / 2
    Φ = top_load_potential(ibg, BuoyancyTracer(), (; b))
    field_loaded_ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.98, top; top_load=Φ))

    top_load = TopLoad(BuoyancyTracer(), (; b))
    loaded_ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottomAndTop(-0.98, top; top_load))
    @test Array(bottom_height_interior(loaded_ibg.immersed_boundary.top_load)) == Array(interior(Φ))
    @test loaded_ibg.immersed_boundary == field_loaded_ibg.immersed_boundary

    max_u, max_w, max_η = rest_state_max_velocity(loaded_ibg)
    tol = 5000 * eps(FT)
    @test max_u ≤ tol
    @test max_w ≤ tol
    @test max_η ≤ tol

    return nothing
end

@testset "PartialCellBottomAndTop" begin
    for arch in archs, FT in float_types
        @info "  Testing PartialCellBottomAndTop [$FT, $(typeof(arch))]..."
        @testset "Construction [$FT, $(typeof(arch))]"                   test_partial_cell_bottom_and_top_grid_construction(FT, arch)
        @testset "Immersed cell pattern [$FT, $(typeof(arch))]"          test_partial_cell_bottom_and_top_immersed_cell_pattern(FT, arch)
        @testset "Minimum fractional cell height [$FT, $(typeof(arch))]" test_partial_cell_bottom_and_top_minimum_thickness(FT, arch)
        @testset "Volume [$FT, $(typeof(arch))]"                  test_partial_cell_bottom_and_top_volume(FT, arch)
        @testset "Reduced field masking [$FT, $(typeof(arch))]"          test_partial_cell_bottom_and_top_reduced_field_masking(FT, arch)
        @testset "Top-load potential [$FT, $(typeof(arch))]"             test_partial_cell_top_load_potential(FT, arch)
        @testset "Equality and show [$FT, $(typeof(arch))]"              test_partial_cell_bottom_and_top_equality_and_show(FT, arch)
        @testset "Model at rest [$FT, $(typeof(arch))]"                  test_partial_cell_bottom_and_top_model_at_rest(FT, arch)
        @testset "Rest state beneath an immersed top [$FT, $(typeof(arch))]"     test_partial_cell_bottom_and_top_rest_state(FT, arch)
    end
end
