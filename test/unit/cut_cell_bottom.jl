include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: constructor_arguments, znodes, peripheral_node, with_halo, inactive_cell,
                          static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ
using Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶠᶜᶜ, Δrᶜᶠᶜ, Δrᶠᶠᶜ, Vᶠᶜᶜ
using Oceananigans.Advection: div_𝐯u
using Oceananigans.ImmersedBoundaries: immersed_cell
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_momentum_tendencies!

const c = Center()
const f = Face()

bumpy(ξ) = -0.9 + 0.4 * sin(3π * ξ) + 0.1 * cos(11π * ξ)
slice_grid(FT, arch; Nx=16, Nz=8) = RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded),
                                                    size=(Nx, Nz), halo=(4, 4), x=(0, 1), z=(-1, 0))

@inline column_depthᶜᶜᶜ(i, j, k, grid) = static_column_depthᶜᶜᵃ(i, j, grid)
@inline column_depthᶠᶜᶜ(i, j, k, grid) = static_column_depthᶠᶜᵃ(i, j, grid)
@inline u_peripheral(i, j, k, grid) = peripheral_node(i, j, k, grid, f, c, c)
@inline v_peripheral(i, j, k, grid) = peripheral_node(i, j, k, grid, c, f, c)
@inline closed_between_active_cells(i, j, k, grid) = u_peripheral(i, j, k, grid) & !inactive_cell(i, j, k, grid) & !inactive_cell(i-1, j, k, grid)

cpu_values(LX, LY, LZ, func, grid, args...) = Array(interior(Field(KernelFunctionOperation{LX, LY, LZ}(func, grid, args...))))

function test_cut_cell_reduces_to_partial_cells(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 8), extent=(1, 1, 1))
    Nx, Ny, Nz = size(underlying_grid)

    # A flat bottom leaves nothing to cut, whether or not it falls on an interface.
    for bottom_height in (-1/2, -1/2 - 1/2Nz)
        partial = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height))
        cut = ImmersedBoundaryGrid(underlying_grid, CutCellBottom(bottom_height))

        @allowscalar for i in 1:Nx, j in 1:Ny, k in 1:Nz
            @test Δrᶜᶜᶜ(i, j, k, cut) ≈ Δrᶜᶜᶜ(i, j, k, partial)
            @test Δrᶠᶜᶜ(i, j, k, cut) ≈ Δrᶠᶜᶜ(i, j, k, partial)
            @test Δrᶜᶠᶜ(i, j, k, cut) ≈ Δrᶜᶠᶜ(i, j, k, partial)
        end
    end

    return nothing
end

function test_cut_cell_land_stays_out_of_the_sea(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(6, 6, 8), extent=(1, 1, 1))
    Nx, Ny, Nz = size(underlying_grid)

    island_bottom_height(land_height, sea_height) = [2 ≤ i ≤ 4 && 3 ≤ j ≤ 4 ? FT(land_height) : FT(sea_height) for i in 1:Nx, j in 1:Ny]

    # An island on a flat sea floor stays dry, and the height given to the land does not reach the sea.
    for sea_height in (-1/2, -1/2 - 1/2Nz)
        partial = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(island_bottom_height(100, sea_height)))
        cut = ImmersedBoundaryGrid(underlying_grid, CutCellBottom(island_bottom_height(100, sea_height)))
        flush_cut = ImmersedBoundaryGrid(underlying_grid, CutCellBottom(island_bottom_height(0, sea_height)))

        @test cut.immersed_boundary == flush_cut.immersed_boundary

        @allowscalar for i in 1:Nx, j in 1:Ny, k in 1:Nz
            @test immersed_cell(i, j, k, cut) == immersed_cell(i, j, k, partial)
        end
    end

    return nothing
end

function test_cut_cell_geometry(FT, arch)
    Nx, Nz, Lz = 16, 8, 1
    underlying_grid = slice_grid(FT, arch; Nx, Nz)
    Δz = Lz / Nz

    slope(x) = -Lz + Lz/2 * x + Lz/8
    ibg = ImmersedBoundaryGrid(underlying_grid, CutCellBottom(slope, minimum_fractional_cell_height=0))

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

function test_cut_cells_are_bounded_by_their_faces(FT, arch)
    Nx, Nz, Lz = 16, 8, 1
    Δz = Lz / Nz
    ϵ = 0.2
    ibg = ImmersedBoundaryGrid(slice_grid(FT, arch; Nx, Nz), CutCellBottom(bumpy, minimum_fractional_cell_height=ϵ))

    opening(i, k) = peripheral_node(i, 1, k, ibg, f, c, c) ? zero(FT) : Δrᶠᶜᶜ(i, 1, k, ibg)

    @allowscalar for i in 2:Nx-1
        kᵇ = findfirst(k -> !immersed_cell(i, 1, k, ibg), 1:Nz)
        isnothing(kᵇ) && continue
        @test Δrᶜᶜᶜ(i, 1, kᵇ, ibg) ≈ (opening(i, kᵇ) + opening(i+1, kᵇ)) / 2
        @test Δrᶜᶜᶜ(i, 1, kᵇ, ibg) ≥ ϵ * Δz - 10eps(FT)
    end

    return nothing
end

function test_cut_cell_column_depth_consistency(FT, arch)
    Nx, Nz, Lz = 16, 8, 1
    Δz = Lz / Nz
    ϵ = 0.2
    ibg = ImmersedBoundaryGrid(slice_grid(FT, arch; Nx, Nz), CutCellBottom(bumpy, minimum_fractional_cell_height=ϵ))

    # The barotropic mode integrates the water column that the cut face heights describe.
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

function test_cut_cell_flat_topologies(FT, arch)
    yflat = ImmersedBoundaryGrid(slice_grid(FT, arch), CutCellBottom(bumpy))

    xflat_grid = RectilinearGrid(arch, FT, topology=(Flat, Bounded, Bounded), size=(16, 8), halo=(4, 4), y=(0, 1), z=(-1, 0))
    xflat = ImmersedBoundaryGrid(xflat_grid, CutCellBottom(bumpy))
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

function test_cut_cell_with_halo(FT, arch)
    underlying_grid = LatitudeLongitudeGrid(arch, FT, size=(32, 24, 6), halo=(3, 3, 3), longitude=(0, 360), latitude=(-60, 60), z=(-1000, 0))
    bottom(λ, φ) = -600 - 300 * sin(2π * λ / 360) * cos(2π * φ / 120)
    cut_cell_bottom = CutCellBottom(bottom, minimum_fractional_cell_height=0.5)

    grid = ImmersedBoundaryGrid(underlying_grid, cut_cell_bottom)

    for new_halo in ((3, 18, 3), (5, 5, 3))
        extended_grid = with_halo(new_halo, grid)
        rebuilt_grid = ImmersedBoundaryGrid(with_halo(new_halo, underlying_grid), cut_cell_bottom)
        @test extended_grid.immersed_boundary == rebuilt_grid.immersed_boundary
        @test extended_grid.immersed_boundary == grid.immersed_boundary
        @test with_halo((3, 3, 3), extended_grid).immersed_boundary == grid.immersed_boundary
    end

    return nothing
end

function test_cut_cell_openings_are_rounded(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(4, 4, 8), extent=(1, 1, 1))
    Δz = 1 / size(underlying_grid, 3)
    ϵ = 1/5

    # An opening thinner than ϵ Δz / 2 closes, and a thicker one is at least ϵ Δz tall.
    for (sliver, opening) in ((Δz / 100, 0), (3Δz / 20, ϵ * Δz), (3Δz / 10, 3Δz / 10))
        ibg = ImmersedBoundaryGrid(underlying_grid, CutCellBottom(-1/2 - sliver, minimum_fractional_cell_height=ϵ))
        @test all(cpu_values(Center, Center, Center, column_depthᶜᶜᶜ, ibg) .≈ 1/2 + opening)
    end

    return nothing
end

function test_cut_cell_slices_have_two_faces(FT, arch)
    ϵ = 1/5
    ramp(ξ) = -29/10 + 2ξ/5
    yflat = ImmersedBoundaryGrid(RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded), size=(7, 4), halo=(4, 4), x=(0, 7), z=(-4, 0)),
                                 CutCellBottom(ramp, minimum_fractional_cell_height=ϵ))
    xflat = ImmersedBoundaryGrid(RectilinearGrid(arch, FT, topology=(Flat, Bounded, Bounded), size=(7, 4), halo=(4, 4), y=(0, 7), z=(-4, 0)),
                                 CutCellBottom(ramp, minimum_fractional_cell_height=ϵ))
    Δz = 1

    # The phantom face of the Flat direction is not a face of the cell: the lowest cell of each column is at least
    # ϵ Δz tall, and each side of a plane lies between the height of its column and the ramp at that side.
    heights = cpu_values(Center, Center, Center, Δrᶜᶜᶜ, yflat)[:, 1, :]
    immersed = cpu_values(Center, Center, Center, immersed_cell, yflat)[:, 1, :] .== 1
    @test all(heights[i, findfirst(.!immersed[i, :])] ≥ ϵ * Δz - 10eps(FT) for i in 1:size(heights, 1))

    xᶠ = Array(xnodes(yflat, f))
    ib = on_architecture(CPU(), yflat.immersed_boundary)
    between(r, a, b) = min(a, b) - 10eps(FT) ≤ r ≤ max(a, b) + 10eps(FT)
    for i in 1:size(yflat, 1)
        rᵇ = ib.bottom_height[i, 1, 1]
        rᵂ = ib.west_bottom_height[i, 1, 1]
        @test between(rᵂ, rᵇ, ramp(xᶠ[i])) && between(2rᵇ - rᵂ, rᵇ, ramp(xᶠ[i+1]))
    end

    # and the velocity across the phantom face is masked only where its cell is.
    @test cpu_values(Center, Center, Center, v_peripheral, yflat) == cpu_values(Center, Center, Center, inactive_cell, yflat)
    @test cpu_values(Center, Center, Center, u_peripheral, xflat) == cpu_values(Center, Center, Center, inactive_cell, xflat)

    return nothing
end

function test_cut_cell_coast(FT, arch)
    grid = RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded), size=(6, 4), halo=(4, 4), x=(0, 6), z=(-4, 0))
    coast(x) = x ≤ 2 ? 1/2 : -5/2
    cut = ImmersedBoundaryGrid(grid, CutCellBottom(coast))
    partial = ImmersedBoundaryGrid(grid, PartialCellBottom(coast))

    # A face on land stays closed: the sea reaches no deeper than its floor, and the faces of the land are dry.
    @test all(cpu_values(Center, Center, Center, immersed_cell, cut)[:, :, 1] .== 1)
    @test cpu_values(Face, Center, Center, column_depthᶠᶜᶜ, cut) ≈ cpu_values(Face, Center, Center, column_depthᶠᶜᶜ, partial)

    return nothing
end

function test_cut_cell_closed_faces_carry_no_viscous_stress(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(24, 24, 10), x=(0, 1), y=(0, 1), z=(-1, 0), topology=(Periodic, Periodic, Bounded))
    seamount(x, y) = -9/10 + 7/10 * exp(-25 * ((x - 1/2)^2 + (y - 1/2)^2))
    ibg = ImmersedBoundaryGrid(underlying_grid, CutCellBottom(seamount))
    @test any(cpu_values(Face, Center, Center, closed_between_active_cells, ibg) .== 1)

    model = HydrostaticFreeSurfaceModel(ibg; closure=VerticalScalarDiffusivity(ν=1), momentum_advection=nothing,
                                        tracer_advection=nothing, buoyancy=nothing, tracers=())
    set!(model, u=1)
    update_state!(model)
    compute_momentum_tendencies!(model, [])

    # A uniform flow over free-slip boundaries feels no viscous stress, also above faces cut away by the bottom.
    open_u = cpu_values(Face, Center, Center, u_peripheral, ibg) .== 0
    Gu = Array(interior(model.timestepper.Gⁿ.u))
    @test maximum(abs, Gu[open_u]) < 100eps(FT)

    return nothing
end

function test_cut_cell_closed_faces_absorb_no_momentum(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(32, 32, 12), x=(0, 32e3), y=(0, 32e3), z=(-1200, 0), halo=(7, 7, 5), topology=(Periodic, Periodic, Bounded))
    seamount(x, y) = -1100 + 800 * exp(-((x - 16e3)^2 + (y - 16e3)^2) / 6e3^2)
    ibg = ImmersedBoundaryGrid(underlying_grid, CutCellBottom(seamount))
    @test any(cpu_values(Face, Center, Center, closed_between_active_cells, ibg) .== 1)

    model = HydrostaticFreeSurfaceModel(ibg; momentum_advection=Centered(order=2), tracer_advection=nothing, buoyancy=nothing,
                                        tracers=(), free_surface=ImplicitFreeSurface())
    set!(model, u=1/10)
    update_state!(model)

    # Over a symmetric seamount the leeward side returns what the windward side takes from a uniform flow.
    Vu = cpu_values(Face, Center, Center, Vᶠᶜᶜ, ibg)
    Au = cpu_values(Face, Center, Center, div_𝐯u, ibg, model.advection.momentum, model.velocities, model.velocities.u)
    open_u = cpu_values(Face, Center, Center, u_peripheral, ibg) .== 0
    @test abs(sum((Au .* Vu)[open_u])) < sqrt(eps(FT)) * sum(abs, (Au .* Vu)[open_u])

    return nothing
end

function test_cut_cell_minimum_height_above_the_lowest_cell(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(24, 24, 10), x=(0, 1), y=(0, 1), z=(-1, 0), topology=(Periodic, Periodic, Bounded))
    Δz = 1 / size(underlying_grid, 3)
    bottom = -9/10 .+ 8/10 * rand(Xoshiro(1234), FT, 24, 24)

    # Each open face of a cell gives it at least a quarter of ϵ Δz, in every level of the column.
    for ϵ in (17//50, 2//5, 1//2)
        ibg = ImmersedBoundaryGrid(underlying_grid, CutCellBottom(bottom, minimum_fractional_cell_height=FT(ϵ)))
        u_open = cpu_values(Face, Center, Center, u_peripheral, ibg) .== 0
        v_open = cpu_values(Center, Face, Center, v_peripheral, ibg) .== 0
        n = u_open .+ circshift(u_open, (-1, 0, 0)) .+ v_open .+ circshift(v_open, (0, -1, 0))
        heights = cpu_values(Center, Center, Center, Δrᶜᶜᶜ, ibg)
        active = cpu_values(Center, Center, Center, inactive_cell, ibg) .== 0
        @test all(heights[active] .≥ n[active] .* FT(ϵ) ./ 4 .* Δz .- 10eps(FT))
    end

    return nothing
end

function test_cut_cell_rebuilt_from_its_constructor_arguments(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(6, 6, 8), extent=(1, 1, 1))
    Nx, Ny, _ = size(underlying_grid)
    island = [2 ≤ i ≤ 4 && 3 ≤ j ≤ 4 ? FT(100) : FT(-1/2) for i in 1:Nx, j in 1:Ny]
    grid = ImmersedBoundaryGrid(underlying_grid, CutCellBottom(island))

    _, _, args = constructor_arguments(grid)
    rebuilt_bottom = CutCellBottom(args[:bottom_height], nothing, nothing, args[:corner_bottom_height], args[:minimum_fractional_cell_height], nothing)
    @test ImmersedBoundaryGrid(underlying_grid, rebuilt_bottom).immersed_boundary == grid.immersed_boundary

    return nothing
end

function test_cut_cell_time_stepping(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, topology=(Bounded, Flat, Bounded),
                                      size=(32, 16), halo=(5, 5), x=(0, 1000), z=(-100, 0))

    slope(x) = -100 + 80 * clamp((x - 200) / 400, 0, 1)
    ibg = ImmersedBoundaryGrid(underlying_grid, CutCellBottom(slope))

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

@testset "CutCellBottom" begin
    @info "Testing CutCellBottom..."

    for arch in archs, FT in float_types
        @info "  Testing CutCellBottom [$FT, $(typeof(arch))]..."
        test_cut_cell_reduces_to_partial_cells(FT, arch)
        test_cut_cell_land_stays_out_of_the_sea(FT, arch)
        test_cut_cell_geometry(FT, arch)
        test_cut_cells_are_bounded_by_their_faces(FT, arch)
        test_cut_cell_column_depth_consistency(FT, arch)
        test_cut_cell_flat_topologies(FT, arch)
        test_cut_cell_with_halo(FT, arch)
        test_cut_cell_openings_are_rounded(FT, arch)
        test_cut_cell_slices_have_two_faces(FT, arch)
        test_cut_cell_coast(FT, arch)
        test_cut_cell_closed_faces_carry_no_viscous_stress(FT, arch)
        test_cut_cell_closed_faces_absorb_no_momentum(FT, arch)
        test_cut_cell_minimum_height_above_the_lowest_cell(FT, arch)
        test_cut_cell_rebuilt_from_its_constructor_arguments(FT, arch)
        test_cut_cell_time_stepping(FT, arch)
    end
end
