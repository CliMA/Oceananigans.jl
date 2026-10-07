include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: div_Uc, materialize_advection
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: TopLoad, bottom_height_interior, immersed_cell, mask_immersed_field!
using Oceananigans.Models: top_load_potential
using Oceananigans.Grids: znode
using Oceananigans.Models.HydrostaticFreeSurfaceModels: immersed_top_advective_form_correctionᶜᶜᶜ, compute_w_from_continuity!,
                                                        update_zstar_scaling!
using Oceananigans.TurbulenceClosures: z_top, z_bottom, depthᶜᶜᶠ, height_above_bottomᶜᶜᶠ, wall_vertical_distanceᶜᶜᶠ

immersed_top_values(op, grid, args...) =
    Array(interior(compute!(Field(KernelFunctionOperation{Center, Center, Center}(op, grid, args...)))))

z_topᶜᶜᵃ(i, j, k, grid) = z_top(i, j, grid)
z_bottomᶜᶜᵃ(i, j, k, grid) = z_bottom(i, j, grid)

function test_immersed_top_wall_distances(FT, arch)
    # The bottom snaps to -0.8 and the top to -0.3
    underlying_grid = RectilinearGrid(arch, FT, size=(2, 2, 10), extent=(1, 1, 1))

    top(x, y) = x < 0.5 ? -0.25 : 0
    ibg = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-0.85; top_height=top))

    @test all(immersed_top_values(z_topᶜᶜᵃ, ibg)[1, :, 1] .≈ FT(-0.3))
    @test all(immersed_top_values(z_topᶜᶜᵃ, ibg)[2, :, 1] .≈ 0)
    @test all(immersed_top_values(z_bottomᶜᶜᵃ, ibg)[:, :, 1] .≈ FT(-0.8))

    # Face k = 6 is at z = -0.5
    depth = immersed_top_values(depthᶜᶜᶠ, ibg)[:, 1, 6]
    height = immersed_top_values(height_above_bottomᶜᶜᶠ, ibg)[:, 1, 6]
    distance = immersed_top_values(wall_vertical_distanceᶜᶜᶠ, ibg)[:, 1, 6]

    @test depth ≈ FT[0.2, 0.5]
    @test height ≈ FT[0.3, 0.3]
    @test distance ≈ FT[0.2, 0.3]

    return nothing
end

function test_immersed_top_catke(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(8, 10), x=(0, 1), z=(-1, 0),
                                      topology=(Periodic, Flat, Bounded))

    top(x) = min(0, -0.6 + x)
    ibg = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-1; top_height=top))

    model = HydrostaticFreeSurfaceModel(ibg; buoyancy=BuoyancyTracer(), tracers=:b,
                                        closure=CATKEVerticalDiffusivity(FT))
    set!(model, b=(x, z) -> z / 2)

    for _ in 1:3
        time_step!(model, 1e-3)
    end

    @test all(isfinite, Array(interior(model.velocities.u)))
    @test all(isfinite, Array(interior(model.tracers.b)))
    @test all(isfinite, Array(interior(model.tracers.e)))

    return nothing
end

@inline flux_form_tendencyᶜᶜᶜ(i, j, k, grid, advection, U, c) = - div_Uc(i, j, k, grid, advection, U, c)

@inline corrected_tendencyᶜᶜᶜ(i, j, k, grid, advection, U, c) =
    - div_Uc(i, j, k, grid, advection, U, c) + immersed_top_advective_form_correctionᶜᶜᶜ(i, j, k, grid, advection, U, c)

function test_immersed_top_uniform_tracer(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(8, 16, 12), x=(0, 1), y=(0, 2), z=(-1, 0),
                                      topology=(Periodic, Bounded, Bounded))

    top(x, y) = y < 1.5 ? -0.8 + 0.5y : 1
    ibg = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(-1; top_height=top))

    u = XFaceField(ibg)
    v = YFaceField(ibg)
    w = ZFaceField(ibg)
    set!(u, (x, y, z) -> sin(2π * x) * cos(π * y) * (1 + z))
    set!(v, (x, y, z) -> cos(2π * x) * sin(π * y) / 2)
    mask_immersed_field!(u)
    mask_immersed_field!(v)
    fill_halo_regions!((u, v))

    U = (; u, v, w)
    compute_w_from_continuity!(U, ibg)
    fill_halo_regions!(w)

    c = CenterField(ibg)
    set!(c, 1)
    fill_halo_regions!(c)

    for scheme in (Centered(), WENO())
        advection = materialize_advection(scheme, ibg)
        flux_form = immersed_top_values(flux_form_tendencyᶜᶜᶜ, ibg, advection, U, c)
        corrected = immersed_top_values(corrected_tendencyᶜᶜᶜ, ibg, advection, U, c)

        # Without the correction a uniform tracer is not preserved beneath the immersed top
        @test maximum(abs, flux_form) > 1e-2
        @test maximum(abs, corrected) ≤ 100 * eps(FT)
    end

    return nothing
end

function test_immersed_top_zstar_conservation(FT, arch)
    Lx = 20kilometers
    z = MutableVerticalDiscretization(collect(-5:0))
    underlying_grid = RectilinearGrid(arch, FT; size=(8, 4, 5), x=(0, Lx), y=(0, 2kilometers), z,
                                      topology=(Bounded, Periodic, Bounded))

    top(x, y) = x < Lx / 2 ? -1.5 - 2x / Lx : 0
    bᵢ(x, y, z) = x < Lx / 4 ? 0.06 : 0.01

    for Bottom in (GridFittedBottom, PartialCellBottom)
        ibg = ImmersedBoundaryGrid(underlying_grid, Bottom(-4.5; top_height=top))

        for (free_surface, Δt) in ((ExplicitFreeSurface(), 10), (SplitExplicitFreeSurface(ibg; substeps=8), 2minutes))
            model = HydrostaticFreeSurfaceModel(ibg; free_surface,
                                                tracers = (:b, :c, :constant),
                                                timestepper = :SplitRungeKutta3,
                                                buoyancy = BuoyancyTracer(),
                                                vertical_coordinate = ZStarCoordinate())

            set!(model, b=bᵢ, c=(x, y, z) -> x / Lx, constant=1)

            Bᵢ = Array(interior(compute!(Field(Integral(model.tracers.b)))))[1]
            Cᵢ = Array(interior(compute!(Field(Integral(model.tracers.c)))))[1]

            for _ in 1:20
                time_step!(model, Δt)
            end

            η = Array(interior(model.free_surface.displacement))
            Bₙ = Array(interior(compute!(Field(Integral(model.tracers.b)))))[1]
            Cₙ = Array(interior(compute!(Field(Integral(model.tracers.c)))))[1]
            constant = Array(interior(model.tracers.constant))
            active = immersed_top_values(immersed_cell, ibg) .== 0

            @test maximum(abs, η) > 0
            @test Bₙ ≈ Bᵢ
            @test Cₙ ≈ Cᵢ
            @test all(constant[active] .≈ 1)
        end
    end

    return nothing
end

function top_wet_cell_values(c, grid)
    wet = immersed_top_values(immersed_cell, grid) .== 0
    values = Array(interior(c))
    Nx, Ny, _ = size(grid)
    return [values[i, j, findlast(wet[i, j, :])] for i in 1:Nx, j in 1:Ny]
end

integrated(c) = Array(interior(compute!(Field(Integral(c)))))[1]

function test_immersed_top_zstar_freshwater_flux(FT, arch)
    Lx, Ly = 20kilometers, 2kilometers
    z = MutableVerticalDiscretization(collect(-5:0))
    underlying_grid = RectilinearGrid(arch, FT; size=(8, 4, 5), x=(0, Lx), y=(0, Ly), z,
                                      topology=(Bounded, Periodic, Bounded))
    Nx, Ny, _ = size(underlying_grid)
    Az = Lx / Nx * Ly / Ny

    F₀ = convert(FT, 1e-4)
    Fη(x, y, z, t) = ifelse(x < Lx / 2, F₀, zero(F₀))
    top(x, y) = x < Lx / 2 ? -1.5 - 2x / Lx : 0

    for Bottom in (GridFittedBottom, PartialCellBottom)
        ibg = ImmersedBoundaryGrid(underlying_grid, Bottom(-4.5; top_height=top))
        free_surfaces = ((ExplicitFreeSurface(), 10),
                         (SplitExplicitFreeSurface(ibg; substeps=8), 2minutes),
                         (ImplicitFreeSurface(), 2minutes))

        for (free_surface, Δt) in free_surfaces
            model = HydrostaticFreeSurfaceModel(ibg; free_surface,
                                                forcing = (; η = Forcing(Fη)),
                                                tracers = (:c, :constant),
                                                timestepper = :SplitRungeKutta3,
                                                vertical_coordinate = ZStarCoordinate())

            set!(model, c=(x, y, z) -> 1 + z / 5, constant=1)

            volume = CenterField(ibg)
            set!(volume, 1)
            Vᵢ = integrated(volume)
            Cᵢ = integrated(model.tracers.c)
            cᵢ = top_wet_cell_values(model.tracers.c, ibg)

            Nt = 20
            for _ in 1:Nt
                time_step!(model, Δt)
            end

            t = Nt * Δt
            F = [Fη(x, 0, 0, 0) for x in Array(xnodes(ibg, Center())), _ in 1:Ny]
            Vₙ = integrated(volume)
            Cₙ = integrated(model.tracers.c)
            cₙ = top_wet_cell_values(model.tracers.c, ibg)
            constant = Array(interior(model.tracers.constant))
            active = immersed_top_values(immersed_cell, ibg) .== 0

            # The added water carries the concentration of the topmost wet cell, which drifts between cᵢ and cₙ
            carriedᵢ = sum(F .* cᵢ) * Az * t
            carriedₙ = sum(F .* cₙ) * Az * t

            @test Vₙ - Vᵢ ≈ sum(F) * Az * t
            @test all(constant[active] .≈ 1)
            @test min(carriedᵢ, carriedₙ) ≤ Cₙ - Cᵢ ≤ max(carriedᵢ, carriedₙ)
        end
    end

    return nothing
end

function test_immersed_top_zstar_freshwater_flux_rest_state(FT, arch)
    z = MutableVerticalDiscretization(collect(-5:0))
    underlying_grid = RectilinearGrid(arch, FT; size=(8, 4, 5), x=(0, 20kilometers), y=(0, 2kilometers), z,
                                      topology=(Bounded, Periodic, Bounded))
    F₀ = convert(FT, 1e-4)

    for Bottom in (GridFittedBottom, PartialCellBottom)
        ibg = ImmersedBoundaryGrid(underlying_grid, Bottom(-4.5; top_height=-1.5))
        model = HydrostaticFreeSurfaceModel(ibg; free_surface = SplitExplicitFreeSurface(ibg; substeps=8),
                                            forcing = (; η = Forcing((x, y, z, t) -> F₀)),
                                            timestepper = :SplitRungeKutta3,
                                            vertical_coordinate = ZStarCoordinate())

        for _ in 1:20
            time_step!(model, 2minutes)
        end

        η = Array(interior(model.free_surface.displacement))
        @test all(η .≈ F₀ * 40minutes)
        @test maximum(abs, interior(model.velocities.u)) ≤ 5000 * eps(FT)
        @test maximum(abs, interior(model.velocities.v)) ≤ 5000 * eps(FT)
    end

    return nothing
end

function test_immersed_top_zstar_znode(FT, arch)
    z = MutableVerticalDiscretization(collect(range(-1, 0, length=11)))
    underlying_grid = RectilinearGrid(arch, FT; size=(4, 10), x=(0, 1), z, topology=(Bounded, Flat, Bounded))
    Nx, _, Nz = size(underlying_grid)
    top(x) = x < 1/2 ? -1/2 : 0
    η₀ = FT(0.01)

    for Bottom in (GridFittedBottom, PartialCellBottom)
        ibg = ImmersedBoundaryGrid(underlying_grid, Bottom(-1; top_height=top))
        model = HydrostaticFreeSurfaceModel(ibg; vertical_coordinate=ZStarCoordinate())
        set!(model.free_surface.displacement, η₀)
        update_zstar_scaling!(ibg, model.free_surface.displacement)

        zᶠ = Array(interior(compute!(Field(KernelFunctionOperation{Center, Center, Face}(znode, ibg, Center(), Center(), Face())))))
        covered_columns = 1:Nx÷2
        open_columns = Nx÷2+1:Nx
        @test all(zᶠ[:, 1, 1] .≈ -1)
        @test all(zᶠ[covered_columns, 1, Nz÷2+1] .≈ -FT(1/2) + η₀)
        @test all(zᶠ[open_columns, 1, Nz+1] .≈ η₀)
    end
    return nothing
end

function test_immersed_top_load_potential(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(5, 3, 4), x=(0, 5), y=(0, 1), z=(-1, 0),
                                      topology=(Bounded, Periodic, Bounded))

    # Open, face-aligned top, closed, open with a raised bottom, partial top
    bottom(x, y) = 3 ≤ x < 4 ? -0.6 : -1
    top(x, y) = x < 1 ? 0 :
                    x < 2 ? -0.5 :
                    x < 3 ? -1 :
                    x < 4 ? 0 : -0.6

    # The partial top of column 5 is snapped to the face at -0.5 by GridFittedBottom
    for (Bottom, Φ⁵) in ((GridFittedBottom, FT(0.125)), (PartialCellBottom, FT(0.1875)))
        ibg = ImmersedBoundaryGrid(underlying_grid, Bottom(bottom; top_height=top))

        Φ = Array(interior(top_load_potential(ibg, BuoyancyTracer(), (; b = (x, y, z) -> z))))

        @test all(Φ[1, :, 1] .== 0)
        @test all(Φ[3, :, 1] .== 0)
        @test all(Φ[4, :, 1] .== 0)
        @test all(Φ[2, :, 1] .≈ FT(0.125))
        @test all(Φ[5, :, 1] .≈ Φ⁵)

        Φ² = Array(interior(top_load_potential(ibg, BuoyancyTracer(), (; b = (x, y, z) -> 2z))))
        @test all(isapprox.(Φ², 2 .* Φ; atol=100 * eps(FT)))

        T(x, y, z) = 20 + 2z
        S(x, y, z) = 35
        Φˢʷ = Array(interior(top_load_potential(ibg, SeawaterBuoyancy(FT), (; T, S))))
        @test all(Φˢʷ[[1, 3, 4], :, 1] .== 0)
        @test all(abs.(Φˢʷ[[2, 5], :, 1]) .> 0)
    end

    return nothing
end

function rest_state_model(ibg; Δt=0.01, Nt=10, kw...)
    model = HydrostaticFreeSurfaceModel(ibg; buoyancy=BuoyancyTracer(), tracers=:b, kw...)
    set!(model, b=(x, z) -> z / 2)

    for _ in 1:Nt
        time_step!(model, Δt)
    end

    return model
end

function test_immersed_top_rest_state(FT, arch)
    underlying_grid = RectilinearGrid(arch, FT, size=(16, 20), x=(0, 1), z=(-1, 0),
                                      topology=(Bounded, Flat, Bounded))

    top(x) = min(0, -0.93 + 1.7x)
    b(x, z) = z / 2
    top_load = TopLoad(BuoyancyTracer(), (; b))

    for Bottom in (GridFittedBottom, PartialCellBottom)
        ibg = ImmersedBoundaryGrid(underlying_grid, Bottom(-0.98; top_height=top))
        @test isnothing(ibg.immersed_boundary.top_load)

        control = rest_state_model(ibg)
        @test maximum(abs, interior(control.velocities.u)) > 1e-3

        Φ = top_load_potential(ibg, BuoyancyTracer(), (; b))
        field_loaded_ibg = ImmersedBoundaryGrid(underlying_grid, Bottom(-0.98; top_height=top, top_load=Φ))
        loaded_ibg = ImmersedBoundaryGrid(underlying_grid, Bottom(-0.98; top_height=top, top_load))

        @test Array(bottom_height_interior(loaded_ibg.immersed_boundary.top_load)) == Array(interior(Φ))
        @test loaded_ibg.immersed_boundary == field_loaded_ibg.immersed_boundary
        @test loaded_ibg.immersed_boundary != ibg.immersed_boundary

        model = rest_state_model(loaded_ibg)
        tol = 5000 * eps(FT)
        @test maximum(abs, interior(model.velocities.u)) ≤ tol
        @test maximum(abs, interior(model.velocities.w)) ≤ tol
        @test maximum(abs, interior(model.free_surface.displacement)) ≤ tol
    end

    return nothing
end

function test_immersed_top_zstar_rest_state(FT, arch)
    z = MutableVerticalDiscretization(collect(range(-1, 0, length=21)))
    underlying_grid = RectilinearGrid(arch, FT; size=(16, 20), x=(0, 1), z, topology=(Bounded, Flat, Bounded))

    top(x) = min(0, -0.93 + 1.7x)
    top_load = TopLoad(BuoyancyTracer(), (; b = (x, z) -> z / 2))

    for Bottom in (GridFittedBottom, PartialCellBottom)
        ibg = ImmersedBoundaryGrid(underlying_grid, Bottom(-0.98; top_height=top, top_load))
        model = rest_state_model(ibg; vertical_coordinate=ZStarCoordinate(), timestepper=:SplitRungeKutta3)

        tol = 5000 * eps(FT)
        @test maximum(abs, interior(model.velocities.u)) ≤ tol
        @test maximum(abs, interior(model.velocities.w)) ≤ tol
        @test maximum(abs, interior(model.free_surface.displacement)) ≤ tol
    end

    return nothing
end

@testset "Immersed top dynamics" begin
    for arch in archs, FT in float_types
        @testset "Wall distances [$FT, $(typeof(arch))]"                       test_immersed_top_wall_distances(FT, arch)
        @testset "CATKE beneath an immersed top [$FT, $(typeof(arch))]"        test_immersed_top_catke(FT, arch)
        @testset "Uniform tracer beneath an immersed top [$FT, $(typeof(arch))]" test_immersed_top_uniform_tracer(FT, arch)
        @testset "z★ conservation beneath an immersed top [$FT, $(typeof(arch))]" test_immersed_top_zstar_conservation(FT, arch)
        @testset "z★ freshwater flux beneath an immersed top [$FT, $(typeof(arch))]" test_immersed_top_zstar_freshwater_flux(FT, arch)
        @testset "z★ freshwater flux rest state beneath an immersed top [$FT, $(typeof(arch))]" test_immersed_top_zstar_freshwater_flux_rest_state(FT, arch)
        @testset "z★ znode beneath an immersed top [$FT, $(typeof(arch))]"     test_immersed_top_zstar_znode(FT, arch)
        @testset "Top load potential [$FT, $(typeof(arch))]"                   test_immersed_top_load_potential(FT, arch)
        @testset "Rest state beneath a loaded immersed top [$FT, $(typeof(arch))]" test_immersed_top_rest_state(FT, arch)
        @testset "z★ rest state beneath a loaded immersed top [$FT, $(typeof(arch))]" test_immersed_top_zstar_rest_state(FT, arch)
    end
end
