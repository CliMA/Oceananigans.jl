include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, mask_immersed_field!
using Oceananigans.Utils: NormalDivision
using Oceananigans.Advection:
        GhostCells,
        ghost_cell_reconstruction,
        _symmetric_interpolate_xᶠᵃᵃ,
        _symmetric_interpolate_xᶜᵃᵃ,
        _symmetric_interpolate_yᵃᶠᵃ,
        _symmetric_interpolate_yᵃᶜᵃ,
        _biased_interpolate_xᶜᵃᵃ,
        _biased_interpolate_xᶠᵃᵃ,
        _biased_interpolate_yᵃᶜᵃ,
        _biased_interpolate_yᵃᶠᵃ,
        FluxFormAdvection,
        LeftBias,
        RightBias,
        materialize_advection,
        div_𝐯u

linear_advection_schemes = [Centered, UpwindBiased]
advection_schemes = [linear_advection_schemes... WENO]

@inline advective_order(buffer, ::Type{Centered}) = buffer * 2
@inline advective_order(buffer, AdvectionType)    = buffer * 2 - 1

# Schemes wider than the lowest order are reduced in the z-direction of grids with a single cell
reduced_z_advection_logs(adv, buffer) =
    buffer == 1      ? () :
    adv == Centered  ? ((:info, "Using the advection scheme Centered(order=2) in the z-direction because size(grid, 3) = 1"),) :
                       ((:info, "Using the advection scheme UpwindBiased(order=1) in the z-direction because size(grid, 3) = 1"),)

function run_tracer_interpolation_test(c, ibg, scheme)
    scheme = materialize_advection(scheme, ibg)
    for j in 6:19, i in 6:19
        if typeof(scheme) <: Centered
            @test @allowscalar  _symmetric_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, c) ≈ 1.0
        else
            @test @allowscalar _biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, LeftBias,  c) ≈ 1.0
            @test @allowscalar _biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, RightBias, c) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, scheme, LeftBias,  c) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, scheme, RightBias, c) ≈ 1.0
        end
    end
end

function run_tracer_conservation_test(grid, scheme)
    scheme = materialize_advection(scheme, grid)
    model = HydrostaticFreeSurfaceModel(grid; tracers = :c,
                                        free_surface = ExplicitFreeSurface(),
                                        tracer_advection = scheme)

    c = model.tracers.c
    set!(model, c = 1)
    fill_halo_regions!(c)

    η = model.free_surface.displacement

    indices = model.grid isa ImmersedBoundaryGrid ? (5:7, 3:6, 1) : (2:5, 3:6, 1)

    interior(η, indices...) .= - 0.05
    fill_halo_regions!(η)

    wave_speed = sqrt(model.free_surface.gravitational_acceleration)
    dt = 0.1 / wave_speed
    for _ in 1:10
        time_step!(model, dt)
    end

    @test maximum(c) ≈ 1.0
    @test minimum(c) ≈ 1.0
    @test mean(c)    ≈ 1.0

    return nothing
end

function run_momentum_interpolation_test(u, v, ibg, scheme)
    scheme = materialize_advection(scheme, ibg)

    # ensure also immersed boundaries have a value of 1
    interior(u, 6, :, 1) .= 1.0
    interior(v, :, 6, 1) .= 1.0

    for i in 7:19, j in 7:19
        if typeof(scheme) <: Centered
            @test @allowscalar  _symmetric_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, u) ≈ 1.0
            @test @allowscalar  _symmetric_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, v) ≈ 1.0
            @test @allowscalar  _symmetric_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, u) ≈ 1.0
            @test @allowscalar  _symmetric_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, v) ≈ 1.0
        else
            @test @allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, LeftBias,  u) ≈ 1.0
            @test @allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, RightBias, u) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, LeftBias,  u) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, RightBias, u) ≈ 1.0

            @test @allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, LeftBias,  v) ≈ 1.0
            @test @allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, RightBias, v) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, LeftBias,  v) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, RightBias, v) ≈ 1.0
        end
    end

    return nothing
end

function left_of_thin_wall_tendency(arch, scheme, right_velocity)
    grid = RectilinearGrid(arch, size=(17, 8), x=(0, 17), z=(0, 8), halo=(6, 6), topology=(Bounded, Flat, Bounded))
    ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, z) -> 8 < x < 9))

    u = XFaceField(ibg)
    set!(u, (x, z) -> x < 8 ? -1 : x > 9 ? right_velocity : 0)
    velocities = (u, YFaceField(ibg), ZFaceField(ibg))

    advection = materialize_advection(scheme, ibg)
    G = Field(KernelFunctionOperation{Face, Center, Center}(div_𝐯u, ibg, advection, velocities, u))

    return Array(interior(G, 1:8, 1, :))
end

for arch in archs
    @testset "Immersed tracer reconstruction" begin
        grid = RectilinearGrid(arch, size=(20, 20), extent=(20, 20), halo = (6, 6), topology=(Bounded, Bounded, Flat))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y) -> (x < 5 || y < 5)))

        c = CenterField(ibg)
        set!(c, 1)
        mask_immersed_field!(c)
        fill_halo_regions!(c)

        for adv in linear_advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))

            @testset "$(summary(scheme)) [$(summary(arch))]" begin
                run_tracer_interpolation_test(c, ibg, scheme)
            end
        end

        for buffer in [2, 3, 4, 5], bounds in (nothing, (0, 1))
            scheme = WENO(; order = advective_order(buffer, WENO), bounds)

            @testset "$(summary(scheme)) [$(summary(arch))]" begin
                run_tracer_interpolation_test(c, ibg, scheme)
            end
        end
    end

    @testset "Immersed tracer conservation" begin
        grid = RectilinearGrid(arch, size=(10, 8, 1), extent=(10, 8, 1), halo = (6, 6, 6), topology=(Bounded, Periodic, Bounded))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> ifelse(x < 2, 0, -1)))

        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))

            @testset "$(summary(scheme)) on $(nameof(typeof(g))) [$(summary(arch))]" for g in [grid, ibg]
                @test_logs reduced_z_advection_logs(adv, buffer)... run_tracer_conservation_test(g, scheme)
            end
        end

        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            directional_scheme = adv(order = advective_order(buffer, adv))
            scheme = FluxFormAdvection(directional_scheme, directional_scheme, directional_scheme)

            @testset "$(summary(scheme)) on $(nameof(typeof(g))) [$(summary(arch))]" for g in [grid, ibg]
                @test_logs reduced_z_advection_logs(adv, buffer)... run_tracer_conservation_test(g, scheme)
            end
        end
    end

    @testset "GhostCells boundary reconstruction" begin
        @info "Running GhostCells boundary reconstruction tests..."

        ghost_cells(; kw...) = WENO(order=7, weight_computation=NormalDivision, boundary_scheme=GhostCells(; kw...)).buffer_scheme

        function reconstruct(scheme, bias, values, activity)
            upwind_values   = bias == LeftBias ? values   : reverse(values)
            upwind_activity = bias == LeftBias ? activity : reverse(activity)
            upstream_run    = accumulate(&, upwind_activity[3:-1:1])
            downstream_run  = accumulate(&, upwind_activity[5:7])
            return ghost_cell_reconstruction(scheme, bias, upwind_values[1:7], upstream_run, downstream_run, nothing)
        end

        active = ntuple(_ -> true, 8)
        wall   = (false, false, true, true, true, true, true, true)
        scheme = ghost_cells()

        # the values in inactive cells never enter the reconstruction
        linear = (1e3, 1e3, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

        for bias in (LeftBias, RightBias)
            @test reconstruct(scheme, bias, ntuple(_ -> 3.0, 8), wall) == 3.0
            @test reconstruct(scheme, bias, linear, wall) ≈ 4.5
            @test reconstruct(scheme, bias, Tuple(1.0:8.0), active) == 4.5
        end

        pulse = (0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)

        for bias in (LeftBias, RightBias)
            @test 0 ≤ reconstruct(scheme, bias, pulse, wall) ≤ 1
            @test reconstruct(ghost_cells(monotone=false), bias, pulse, wall) > 1
        end

        grid = RectilinearGrid(arch, size=(20, 20), extent=(20, 20), halo=(6, 6), topology=(Bounded, Bounded, Flat))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y) -> (x < 5 || y < 5)))

        c = CenterField(ibg)
        set!(c, 1)
        mask_immersed_field!(c, 1e3)
        fill_halo_regions!(c)

        weno = materialize_advection(WENO(order=7, boundary_scheme=GhostCells()), ibg)
        for j in 6:19, i in 6:19, bias in (LeftBias, RightBias)
            @test @allowscalar _biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, weno, bias, c) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, weno, bias, c) ≈ 1.0
        end
    end

    @testset "Immersed momentum reconstruction" begin
        grid = RectilinearGrid(arch, size=(20, 20), extent=(20, 20), halo = (6, 6), topology=(Bounded, Bounded, Flat))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y) -> (x < 5 || y < 5)))

        u = XFaceField(ibg)
        v = YFaceField(ibg)
        set!(u, 1)
        set!(v, 1)

        mask_immersed_field!(u)
        mask_immersed_field!(v)

        fill_halo_regions!(u)
        fill_halo_regions!(v)

        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))

            @testset "$(summary(scheme)) [$(summary(arch))]" begin
                run_momentum_interpolation_test(u, v, ibg, scheme)
            end
        end
    end

    @testset "Momentum advection does not reach across a one-cell immersed wall" begin
        @testset "$(summary(scheme)) [$(summary(arch))]" for scheme in (UpwindBiased(order=5), WENO(), WENO(order=9))
            @test left_of_thin_wall_tendency(arch, scheme, 0) == left_of_thin_wall_tendency(arch, scheme, 1)
        end
    end
end
