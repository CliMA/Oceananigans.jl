include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, mask_immersed_field!
using Oceananigans.Advection:
        symmetric_interpolate_xᶠᶜᶜ,
        symmetric_interpolate_xᶜᶜᶜ,
        symmetric_interpolate_xᶜᶠᶜ,
        symmetric_interpolate_yᶠᶜᶜ,
        symmetric_interpolate_yᶜᶜᶜ,
        biased_interpolate_xᶠᶜᶜ,
        biased_interpolate_xᶜᶜᶜ,
        biased_interpolate_xᶜᶠᶜ,
        biased_interpolate_yᶜᶠᶜ,
        biased_interpolate_yᶠᶜᶜ,
        biased_interpolate_yᶜᶜᶜ,
        FluxFormAdvection,
        LeftBias,
        RightBias,
        materialize_advection,
        div_𝐯u,
        div_Uc,
        _advective_tracer_flux_x,
        _advective_tracer_flux_y,
        _advective_tracer_flux_z

using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.TimeSteppers: compute_tendencies!
using Oceananigans.Operators: Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ, V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ
using Oceananigans.Grids: inactive_cell, peripheral_node

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
            @test @allowscalar  symmetric_interpolate_xᶠᶜᶜ(i+1, j, 1, ibg, scheme, c) ≈ 1.0
        else
            @test @allowscalar biased_interpolate_xᶠᶜᶜ(i+1, j, 1, ibg, scheme, LeftBias,  c) ≈ 1.0
            @test @allowscalar biased_interpolate_xᶠᶜᶜ(i+1, j, 1, ibg, scheme, RightBias, c) ≈ 1.0
            @test @allowscalar biased_interpolate_yᶜᶠᶜ(i, j+1, 1, ibg, scheme, LeftBias,  c) ≈ 1.0
            @test @allowscalar biased_interpolate_yᶜᶠᶜ(i, j+1, 1, ibg, scheme, RightBias, c) ≈ 1.0
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
            @test @allowscalar  symmetric_interpolate_xᶜᶜᶜ(i+1, j, 1, ibg, scheme, u) ≈ 1.0
            @test @allowscalar  symmetric_interpolate_xᶜᶠᶜ(i+1, j, 1, ibg, scheme, v) ≈ 1.0
            @test @allowscalar  symmetric_interpolate_yᶠᶜᶜ(i, j+1, 1, ibg, scheme, u) ≈ 1.0
            @test @allowscalar  symmetric_interpolate_yᶜᶜᶜ(i, j+1, 1, ibg, scheme, v) ≈ 1.0
        else
            @test @allowscalar biased_interpolate_xᶜᶜᶜ(i+1, j, 1, ibg, scheme, LeftBias,  u) ≈ 1.0
            @test @allowscalar biased_interpolate_xᶜᶜᶜ(i+1, j, 1, ibg, scheme, RightBias, u) ≈ 1.0
            @test @allowscalar biased_interpolate_yᶠᶜᶜ(i, j+1, 1, ibg, scheme, LeftBias,  u) ≈ 1.0
            @test @allowscalar biased_interpolate_yᶠᶜᶜ(i, j+1, 1, ibg, scheme, RightBias, u) ≈ 1.0

            @test @allowscalar biased_interpolate_xᶜᶠᶜ(i+1, j, 1, ibg, scheme, LeftBias,  v) ≈ 1.0
            @test @allowscalar biased_interpolate_xᶜᶠᶜ(i+1, j, 1, ibg, scheme, RightBias, v) ≈ 1.0
            @test @allowscalar biased_interpolate_yᶜᶜᶜ(i, j+1, 1, ibg, scheme, LeftBias,  v) ≈ 1.0
            @test @allowscalar biased_interpolate_yᶜᶜᶜ(i, j+1, 1, ibg, scheme, RightBias, v) ≈ 1.0
        end
    end

    return nothing
end

staircase(ξ) = (5 + sum(tanh(40 * (ξ - n / 6)) for n in 1:5)) / 20

function run_kinetic_energy_conservation_test(arch, Bottom)
    grid = RectilinearGrid(arch, size=(16, 16, 32), x=(0, 1), y=(0, 1), z=(0, 1), topology=(Bounded, Bounded, Bounded))
    ibg  = ImmersedBoundaryGrid(grid, Bottom((x, y) -> staircase(x) + staircase(y)))

    pressure_solver = ConjugateGradientPoissonSolver(ibg; reltol=1e-14, abstol=0, maxiter=1000)
    model = NonhydrostaticModel(ibg; advection=Centered(), pressure_solver)

    u, v, w = model.velocities
    set!(model, u = rand(size(u)...) .- 1/2, v = rand(size(v)...) .- 1/2, w = rand(size(w)...) .- 1/2)
    compute_tendencies!(model, [])
    G = model.timestepper.Gⁿ

    Vu = KernelFunctionOperation{Face, Center, Center}(Vᶠᶜᶜ, ibg)
    Vv = KernelFunctionOperation{Center, Face, Center}(Vᶜᶠᶜ, ibg)
    Vw = KernelFunctionOperation{Center, Center, Face}(Vᶜᶜᶠ, ibg)

    production = sum(Vu * u * G.u) + sum(Vv * v * G.v) + sum(Vw * w * G.w)
    magnitude  = sum(abs, Vu * u * G.u) + sum(abs, Vv * v * G.v) + sum(abs, Vw * w * G.w)

    @test abs(production) < 1e-12 * magnitude

    return nothing
end

function run_staircase_convection_test(arch)
    grid = RectilinearGrid(arch, size=(16, 128), x=(0, 1), z=(0, 1), topology=(Bounded, Flat, Bounded))
    ibg  = ImmersedBoundaryGrid(grid, PartialCellBottom(x -> 2staircase(x)))

    model = NonhydrostaticModel(ibg; advection=WENO(), tracers=:b, buoyancy=BuoyancyTracer(),
                                pressure_solver=ConjugateGradientPoissonSolver(ibg))

    set!(model, b = (x, z) -> - exp(-((x - 1/2)^2 + (z - 0.55)^2) / (2 * 0.05^2)))

    simulation = Simulation(model; Δt=1e-3, stop_time=1, verbose=false)
    conjure_time_step_wizard!(simulation, cfl=0.7)
    run!(simulation)

    @test maximum(abs, model.velocities.u) < 1

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

@inline face_flux_divergence(i, j, k, grid, advection, U, c) =
    V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, U.u, c) +
                             δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, U.v, c) +
                             δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, advection, U.w, c))

@inline active_cell_difference(i, j, k, grid, advection, U, c) =
    ifelse(inactive_cell(i, j, k, grid), zero(grid), div_Uc(i, j, k, grid, advection, U, c) - face_flux_divergence(i, j, k, grid, advection, U, c))

@inline no_wall_flow(i, j, k, grid, ψ, ℓ) = ifelse(peripheral_node(i, j, k, grid, ℓ...), zero(grid), @inbounds ψ[i, j, k])

# the largest difference between div_Uc and the divergence of the face fluxes over the active cells, relative to the latter,
# from random fields without flow through the walls
function cell_and_face_tracer_divergence(arch, scheme)
    grid = RectilinearGrid(arch, size=(12, 12, 12), extent=(1, 1, 1), halo=(6, 6, 6), topology=(Periodic, Bounded, Bounded))
    ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -0.6 + 0.25 * sin(2π * x) * cos(π * y)))

    Random.seed!(1234)
    u, v, w, c = XFaceField(ibg), YFaceField(ibg), ZFaceField(ibg), CenterField(ibg)
    for ψ in (u, v, w, c)
        set!(ψ, rand(size(ψ)...) .- 0.3)
        fill_halo_regions!(ψ)
    end

    u = compute!(Field(KernelFunctionOperation{Face, Center, Center}(no_wall_flow, ibg, u, (Face(), Center(), Center()))))
    v = compute!(Field(KernelFunctionOperation{Center, Face, Center}(no_wall_flow, ibg, v, (Center(), Face(), Center()))))
    w = compute!(Field(KernelFunctionOperation{Center, Center, Face}(no_wall_flow, ibg, w, (Center(), Center(), Face()))))
    foreach(fill_halo_regions!, (u, v, w))

    advection  = materialize_advection(scheme, ibg)
    U          = (; u, v, w)
    difference = compute!(Field(KernelFunctionOperation{Center, Center, Center}(active_cell_difference, ibg, advection, U, c)))
    divergence = compute!(Field(KernelFunctionOperation{Center, Center, Center}(face_flux_divergence, ibg, advection, U, c)))

    return maximum(abs, difference) / maximum(abs, divergence)
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

    @testset "Immersed momentum advection conserves kinetic energy" begin
        @testset "$Bottom [$(summary(arch))]" for Bottom in (GridFittedBottom, PartialCellBottom)
            run_kinetic_energy_conservation_test(arch, Bottom)
        end
    end

    @testset "Convection over an immersed staircase stays bounded [$(summary(arch))]" begin
        run_staircase_convection_test(arch)
    end

    @testset "Momentum advection does not reach across a one-cell immersed wall" begin
        @testset "$(summary(scheme)) [$(summary(arch))]" for scheme in (UpwindBiased(order=5), WENO(), WENO(order=9))
            @test left_of_thin_wall_tendency(arch, scheme, 0) == left_of_thin_wall_tendency(arch, scheme, 1)
        end
    end

    @testset "Tracer flux divergence of the cell stencils equals that of the face fluxes" begin
        @testset "$(summary(scheme)) [$(summary(arch))]" for scheme in (Centered(order=4), UpwindBiased(order=5), WENO(order=5), WENO(order=7), WENO(order=9))
            @test cell_and_face_tracer_divergence(arch, scheme) < 1e-12
        end
    end
end
