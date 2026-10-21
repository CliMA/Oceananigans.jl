include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, mask_immersed_field!, immersed_cell, peripheral_node
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.TimeSteppers: compute_tendencies!
using Oceananigans.Operators: Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ
using Oceananigans.Utils: NormalDivision
using Oceananigans.Advection:
        div_𝐯u,
        div_𝐯v,
        div_𝐯w,
        HierarchicalExtrapolation,
        ghost_cell_reconstruction,
        node_differences,
        symmetric_interpolate_xᶠᶜᶜ,
        symmetric_interpolate_xᶜᶜᶜ,
        symmetric_interpolate_xᶠᶠᶜ,
        symmetric_interpolate_yᶜᶠᶜ,
        symmetric_interpolate_yᶠᶠᶜ,
        symmetric_interpolate_yᶜᶜᶜ,
        biased_interpolate_xᶠᶜᶜ,
        biased_interpolate_xᶜᶜᶜ,
        biased_interpolate_xᶠᶠᶜ,
        biased_interpolate_yᶜᶠᶜ,
        biased_interpolate_yᶠᶠᶜ,
        biased_interpolate_yᶜᶜᶜ,
        FluxFormAdvection,
        LeftBias,
        RightBias,
        materialize_advection,
        materialize_wall_values,
        wall_values

using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

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

    # u at (Face, Center, Center) reconstructs to (Center, Center, Center) along x and (Face, Face, Center) along y,
    # v at (Center, Face, Center) to (Face, Face, Center) along x and (Center, Center, Center) along y
    for i in 7:19, j in 7:19
        if typeof(scheme) <: Centered
            @test @allowscalar  symmetric_interpolate_xᶜᶜᶜ(i, j, 1, ibg, scheme, u) ≈ 1.0
            @test @allowscalar  symmetric_interpolate_xᶠᶠᶜ(i, j, 1, ibg, scheme, v) ≈ 1.0
            @test @allowscalar  symmetric_interpolate_yᶠᶠᶜ(i, j, 1, ibg, scheme, u) ≈ 1.0
            @test @allowscalar  symmetric_interpolate_yᶜᶜᶜ(i, j, 1, ibg, scheme, v) ≈ 1.0
        else
            for bias in (LeftBias, RightBias)
                @test @allowscalar biased_interpolate_xᶜᶜᶜ(i, j, 1, ibg, scheme, bias, u) ≈ 1.0
                @test @allowscalar biased_interpolate_yᶠᶠᶜ(i, j, 1, ibg, scheme, bias, u) ≈ 1.0
                @test @allowscalar biased_interpolate_xᶠᶠᶜ(i, j, 1, ibg, scheme, bias, v) ≈ 1.0
                @test @allowscalar biased_interpolate_yᶜᶜᶜ(i, j, 1, ibg, scheme, bias, v) ≈ 1.0
            end
        end
    end

    return nothing
end

staircase(ξ) = (5 + sum(tanh(40 * (ξ - n / 6)) for n in 1:5)) / 20

function run_uniform_momentum_preservation_test(arch, Bottom, scheme)
    grid = RectilinearGrid(arch, size=(16, 16, 32), halo=(6, 6, 6), x=(0, 1), y=(0, 1), z=(0, 1), topology=(Bounded, Bounded, Bounded))
    ibg  = ImmersedBoundaryGrid(grid, Bottom((x, y) -> staircase(x) + staircase(y)))

    pressure_solver = ConjugateGradientPoissonSolver(ibg; reltol=1e-14, abstol=0, maxiter=1000)
    model = NonhydrostaticModel(ibg; advection=scheme, pressure_solver)

    u, v, w = model.velocities
    set!(model, u = rand(size(u)...) .- 1/2, v = rand(size(v)...) .- 1/2, w = rand(size(w)...) .- 1/2)

    uniform = XFaceField(ibg)
    parent(uniform) .= 1

    advection = model.advection.momentum
    Gu = KernelFunctionOperation{Face, Center, Center}(div_𝐯u, ibg, advection, model.velocities, uniform)
    Gv = KernelFunctionOperation{Center, Face, Center}(div_𝐯v, ibg, advection, model.velocities, uniform)
    Gw = KernelFunctionOperation{Center, Center, Face}(div_𝐯w, ibg, advection, model.velocities, uniform)

    active(ℓx, ℓy, ℓz) = (i, j, k, grid, G) -> !peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz)
    scale = maximum(abs, u) / minimum_xspacing(ibg)

    @test maximum(abs, Gu; condition=active(Face(), Center(), Center())) < 1e-10 * scale
    @test maximum(abs, Gv; condition=active(Center(), Face(), Center())) < 1e-10 * scale
    @test maximum(abs, Gw; condition=active(Center(), Center(), Face())) < 1e-10 * scale

    return nothing
end

function run_uniform_through_flow_test(arch, scheme)
    grid = RectilinearGrid(arch, size=(16, 16), halo=(5, 5), x=(0, 1), z=(0, 1), topology=(Bounded, Flat, Bounded))

    through_flow = NormalFlowBoundaryCondition(1)
    u_bcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); west=through_flow, east=through_flow)
    u = XFaceField(grid, boundary_conditions=u_bcs)
    set!(u, 1)
    fill_halo_regions!(u)

    velocities = (; u, v=YFaceField(grid), w=ZFaceField(grid))
    Gu = KernelFunctionOperation{Face, Center, Center}(div_𝐯u, grid, materialize_advection(scheme, grid), velocities, u)
    interior_face(i, j, k, grid, G) = !peripheral_node(i, j, k, grid, Face(), Center(), Center())

    @test maximum(abs, Gu; condition=interior_face) < 1e-12

    return nothing
end

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

    @testset "Boundary reconstruction with ghosts" begin
        scheme = WENO(order=7, weight_computation=NormalDivision)

        # L and R are the nodes 4 and 5; nᴸ and nᴿ count the nodes beyond them before the first peripheral one
        function reconstruct(scheme, bias, values, peripheral)
            nᴸ = something(findfirst(peripheral[3:-1:1]), 4) - 1
            nᴿ = something(findfirst(peripheral[6:8]), 4) - 1
            return ghost_cell_reconstruction(scheme, values, node_differences(values), nᴸ, nᴿ, bias, nothing, HierarchicalExtrapolation())
        end

        # Averages over the unit cells [n - 1, n] of a polynomial with primitive P; the reconstructed face is at x = 4
        averages(P) = ntuple(n -> P(n) - P(n - 1), 8)

        none     = ntuple(_ -> false, 8)
        wall     = (true, true, false, false, false, false, false, false)
        two_left = (true, true, true, false, false, true, true, true)

        linear_primitive(x) = x^2 / 2 + 2x # p(x) = x + 2

        for bias in (LeftBias, RightBias)
            # the values in inactive cells never enter the reconstruction
            @test reconstruct(scheme, bias, (1e3, 1e3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), wall) == 3.0
            @test reconstruct(scheme, bias, Tuple(1.0:8.0), none) == 4.5

            # the ghosts extrapolate linearly, so a uniform stratification next to a wall is reconstructed exactly
            @test reconstruct(scheme, bias, averages(linear_primitive), wall) ≈ 4 + 2
            @test reconstruct(scheme, bias, averages(linear_primitive), two_left) ≈ 4 + 2
        end


        # the ghosts lie on the line through the active cells in index coordinates, those of the WENO reconstruction
        affine = Tuple(2.5 .- 0.7 .* (1:8))
        lowest_peripheral = (true, false, false, false, false, false, false, false)
        @test reconstruct(scheme, LeftBias, (1e3, affine[2:8]...), lowest_peripheral) ≈ reconstruct(scheme, LeftBias, affine, none)

        grid = RectilinearGrid(arch, size=(20, 20), extent=(20, 20), halo=(6, 6), topology=(Bounded, Bounded, Flat))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y) -> (x < 5 || y < 5)))

        c = CenterField(ibg)
        set!(c, 1)
        mask_immersed_field!(c, 1e3)
        fill_halo_regions!(c)

        weno = materialize_advection(WENO(order=7), ibg)
        for j in 6:19, i in 6:19, bias in (LeftBias, RightBias)
            @test @allowscalar biased_interpolate_xᶠᶜᶜ(i+1, j, 1, ibg, weno, bias, c) ≈ 1.0
            @test @allowscalar biased_interpolate_yᶜᶠᶜ(i, j+1, 1, ibg, weno, bias, c) ≈ 1.0
        end
    end

    @testset "Tracer advection with ghosts over a step" begin
        Nx, Nz = 64, 32
        grid = RectilinearGrid(arch, size=(Nx, Nz), x=(0, 1), z=(-1, 0), halo=(6, 6), topology=(Periodic, Flat, Bounded))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom(x -> 0.35 < x < 0.65 ? -0.35 : -1.0))

        # A divergence-free velocity from a streamfunction that vanishes on the staircase
        ψ(x, z) = 0.05 * sin(2π * x) * sin(π * (z + 1))
        dry(i, k) = k < 1 || k > Nz || (0.35 < (mod1(i, Nx) - 0.5) / Nx < 0.65 && -1 + (k - 0.5) / Nz < -0.35)
        Ψ = [(dry(i-1, k-1) | dry(i, k-1) | dry(i-1, k) | dry(i, k)) ? 0.0 : ψ((i - 1) / Nx, -1 + (k - 1) / Nz) for i in 1:Nx+1, k in 1:Nz+1]
        u = XFaceField(ibg)
        w = ZFaceField(ibg)
        set!(u, [-(Ψ[i, k+1] - Ψ[i, k]) * Nz for i in 1:Nx, _ in 1:1, k in 1:Nz])
        set!(w, [ (Ψ[i+1, k] - Ψ[i, k]) * Nx for i in 1:Nx, _ in 1:1, k in 1:Nz+1])

        # The inactive cells of a tracer advected with the default `TransportedWallValue` store the wall values
        cpu_ibg = on_architecture(CPU(), ibg)
        wet = [!immersed_cell(i, 1, k, cpu_ibg) for i in 1:Nx, _ in 1:1, k in 1:Nz]

        function advected_tracer(scheme; timestepper = :SplitRungeKutta3, Δt = 0.01, steps = 100)
            model = HydrostaticFreeSurfaceModel(ibg; velocities = PrescribedVelocityFields(; u, w), tracers = :c,
                                                tracer_advection = scheme, timestepper)
            set!(model, c = (x, z) -> ifelse(-0.6 < z < -0.4, 1.0, 0.0))
            c = model.tracers.c
            total₀ = sum(Array(interior(c))[wet])
            cmin, cmax = 0.0, 1.0
            for _ in 1:steps
                time_step!(model, Δt)
                cmin = min(cmin, minimum(Array(interior(c))[wet]))
                cmax = max(cmax, maximum(Array(interior(c))[wet]))
            end
            return (; cmin, cmax, drift = sum(Array(interior(c))[wet]) - total₀)
        end

        hierarchical = advected_tracer(WENO(order=7, extrapolation=HierarchicalExtrapolation()))
        wall_values = advected_tracer(WENO(order=7))

        # The one-cell wall jet of this flow leaves the interior WENO reconstruction an excess of about 5%
        @test -0.02 < hierarchical.cmin
        @test hierarchical.cmax < 1.06
        @test abs(hierarchical.drift) < 1e-10

        @test -0.02 < wall_values.cmin
        @test wall_values.cmax < 1.05
        @test abs(wall_values.drift) < 1e-10

        # The Adams-Bashforth time stepper advances the wall values too, within its smaller stable time step
        adams_bashforth = advected_tracer(WENO(order=7); timestepper = :QuasiAdamsBashforth2, Δt = 0.0025, steps = 400)
        @test -0.02 < adams_bashforth.cmin
        @test adams_bashforth.cmax < 1.05
        @test abs(adams_bashforth.drift) < 1e-10
    end

    @testset "Wall values in a NonhydrostaticModel" begin
        grid = RectilinearGrid(arch, size=(64, 16), x=(0, 4), z=(-1, 0), halo=(6, 6), topology=(Bounded, Flat, Bounded))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom(x -> -1 + 0.4 * exp(-(x - 2)^2 / 0.1)))
        cpu_ibg = on_architecture(CPU(), ibg)
        wet = [!immersed_cell(i, 1, k, cpu_ibg) for i in 1:64, _ in 1:1, k in 1:16]

        for timestepper in (:RungeKutta3, :QuasiAdamsBashforth2)
            model = NonhydrostaticModel(ibg; tracers = :b, buoyancy = BuoyancyTracer(), timestepper, advection = WENO(order=5),
                                        pressure_solver = ConjugateGradientPoissonSolver(ibg))
            set!(model, b = (x, z) -> ifelse(x < 1, 0.01, 0.0))
            for _ in 1:100
                time_step!(model, 0.05)
            end
            b = Array(interior(model.tracers.b))[wet]
            @test -2e-5 < minimum(b) && maximum(b) < 0.01 + 2e-5
        end
    end

    @testset "Ghosts on a z-star grid" begin
        Nx, Nz = 64, 32
        grid = RectilinearGrid(arch, size=(Nx, Nz), x=(-1e6, 1e6), z=MutableVerticalDiscretization((-2000, 0)), halo=(6, 6),
                               topology=(Periodic, Flat, Bounded))
        ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(x -> -2000 + 750 * exp(-x^2 / 8e8)))
        model = HydrostaticFreeSurfaceModel(ibg; timestepper = :SplitRungeKutta3, buoyancy = BuoyancyTracer(), tracers = :b,
                                            tracer_advection = WENO(order=7),
                                            momentum_advection = WENO(order=5))
        set!(model, u = (x, z) -> 0.3 * exp(-x^2 / 1e10), b = (x, z) -> 1e-4 * z)
        for _ in 1:20
            time_step!(model, 600)
        end

        b = Array(interior(model.tracers.b))
        @test @allowscalar maximum(abs, model.free_surface.displacement) > 0.1
        @test all(isfinite, b)
        @test -0.2 - 1e-10 ≤ minimum(b) && maximum(b) ≤ 1e-10
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

    @testset "Immersed momentum advection preserves a uniform field" begin
        @testset "$(summary(scheme)), $Bottom [$(summary(arch))]" for Bottom in (GridFittedBottom, PartialCellBottom),
                                                                      scheme in (Centered(), UpwindBiased(), WENO(), WENO(order=9))
            run_uniform_momentum_preservation_test(arch, Bottom, scheme)
        end
    end

    @testset "Uniform through-flow stays uniform [$(summary(arch))]" begin
        @testset "$(summary(scheme))" for scheme in (Centered(), UpwindBiased(), WENO(), WENO(order=9))
            run_uniform_through_flow_test(arch, scheme)
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
end

@testset "Wall faces and stored wall values" begin
    grid = RectilinearGrid(CPU(), size=16, x=(0, 1), halo=6, topology=(Bounded, Flat, Flat))
    N    = size(grid, 1)
    xᶠ   = xnodes(grid, Face())
    bcs  = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :c)
    line(x) = 3//10 + 17x / 10

    c = CenterField(grid)
    set!(c, line)

    face_value(scheme::Centered, i, bias) = symmetric_interpolate_xᶠᶜᶜ(i, 1, 1, grid, scheme, c)
    face_value(scheme, i, bias) = biased_interpolate_xᶠᶜᶜ(i, 1, 1, grid, scheme, bias, c)

    schemes = (WENO(order=5), WENO(order=9), UpwindBiased(order=3), UpwindBiased(order=5), Centered(order=4), Centered(order=6))

    @testset "A wall face takes the node with data [$(summary(scheme))]" for scheme in schemes
        scheme = materialize_advection(scheme, grid)
        @test all(face_value(scheme, 1,     bias) == c[1, 1, 1] for bias in (LeftBias, RightBias))
        @test all(face_value(scheme, N + 1, bias) == c[N, 1, 1] for bias in (LeftBias, RightBias))
    end

    @testset "Stored wall values reconstruct a line exactly [$(summary(scheme))]" for scheme in schemes
        stored = materialize_wall_values(materialize_advection(scheme, grid), grid, bcs)
        c[0, 1, 1]     = line(0)
        c[N + 1, 1, 1] = line(1)
        @test all(face_value(stored, i, bias) ≈ line(xᶠ[i]) for i in 2:N, bias in (LeftBias, RightBias))

        # the wall value enters the faces next to the wall
        c[0, 1, 1] = line(0) + 1
        @test !all(face_value(stored, i, bias) ≈ line(xᶠ[i]) for i in 2:N, bias in (LeftBias, RightBias))
    end
end

@testset "Wall values with an adapted, direction-based advection" begin
    grid = RectilinearGrid(CPU(), size=(16, 16, 1), x=(0, 1), y=(0, 1), z=(-1, 0), halo=(5, 5, 5), topology=(Bounded, Bounded, Bounded))
    ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> (x - 1/2)^2 + (y - 1/2)^2 < 1/25 ? 1/2 : -1))

    model = @test_logs (:info, r"UpwindBiased\(order=1\) in the z-direction") HydrostaticFreeSurfaceModel(ibg; tracers=:c, buoyancy=nothing,
                                                                                                        tracer_advection=WENO(order=7))
    advection = model.advection.c
    walls = wall_values(advection)

    @test advection isa FluxFormAdvection
    @test !isnothing(walls)
    @test advection.x.extrapolation.wall_values === walls
    @test advection.y.extrapolation.wall_values === walls
end
