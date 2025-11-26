include("dependencies_for_runtests.jl")

using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Fields: Field
using Oceananigans.Forcings: MultipleForcings

""" Take one time step with three forcing arrays on u, v, w. """
function time_step_with_forcing_array(arch)
    grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 1, 1))

    Fu = XFaceField(grid)
    Fv = YFaceField(grid)
    Fw = ZFaceField(grid)

    set!(Fu, (x, y, z) -> 1)
    set!(Fv, (x, y, z) -> 1)
    set!(Fw, (x, y, z) -> 1)

    model = NonhydrostaticModel(; grid, forcing=(u=Fu, v=Fv, w=Fw))
    time_step!(model, 1)

    return true
end

""" Take one time step with three forcing functions on u, v, w. """
function time_step_with_forcing_functions(arch)
    @inline Fu(x, y, z, t) = exp(π * z)
    @inline Fv(x, y, z, t) = cos(42 * x)
    @inline Fw(x, y, z, t) = 1.0

    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(; grid, forcing=(u=Fu, v=Fv, w=Fw))
    time_step!(model, 1)

    return true
end

@inline Fu_discrete_func(i, j, k, grid, clock, model_fields) = @inbounds -model_fields.u[i, j, k]
@inline Fv_discrete_func(i, j, k, grid, clock, model_fields, params) = @inbounds - model_fields.v[i, j, k] / params.τ
@inline Fw_discrete_func(i, j, k, grid, clock, model_fields, params) = @inbounds - model_fields.w[i, j, k]^2 / params.τ

""" Take one time step with a DiscreteForcing function. """
function time_step_with_discrete_forcing(arch)
    Fu = Forcing(Fu_discrete_func, discrete_form=true)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(; grid, forcing=(; u=Fu))
    time_step!(model, 1)

    return true
end

""" Take one time step with ParameterizedForcing forcing functions. """
function time_step_with_parameterized_discrete_forcing(arch)

    Fv = Forcing(Fv_discrete_func, parameters=(; τ=60), discrete_form=true)
    Fw = Forcing(Fw_discrete_func, parameters=(; τ=60), discrete_form=true)

    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(; grid, forcing=(v=Fv, w=Fw))
    time_step!(model, 1)

    return true
end

""" Take one time step with a Forcing forcing function with parameters. """
function time_step_with_parameterized_continuous_forcing(arch)
    Fu = Forcing((x, y, z, t, ω) -> sin(ω * x), parameters=π)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(; grid, forcing=(; u=Fu))
    time_step!(model, 1)
    return true
end

""" Take one time step with a Forcing forcing function with parameters. """
function time_step_with_single_field_dependent_forcing(arch, fld)

    fld_forcing = Forcing((x, y, z, t, fld) -> -fld, field_dependencies=fld)

    forcing = if fld == :A # not a prognostic field
        (; T = fld_forcing)
    else
        (; fld => fld_forcing)
    end

    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    A = Field{Center, Center, Center}(grid)
    model = NonhydrostaticModel(; grid, forcing,
                                buoyancy = SeawaterBuoyancy(),
                                tracers = (:T, :S),
                                auxiliary_fields = (; A))
    time_step!(model, 1)

    return true
end

""" Take one time step with a Forcing forcing function with parameters. """
function time_step_with_multiple_field_dependent_forcing(arch)

    Fu = Forcing((x, y, z, t, v, w, T, A) -> sin(v)*exp(w)*T*A, field_dependencies=(:v, :w, :T, :A))

    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    A = Field{Center, Center, Center}(grid)
    model = NonhydrostaticModel(; grid,
                                forcing = (; u=Fu),
                                buoyancy = SeawaterBuoyancy(),
                                tracers = (:T, :S),
                                auxiliary_fields = (; A))
    time_step!(model, 1)

    return true
end


""" Take one time step with a Forcing forcing function with parameters. """
function time_step_with_parameterized_field_dependent_forcing(arch)
    Fu = Forcing((x, y, z, t, u, p) -> sin(p.ω * x) * u, parameters=(ω=π,), field_dependencies=:u)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(; grid, forcing=(; u=Fu))
    time_step!(model, 1)
    return true
end

""" Take one time step with a FieldTimeSeries forcing function. """
function time_step_with_field_time_series_forcing(arch)

    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

    u_forcing = FieldTimeSeries{Face, Center, Center}(grid, 0:1:3)

    for (t, time) in enumerate(u_forcing.times)
        set!(u_forcing[t], (x, y, z) -> sin(π * x) * time)
    end

    model = NonhydrostaticModel(; grid, forcing=(; u=u_forcing))
    time_step!(model, 1)

    # Make sure the field time series updates correctly
    u_forcing = FieldTimeSeries{Face, Center, Center}(grid, 0:1:4; backend = InMemory(2))

    model = NonhydrostaticModel(; grid, forcing=(; u=u_forcing))
    time_step!(model, 2)
    time_step!(model, 2)

    @test u_forcing.backend.start == 4

    return true
end

function relaxed_time_stepping(arch, mask_type)
    x_relax = Relaxation(rate = 1/60,   mask = mask_type{:x}(center=0.5, width=0.1),
                                      target = LinearTarget{:x}(intercept=π, gradient=ℯ))

    y_relax = Relaxation(rate = 1/60,   mask = mask_type{:y}(center=0.5, width=0.1),
                                      target = LinearTarget{:y}(intercept=π, gradient=ℯ))

    z_relax = Relaxation(rate = 1/60,   mask = mask_type{:z}(center=0.5, width=0.1),
                                      target = π)

    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(; grid, forcing=(u=x_relax, v=y_relax, w=z_relax))
    time_step!(model, 1)

    return true
end

function advective_and_multiple_forcing(grid; model_type=NonhydrostaticModel, immersed=false)

    if immersed
        zmin, zmax = znodes(grid, Face()) |> extrema
        bottom = (zmin + zmax) / 2
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
    end

    constant_slip = AdvectiveForcing(w=1)
    zero_slip = AdvectiveForcing(w=0)
    no_penetration = ImpenetrableBoundaryCondition()
    slip_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()), top=no_penetration, bottom=no_penetration)
    slip_velocity = ZFaceField(grid, boundary_conditions=slip_bcs)
    set!(slip_velocity, 1)
    velocity_field_slip = AdvectiveForcing(w=slip_velocity)
    zero_forcing(x, y, z, t) = 0
    one_forcing(x, y, z, t) = 1

    model = model_type(; grid,
                       timestepper = :QuasiAdamsBashforth2,
                       tracers = (:a, :b, :c),
                       forcing = (a = constant_slip,
                                  b = (zero_forcing, velocity_field_slip),
                                  c = (one_forcing, zero_slip)))

    noise(x, y, z) = rand()
    set!(model, a=noise, b=noise, c=0)
    a₀ = model.tracers.a |> deepcopy
    b₀ = model.tracers.b |> deepcopy

    # Time-step without an error?
    time_step!(model, 1, euler=true)

    a₁ = model.tracers.a
    b₁ = model.tracers.b
    c₁ = model.tracers.c

    a_changed = a₁ ≠ a₀
    b_changed = b₁ ≠ b₀
    effective_bottom = immersed ? (grid.Nz÷2 + 1) : 1
    c_correct = all(interior(c₁, :, :, effective_bottom:grid.Nz) .== model.clock.time)

    return a_changed & b_changed & c_correct
end

function two_forcings(arch)
    grid = RectilinearGrid(arch, size=(4, 5, 6), extent=(1, 1, 1), halo=(4, 4, 4))

    forcing1 = Relaxation(rate=1)
    forcing2 = Relaxation(rate=2)

    forcing = (u = (forcing1, forcing2),
               v = MultipleForcings(forcing1, forcing2),
               w = MultipleForcings((forcing1, forcing2)))

    model = NonhydrostaticModel(; grid, forcing)
    time_step!(model, 1)

    return true
end

function seven_forcings(arch)
    grid = RectilinearGrid(arch, size=(4, 5, 6), extent=(1, 1, 1), halo=(4, 4, 4))

    weird_forcing(x, y, z, t) = x * y + z
    wonky_forcing(x, y, z, t) = z / (x - y)
    strange_forcing(x, y, z, t) = z - t
    bizarre_forcing(x, y, z, t) = y + x
    peculiar_forcing(x, y, z, t) = 2t / z
    eccentric_forcing(x, y, z, t) = x + y + z + t
    unconventional_forcing(x, y, z, t) = 10x * y

    F1 = Forcing(weird_forcing)
    F2 = Forcing(wonky_forcing)
    F3 = Forcing(strange_forcing)
    F4 = Forcing(bizarre_forcing)
    F5 = Forcing(peculiar_forcing)
    F6 = Forcing(eccentric_forcing)
    F7 = Forcing(unconventional_forcing)

    Ft = (F1, F2, F3, F4, F5, F6, F7)
    forcing = (u=Ft, v=MultipleForcings(Ft...), w=MultipleForcings(Ft))
    model = NonhydrostaticModel(; grid, forcing)

    time_step!(model, 1)

    return true
end

function test_settling_tracer_comparison(arch)
    """
    Test that compares settling tracer simulations on regular vs immersed boundary grids.
    Both should conserve tracer mass and have similar maximum values.
    """

    Nz = 16
    Lz = 1

    regular_grid = RectilinearGrid(arch, topology = (Flat, Flat, Bounded), size = (Nz,), z = (-Lz, 0))
    immersed_grid = ImmersedBoundaryGrid(regular_grid, GridFittedBottom(-3Lz/4))

    function build_settling_model(grid, w_settle)
        # Create settling forcing
        settling_forcing = AdvectiveForcing(w = w_settle; grid, normal_boundary_condition=OpenBoundaryCondition(nothing))
        model = NonhydrostaticModel(; grid, advection=WENO(order=5), tracers = :c, forcing = (c = settling_forcing,))

        # Initial condition: patch of tracer c=1 in the upper part
        z_center = -Lz/4  # Upper quarter of domain
        z_width = Lz/8    # Width of initial patch
        c_initial(z) = abs(z - z_center) <= z_width ? 1.0 : 0.0
        set!(model, c = c_initial)
        return model
    end

    # Create models
    w_settle = -0.01
    regular_model = build_settling_model(regular_grid, w_settle)
    immersed_model = build_settling_model(immersed_grid, w_settle)

    regular_initial_integral = Integral(regular_model.tracers.c) |> Field |> deepcopy
    immersed_initial_integral = Integral(immersed_model.tracers.c) |> Field |> deepcopy
    @test regular_initial_integral[] == immersed_initial_integral[]

    # Create simulations
    Δt = abs(w_settle) / minimum_zspacing(regular_grid)
    stop_time = 250
    regular_simulation = Simulation(regular_model, Δt=Δt, stop_time=stop_time)
    immersed_simulation = Simulation(immersed_model, Δt=Δt, stop_time=stop_time)

    # Run simulations
    run!(regular_simulation)
    run!(immersed_simulation)

    # Compute diagnostics
    regular_integral = Integral(regular_model.tracers.c) |> Field
    immersed_integral = Integral(immersed_model.tracers.c) |> Field

    regular_max = maximum(abs, regular_model.tracers.c)
    immersed_max = maximum(abs, immersed_model.tracers.c)

    # Test that mass is approximately conserved and max values are similar
    @test regular_integral[] ≈ immersed_integral[] ≈ regular_initial_integral[] == immersed_initial_integral[]
    @test regular_max ≈ immersed_max rtol=0.01

    return true
end

@testset "Forcings" begin
    @info "Testing forcings..."

    for arch in archs
        A = typeof(arch)
        @testset "Forcing function time stepping [$A]" begin
            @info "  Testing forcing function time stepping [$A]..."

            @testset "Non-parameterized forcing functions [$A]" begin
                @info "      Testing non-parameterized forcing functions [$A]..."
                @test time_step_with_forcing_functions(arch)
                @test time_step_with_forcing_array(arch)
                @test time_step_with_discrete_forcing(arch)
            end

            @testset "Parameterized forcing functions [$A]" begin
                @info "      Testing parameterized forcing functions [$A]..."
                @test time_step_with_parameterized_continuous_forcing(arch)
                @test time_step_with_parameterized_discrete_forcing(arch)
            end

            @testset "Field-dependent forcing functions [$A]" begin
                @info "      Testing field-dependent forcing functions [$A]..."

                for fld in (:u, :v, :w, :T, :A)
                    @test time_step_with_single_field_dependent_forcing(arch, fld)
                end

                @test time_step_with_multiple_field_dependent_forcing(arch)
                @test time_step_with_parameterized_field_dependent_forcing(arch)
            end

            @testset "Relaxation forcing functions [$A]" begin
                @info "      Testing relaxation forcing functions [$A]..."
                @test relaxed_time_stepping(arch, GaussianMask)
                @test relaxed_time_stepping(arch, PiecewiseLinearMask)
            end

            @testset "Advective and multiple forcing [$A]" begin
                @info "      Testing advective and multiple forcing [$A]..."
                rectilinear_grid = RectilinearGrid(arch, size=(4, 5, 6), extent=(1, 1, 1), halo=(4, 4, 4))
                latlon_grid = LatitudeLongitudeGrid(arch, size=(4, 5, 6), longitude=(-180, 180), latitude=(-85, 85), z=(-1, 0), halo=(4, 4, 4))

                for grid in (rectilinear_grid, latlon_grid), model_type in (NonhydrostaticModel, HydrostaticFreeSurfaceModel), immersed in (false, true)
                    @test advective_and_multiple_forcing(grid; model_type=model_type, immersed=immersed)
                end

                @test two_forcings(arch)
                @test seven_forcings(arch)
            end

            @testset "FieldTimeSeries forcing on [$A]" begin
                @info "      Testing FieldTimeSeries forcing [$A]..."
                @test time_step_with_field_time_series_forcing(arch)
            end

            @testset "Settling tracer comparison [$A]" begin
                @info "      Testing settling tracer on regular vs immersed grids [$A]..."
                @test test_settling_tracer_comparison(arch)
            end
        end
    end
end
