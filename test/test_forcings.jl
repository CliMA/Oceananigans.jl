include("dependencies_for_runtests.jl")

using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Fields: Field
using Oceananigans.Forcings: MultipleForcings
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

""" Take one time step with three forcing arrays on u, v, w. """
function time_step_with_forcing_array(arch)
    grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 1, 1))

    Fu = XFaceField(grid)
    Fv = YFaceField(grid)
    Fw = ZFaceField(grid)

    set!(Fu, (x, y, z) -> 1)
    set!(Fv, (x, y, z) -> 1)
    set!(Fw, (x, y, z) -> 1)

    model = NonhydrostaticModel(grid; forcing=(u=Fu, v=Fv, w=Fw))
    time_step!(model, 1)

    return true
end

""" Take one time step with three forcing functions on u, v, w. """
function time_step_with_forcing_functions(arch)
    @inline Fu(x, y, z, t) = exp(π * z)
    @inline Fv(x, y, z, t) = cos(42 * x)
    @inline Fw(x, y, z, t) = 1.0

    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid; forcing=(u=Fu, v=Fv, w=Fw))
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
    model = NonhydrostaticModel(grid; forcing=(; u=Fu))
    time_step!(model, 1)

    return true
end

""" Take one time step with ParameterizedForcing forcing functions. """
function time_step_with_parameterized_discrete_forcing(arch)

    Fv = Forcing(Fv_discrete_func, parameters=(; τ=60), discrete_form=true)
    Fw = Forcing(Fw_discrete_func, parameters=(; τ=60), discrete_form=true)

    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid; forcing=(v=Fv, w=Fw))
    time_step!(model, 1)

    return true
end

""" Take one time step with a Forcing forcing function with parameters. """
function time_step_with_parameterized_continuous_forcing(arch)
    Fu = Forcing((x, y, z, t, ω) -> sin(ω * x), parameters=π)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid; forcing=(; u=Fu))
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
    model = NonhydrostaticModel(grid; forcing,
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
    model = NonhydrostaticModel(grid;
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
    model = NonhydrostaticModel(grid; forcing=(; u=Fu))
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

    model = NonhydrostaticModel(grid; forcing=(; u=u_forcing))
    time_step!(model, 1)

    # Make sure the field time series updates correctly
    u_forcing = FieldTimeSeries{Face, Center, Center}(grid, 0:1:4; backend = InMemory(2))

    model = NonhydrostaticModel(grid; forcing=(; u=u_forcing))
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
    model = NonhydrostaticModel(grid; forcing=(u=x_relax, v=y_relax, w=z_relax))
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

    model = model_type(grid;
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

    model = NonhydrostaticModel(grid; forcing)
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
    model = NonhydrostaticModel(grid; forcing)

    time_step!(model, 1)

    return true
end

"""
Test that momentum advective fluxes are zero at immersed peripheral nodes, for all
advection schemes. This verifies that removing the explicit `conditional_flux` zeroing
from `_advective_momentum_flux_*` on `ImmersedBoundaryGrid` is safe: the combination of
velocity masking in immersed cells and the no-penetration boundary condition at the
immersed boundary face is sufficient to ensure zero momentum flux at peripheral nodes,
independently of the velocity magnitude in the active fluid region.

Tests on CPU only since the flux functions are called with scalar indices.
"""
function test_momentum_flux_zero_at_peripheral_nodes(scheme)
    Nz = 8
    grid = RectilinearGrid(CPU(), size=(4, 4, Nz), extent=(1, 1, 1), halo=(4, 4, 4))

    # Flat bottom at z = -0.5: cells k=1..Nz÷2 are immersed, k=Nz÷2+1..Nz are active.
    ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(-0.5))
    model = NonhydrostaticModel(ibg, advection=scheme)

    # Set non-zero velocities in the entire domain, then mask to zero in immersed region.
    # This ensures the active region has non-trivial velocities (u=v=w=1 in active cells)
    # while immersed cells and the immersed boundary face are zeroed out.
    set!(model, u=1, v=1, w=1)
    mask_immersed_field!(model.velocities.u)
    mask_immersed_field!(model.velocities.v)
    mask_immersed_field!(model.velocities.w)
    fill_halo_regions!(model.velocities)

    u, v, w = model.velocities

    # k_immersed: index of the last immersed cell center.
    # Peripheral nodes for (Center, Center, Center) and (Face, Face, Center) are at immersed cells.
    # Velocities there are zero by masking, so fluxes must be zero.
    k_immersed = Nz ÷ 2

    # k_face: index of the immersed boundary face (ZFace between immersed k and active k+1).
    # Peripheral nodes for (Face, Center, Face) and (Center, Face, Face) are at this face.
    # w = 0 at this face (no-penetration + masking), so z-momentum fluxes must be zero.
    k_face = Nz ÷ 2 + 1

    i, j = 2, 2  # Interior horizontal position, away from domain boundaries

    # Fluxes at (Center, Center, Center) peripheral nodes — immersed cells, all velocities masked
    @test Oceananigans.Advection._advective_momentum_flux_Uu(i, j, k_immersed, ibg, scheme, u, u) == 0
    @test Oceananigans.Advection._advective_momentum_flux_Vv(i, j, k_immersed, ibg, scheme, v, v) == 0
    @test Oceananigans.Advection._advective_momentum_flux_Ww(i, j, k_immersed, ibg, scheme, w, w) == 0

    # Fluxes at (Face, Face, Center) peripheral nodes — immersed cells, all velocities masked
    @test Oceananigans.Advection._advective_momentum_flux_Vu(i, j, k_immersed, ibg, scheme, v, u) == 0
    @test Oceananigans.Advection._advective_momentum_flux_Uv(i, j, k_immersed, ibg, scheme, u, v) == 0

    # Fluxes at (Face, Center, Face) peripheral nodes — immersed boundary face, w = 0
    @test Oceananigans.Advection._advective_momentum_flux_Wu(i, j, k_face, ibg, scheme, w, u) == 0
    @test Oceananigans.Advection._advective_momentum_flux_Uw(i, j, k_face, ibg, scheme, u, w) == 0

    # Fluxes at (Center, Face, Face) peripheral nodes — immersed boundary face, w = 0
    @test Oceananigans.Advection._advective_momentum_flux_Wv(i, j, k_face, ibg, scheme, w, v) == 0
    @test Oceananigans.Advection._advective_momentum_flux_Vw(i, j, k_face, ibg, scheme, v, w) == 0

    return true
end

function test_settling_tracer_comparison(arch; open_bottom=true)
    """
    Test that compares settling tracer simulations on regular vs immersed boundary grids.
    Both should conserve tracer mass and have similar maximum values.
    """

    Nz = 16
    Lz = 1

    regular_grid = RectilinearGrid(arch, topology = (Flat, Flat, Bounded), size = Nz, z = (-Lz, 0))
    immersed_grid = ImmersedBoundaryGrid(regular_grid, GridFittedBottom(-3Lz/4))

    function build_settling_model(grid, w_settle)
        # Create settling velocity as a field with appropriate boundary conditions
        bottom_boundary_conditions = open_bottom ? OpenBoundaryCondition(w_settle) : OpenBoundaryCondition(nothing)
        boundary_conditions = FieldBoundaryConditions(grid, (Center(), Center(), Face()), bottom = bottom_boundary_conditions)
        w_settle_field = ZFaceField(grid; boundary_conditions)

        # Set the velocity and apply boundary conditions to domain boundaries
        set!(w_settle_field, w_settle)
        fill_halo_regions!(w_settle_field)

        # Apply boundary condition to immersed boundaries
        if open_bottom
            mask_immersed_field!(w_settle_field, w_settle)
        else
            mask_immersed_field!(w_settle_field, 0)
        end

        # Create settling forcing with the velocity field
        settling_forcing = AdvectiveForcing(w = w_settle_field)
        model = NonhydrostaticModel(grid; advection=WENO(order=5), tracers = :c, forcing = (c = settling_forcing,))

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

    ∫c_regular = Integral(regular_model.tracers.c) |> Field
    ∫c_immersed = Integral(immersed_model.tracers.c) |> Field

    regular_initial_integral = ∫c_regular |> deepcopy
    immersed_initial_integral = ∫c_immersed |> deepcopy
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
    regular_integral = ∫c_regular |> compute!
    immersed_integral = ∫c_immersed |> compute!

    regular_max = maximum(abs, regular_model.tracers.c)
    immersed_max = maximum(abs, immersed_model.tracers.c)

    # Test that mass is approximately conserved and max values are similar
    @test regular_initial_integral[] == immersed_initial_integral[]
    if open_bottom
        @test (regular_integral[] / regular_initial_integral[]) < 1e-3
        @test (immersed_integral[] / immersed_initial_integral[]) < 1e-3
    else
        # Mass is approximately conserved, with some numerical diffusion
        @test isapprox(regular_integral[], regular_initial_integral[], rtol=1e-3)
        @test isapprox(immersed_integral[], immersed_initial_integral[], rtol=1e-3)
    end

    return true
end

"""
Verify that ContinuousForcing with field_dependencies produces the same
tendency as an equivalent DiscreteForcing on HydrostaticFreeSurfaceModel.

Regression test for a bug where the model_fields NamedTuple used during
forcing materialization excluded w, causing field_dependencies_indices
to be off by one (e.g. reading w instead of T).
"""
function test_hydrostatic_continuous_discrete_forcing_consistency(arch)
    grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))

    continuous_T_forcing = Forcing((x, y, z, t, T) -> -T, field_dependencies=:T)
    @inline discrete_T_forcing_func(i, j, k, grid, clock, model_fields) =
        @inbounds -model_fields.T[i, j, k]
    discrete_T_forcing = Forcing(discrete_T_forcing_func, discrete_form=true)

    model_c = HydrostaticFreeSurfaceModel(grid; forcing=(; T=continuous_T_forcing),
                                          tracers=:T, buoyancy=nothing)
    model_d = HydrostaticFreeSurfaceModel(grid; forcing=(; T=discrete_T_forcing),
                                          tracers=:T, buoyancy=nothing)

    set!(model_c, T=3.0)
    set!(model_d, T=3.0)

    time_step!(model_c, 1)
    time_step!(model_d, 1)

    Gc = Array(interior(model_c.timestepper.Gⁿ.T))
    Gd = Array(interior(model_d.timestepper.Gⁿ.T))

    return all(Gc .≈ Gd) && all(Gc .≈ -3)
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

            @testset "HydrostaticFreeSurfaceModel continuous/discrete forcing consistency [$A]" begin
                @info "      Testing hydrostatic continuous/discrete forcing consistency [$A]..."
                @test test_hydrostatic_continuous_discrete_forcing_consistency(arch)
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

            @testset "Momentum flux zero at immersed peripheral nodes" begin
                @info "      Testing momentum flux is zero at immersed peripheral nodes..."
                for scheme in (Centered(order=2), UpwindBiased(order=3), WENO(order=5))
                    @test test_momentum_flux_zero_at_peripheral_nodes(scheme)
                end
            end

            @testset "FieldTimeSeries forcing on [$A]" begin
                @info "      Testing FieldTimeSeries forcing [$A]..."
                @test time_step_with_field_time_series_forcing(arch)
            end

            @testset "Settling tracer comparison [$A]" begin
                @info "      Testing settling tracer on regular vs immersed grids [$A]..."
                @test test_settling_tracer_comparison(arch, open_bottom=true)
                @test test_settling_tracer_comparison(arch, open_bottom=false)
            end
        end
    end
end
