include("dependencies_for_runtests.jl")

using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Fields: Field
using Oceananigans.Forcings: MultipleForcings, FieldRelaxation, FieldTimeSeriesRelaxation, InterpolatedFieldTarget
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, immersed_peripheral_node, peripheral_node

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

function relaxed_time_stepping(arch, mask_type; mask_kwargs...)
    x_relax = Relaxation(rate = 1/60,   mask = mask_type{:x}(; mask_kwargs...),
                                      target = LinearTarget{:x}(intercept=π, gradient=ℯ))

    y_relax = Relaxation(rate = 1/60,   mask = mask_type{:y}(; mask_kwargs...),
                                      target = LinearTarget{:y}(intercept=π, gradient=ℯ))

    z_relax = Relaxation(rate = 1/60,   mask = mask_type{:z}(; mask_kwargs...),
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

# For each node (i, j, k), store whether the condition
# "if immersed peripheral node then flux == 0" holds.
# That is: !immersed_peripheral_node | (flux == 0).
# @test all(interior(t**)) then verifies zero flux everywhere it is required.
@kernel function _populate_momentum_flux_tests!(tuu, tuv, tuw, tvu, tvv, tvw, twu, twv, tww,
                                                grid, scheme, u, v, w)
    i, j, k = @index(Global, NTuple)

    Fuu = Oceananigans.Advection._advective_momentum_flux_Uu(i, j, k, grid, scheme, u, u)
    Fuv = Oceananigans.Advection._advective_momentum_flux_Uv(i, j, k, grid, scheme, u, v)
    Fuw = Oceananigans.Advection._advective_momentum_flux_Uw(i, j, k, grid, scheme, u, w)
    Fvu = Oceananigans.Advection._advective_momentum_flux_Vu(i, j, k, grid, scheme, v, u)
    Fvv = Oceananigans.Advection._advective_momentum_flux_Vv(i, j, k, grid, scheme, v, v)
    Fvw = Oceananigans.Advection._advective_momentum_flux_Vw(i, j, k, grid, scheme, v, w)
    Fwu = Oceananigans.Advection._advective_momentum_flux_Wu(i, j, k, grid, scheme, w, u)
    Fwv = Oceananigans.Advection._advective_momentum_flux_Wv(i, j, k, grid, scheme, w, v)
    Fww = Oceananigans.Advection._advective_momentum_flux_Ww(i, j, k, grid, scheme, w, w)

    c = Center()
    f = Face()

    @inbounds begin
        tuu[i, j, k] = !peripheral_node(i, j, k, grid, c, c, c) | (Fuu == 0)
        tuv[i, j, k] = !peripheral_node(i, j, k, grid, f, f, c) | (Fuv == 0)
        tuw[i, j, k] = !peripheral_node(i, j, k, grid, f, c, f) | (Fuw == 0)
        tvu[i, j, k] = !peripheral_node(i, j, k, grid, f, f, c) | (Fvu == 0)
        tvv[i, j, k] = !peripheral_node(i, j, k, grid, c, c, c) | (Fvv == 0)
        tvw[i, j, k] = !peripheral_node(i, j, k, grid, c, f, f) | (Fvw == 0)
        twu[i, j, k] = !peripheral_node(i, j, k, grid, f, c, f) | (Fwu == 0)
        twv[i, j, k] = !peripheral_node(i, j, k, grid, c, f, f) | (Fwv == 0)
        tww[i, j, k] = !peripheral_node(i, j, k, grid, c, c, c) | (Fww == 0)
    end
end

"""
Test that all 9 advective momentum flux kernels are zero at every immersed peripheral node,
for a grid with a random bottom boundary. This verifies that removing the explicit
`conditional_flux` zeroing from `_advective_momentum_flux_*` on `ImmersedBoundaryGrid` is
safe: velocity masking in immersed cells and the no-penetration boundary condition at the
immersed boundary face are sufficient to guarantee zero momentum flux at all peripheral
nodes, independently of the velocity magnitude in the active fluid region.
"""
function test_momentum_flux_zero_at_peripheral_nodes(scheme)
    Nx, Ny, Nz = 8, 8, 8
    underlying_grid = RectilinearGrid(CPU(), size=(Nx, Ny, Nz), extent=(1, 1, 1), halo=(4, 4, 4))

    # Random bottom creates a varied immersed boundary topology, stressing all peripheral
    # node configurations (not just a uniform flat slab).
    bottom_height = -1 .+ rand(Nx, Ny) .* 0.8  # varies between -1 and -0.2
    ibg = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
    model = NonhydrostaticModel(ibg, advection=scheme)

    # Set non-zero velocities everywhere, then zero out immersed cells via masking.
    set!(model, u=1, v=1, w=1)
    mask_immersed_field!(model.velocities.u)
    mask_immersed_field!(model.velocities.v)
    mask_immersed_field!(model.velocities.w)
    fill_halo_regions!(model.velocities)

    u, v, w = model.velocities

    # Test fields: store true (1) where the "zero-flux at peripheral node" property holds,
    # false (0) where it is violated.
    tuu = Field{Center, Center, Center}(ibg)
    tuv = Field{Face, Face, Center}(ibg)
    tuw = Field{Face, Center, Face}(ibg)
    tvu = Field{Face, Face, Center}(ibg)
    tvv = Field{Center, Center, Center}(ibg)
    tvw = Field{Center, Face, Face}(ibg)
    twu = Field{Face, Center, Face}(ibg)
    twv = Field{Center, Face, Face}(ibg)
    tww = Field{Center, Center, Center}(ibg)

    launch!(CPU(), ibg,  KernelParameters(0:9, 0:9, 0:9), _populate_momentum_flux_tests!,
            tuu, tuv, tuw, tvu, tvv, tvw, twu, twv, tww,
            ibg, scheme, u, v, w)

    @test all(!iszero, Array(interior(tuu)))
    @test all(!iszero, Array(interior(tuv)))
    @test all(!iszero, Array(interior(tuw)))
    @test all(!iszero, Array(interior(tvu)))
    @test all(!iszero, Array(interior(tvv)))
    @test all(!iszero, Array(interior(tvw)))
    @test all(!iszero, Array(interior(twu)))
    @test all(!iszero, Array(interior(twv)))
    @test all(!iszero, Array(interior(tww)))

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
        bottom_boundary_conditions = open_bottom ? NormalFlowBoundaryCondition(w_settle) : NormalFlowBoundaryCondition(nothing)
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

""" Build a time-invariant FTS where each snapshot equals `f(x, y, z)`. Used to
isolate the spatial-interpolation path: temporal interpolation collapses to a
constant since all snapshots are identical.

`fill_halo_regions!` is required after `set!` because sim cells outside the FTS
*Center* coverage (the boundary sim cells when FTS is coarser than sim) sample
halo cells during interpolation. Without halo-filling, halos are zero-initialized
and bias the interpolation toward zero. """
function spatial_test_fts(grid, times, f::Function)
    fts = FieldTimeSeries{Center, Center, Center}(grid, times)
    for n in eachindex(times)
        set!(fts[n], f)
        fill_halo_regions!(fts[n])
    end
    return fts
end

""" Build a spatially-uniform FTS where snapshot `n` is the scalar `g(times[n])`.
Used with an FTS grid that matches the simulation grid to isolate the
temporal-interpolation path: spatial interpolation collapses to identity. """
function temporal_test_fts(grid, times, g::Function)
    fts = FieldTimeSeries{Center, Center, Center}(grid, times)
    for n in eachindex(times)
        set!(fts[n], g(times[n]))
        fill_halo_regions!(fts[n])
    end
    return fts
end

@testset "Forcings" begin
    @info "Testing forcings..."

    @testset "CosineRampMask cosine ramp" begin
        @info "  Testing CosineRampMask cosine ramp..."

        for (D, eval_at) in ((:x, (m, ξ) -> m(ξ, 0, 0)),
                             (:y, (m, ξ) -> m(0, ξ, 0)),
                             (:z, (m, ξ) -> m(0, 0, ξ)))

            m = CosineRampMask{D}(start=1500.0, stop=2500.0)

            @test eval_at(m, 1400.0) == 0
            @test eval_at(m, 1500.0) == 0
            @test eval_at(m, 2500.0) ≈ 1
            @test eval_at(m, 2600.0) ≈ 1
            @test eval_at(m, 2000.0) ≈ 0.5

            r₁ = eval_at(m, 1750.0)
            r₃ = eval_at(m, 2250.0)
            @test r₁ + r₃ ≈ 1
            @test 0 < r₁ < 0.5 < r₃ < 1

            weights = [eval_at(m, ξ) for ξ in range(1500, 2500, length=11)]
            @test all(diff(weights) .> 0)

            m_rev = CosineRampMask{D}(start=2500.0, stop=1500.0)
            @test eval_at(m_rev, 1500.0) ≈ 1
            @test eval_at(m_rev, 2500.0) == 0
            @test eval_at(m_rev, 2000.0) ≈ 0.5
        end

        @test_throws ArgumentError CosineRampMask{:z}(start=1500, stop=1500)
    end

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
                @test relaxed_time_stepping(arch, GaussianMask;        center=0.5, width=0.1)
                @test relaxed_time_stepping(arch, PiecewiseLinearMask; center=0.5, width=0.1)
                @test relaxed_time_stepping(arch, CosineRampMask;      start=0.4, stop=0.6)
            end

            @testset "Relaxation with FieldTimeSeries target [$A]" begin
                @info "      Testing Relaxation with FieldTimeSeries target [$A]..."

                grid = RectilinearGrid(arch, size=(2, 2, 4), extent=(100, 100, 1000))
                τ     = 60
                c_ref = 5

                fts = FieldTimeSeries{Center, Center, Center}(grid, [0, 1e6])
                for n in eachindex(fts.times)
                    set!(fts[n], c_ref)
                end

                r = Relaxation(rate=1/τ, target=fts)
                model = NonhydrostaticModel(grid; tracers=:c, forcing=(; c=r))

                # Materialization wraps the FTS in a FieldTimeSeriesTarget so the FTS's grid
                # survives Adapt to the device, and records the forced field's location.
                rm = model.forcing.c
                @test rm isa FieldTimeSeriesRelaxation
                @test rm.target.field_time_series === fts
                @test rm.target.grid === fts.grid
                @test rm.relaxed === model.tracers.c
                @test rm.location == (Center(), Center(), Center())

                # Analytical convergence: dc/dt = (c_ref - c)/τ ⇒ after one step,
                # c ≈ c_ref * (1 - exp(-Δt/τ)) for c(0) = 0.
                set!(model, c=0)
                Δt = 1
                time_step!(model, Δt)
                c_after  = Array(interior(model.tracers.c))
                expected = c_ref * (1 - exp(-Δt/τ))
                @test all(isapprox.(c_after, expected; atol=1e-6 * c_ref))

                # Extent validation: FTS strictly smaller than the simulation grid throws.
                small_grid = RectilinearGrid(arch, size=(2, 2, 4), extent=(50, 50, 500))
                fts_small  = FieldTimeSeries{Center, Center, Center}(small_grid, [0, 1e6])
                r_small    = Relaxation(rate=1/τ, target=fts_small)
                @test_throws ArgumentError NonhydrostaticModel(grid; tracers=:c, forcing=(; c=r_small))
            end

            @testset "Relaxation with Field target [$A]" begin
                @info "      Testing Relaxation with Field target [$A]..."

                grid  = RectilinearGrid(arch, size=(2, 2, 4), extent=(100, 100, 1000))
                τ     = 60
                c_ref = 5

                target_field = CenterField(grid)
                set!(target_field, c_ref)

                r     = Relaxation(rate=1/τ, target=target_field)
                model = NonhydrostaticModel(grid; tracers=:c, forcing=(; c=r))

                rm = model.forcing.c
                @test rm isa FieldRelaxation
                @test !(rm isa FieldTimeSeriesRelaxation)
                @test rm.target === target_field
                @test rm.relaxed === model.tracers.c
                @test rm.location == (Center(), Center(), Center())

                set!(model, c=0)
                Δt = 1
                time_step!(model, Δt)
                c_after  = Array(interior(model.tracers.c))
                expected = c_ref * (1 - exp(-Δt/τ))
                @test all(isapprox.(c_after, expected; atol=1e-6 * c_ref))

                # Location mismatch on the same grid: target is auto-wrapped for spatial
                # interpolation so the (Face, Center, Center) staggering is sampled at the
                # tracer's (Center, Center, Center) nodes. A uniform target still drives
                # the tracer toward `c_ref`.
                wrong_loc = XFaceField(grid)
                set!(wrong_loc, c_ref)
                fill_halo_regions!(wrong_loc)
                r_loc = Relaxation(rate=1/τ, target=wrong_loc)
                model_loc = NonhydrostaticModel(grid; tracers=:c, forcing=(; c=r_loc))
                @test model_loc.forcing.c.target isa InterpolatedFieldTarget
                set!(model_loc, c=0)
                time_step!(model_loc, Δt)
                @test all(isapprox.(Array(interior(model_loc.tracers.c)), expected; atol=1e-6 * c_ref))

                # Grid mismatch: target lives on a different grid that fully contains the
                # simulation grid and is auto-wrapped for interpolation. A uniform target
                # of value `c_ref` should still drive the tracer toward `c_ref` after one
                # step, just as the same-grid case does.
                other_grid   = RectilinearGrid(arch, size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(-1000, 0),
                                                topology=(Periodic, Periodic, Bounded))
                wrong_grid   = CenterField(other_grid)
                set!(wrong_grid, c_ref)
                fill_halo_regions!(wrong_grid)
                r_grid = Relaxation(rate=1/τ, target=wrong_grid)
                model_grid = NonhydrostaticModel(grid; tracers=:c, forcing=(; c=r_grid))
                @test model_grid.forcing.c.target isa InterpolatedFieldTarget
                set!(model_grid, c=0)
                time_step!(model_grid, Δt)
                c_grid_after = Array(interior(model_grid.tracers.c))
                @test all(isapprox.(c_grid_after, expected; atol=1e-6 * c_ref))
            end

            @testset "Relaxation with transform=:horizontal_average [$A]" begin
                @info "      Testing Relaxation with transform=:horizontal_average [$A]..."

                Nx, Ny, Nz = 4, 4, 2
                Lx, Ly, Lz = 100.0, 100.0, 100.0
                grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
                τ = 60.0
                c_target = 5.0

                # transform produces the LHS (horizontal average); the user-supplied
                # `target` is the RHS the average is pulled toward.
                r = Relaxation(rate=1/τ, transform=:horizontal_average, target=c_target)
                model = NonhydrostaticModel(grid; tracers=:c, forcing=(; c=r))

                rm = model.forcing.c
                @test rm.transform === :horizontal_average
                @test rm.target === c_target
                @test rm.relaxed isa Field
                @test rm.relaxed !== model.tracers.c

                # IC: zero-mean horizontal sinusoid. The forcing is i,j-independent
                # at fixed k, so it drives only the horizontal mean — fluctuations
                # in (x, y) are preserved.
                set!(model, c=(x, y, z) -> sin(2π * x / Lx))
                c_initial = Array(interior(model.tracers.c))
                fluctuation_initial = c_initial .- mean(c_initial, dims=(1, 2))

                Δt = 0.5
                Nsteps = 20
                for _ in 1:Nsteps
                    time_step!(model, Δt)
                end
                t_end = model.clock.time

                c_final = Array(interior(model.tracers.c))
                mean_final = mean(c_final, dims=(1, 2))
                fluctuation_final = c_final .- mean_final

                # d<c>/dt = (c_target - <c>)/τ with <c>(0) = 0  ⇒  <c>(t) = c_target (1 - exp(-t/τ))
                expected_mean = c_target * (1 - exp(-t_end / τ))
                @test all(isapprox.(mean_final, expected_mean; atol=1e-3 * c_target))
                @test all(isapprox.(fluctuation_final, fluctuation_initial; atol=1e-6))

                # Closure form is equivalent to the :horizontal_average symbol.
                r_closure = Relaxation(rate=1/τ, target=c_target,
                                       transform = f -> Field(Average(f, dims=(1, 2))))
                model_closure = NonhydrostaticModel(grid; tracers=:c, forcing=(; c=r_closure))
                @test model_closure.forcing.c.relaxed isa Field

                # z-varying target profile: <c>(z) relaxes toward LinearTarget at each k.
                c_profile = LinearTarget{:z}(intercept=1.0, gradient=0.01)
                r_profile = Relaxation(rate=1/τ, transform=:horizontal_average, target=c_profile)
                model_profile = NonhydrostaticModel(grid; tracers=:c, forcing=(; c=r_profile))
                set!(model_profile, c=0)
                for _ in 1:Nsteps
                    time_step!(model_profile, Δt)
                end
                t_p = model_profile.clock.time
                c_p = Array(interior(model_profile.tracers.c))
                mean_p = dropdims(mean(c_p, dims=(1, 2)), dims=(1, 2))
                for k in 1:Nz
                    _, _, z = node(1, 1, k, grid, Center(), Center(), Center())
                    expected_k = c_profile(0, 0, z, 0) * (1 - exp(-t_p / τ))
                    @test isapprox(mean_p[k], expected_k; atol=1e-3)
                end

                # Mixing transform with FieldTimeSeries target should error.
                fts = FieldTimeSeries{Center, Center, Center}(grid, [0, 1e6])
                @test_throws ArgumentError NonhydrostaticModel(grid; tracers=:c,
                    forcing=(; c=Relaxation(rate=1/τ, target=fts, transform=:horizontal_average)))
            end

            @testset "Relaxation FTS-target cross-grid spatial interp [$A]" begin
                @info "      Testing Relaxation FTS-target cross-grid spatial interp [$A]..."

                Lx, Ly, Lz = 1000.0, 1000.0, 100.0
                # Bounded topology so Face-node extrema reach the domain edges
                # exactly, matching the physical setting for Davies-style fringes.
                topology = (Bounded, Bounded, Bounded)
                # Coarse-FTS / fine-sim is the primary use case (reanalysis FTS
                # driving an LES interior). The FTS x and y must be **padded**
                # so the FTS Centers bracket the sim Centers — otherwise the
                # sim boundary cells fall outside FTS Center coverage and
                # trilinear interpolation reads zero-initialized FTS halos.
                # The validator `validate_fts_target_extent` checks this at
                # materialize time; the calculation below leaves a comfortable
                # margin past the minimum δ ≈ 53.6 m required at this resolution.
                δ_pad = 100.0
                sim_grid = RectilinearGrid(arch, size=(32, 32, 4), extent=(Lx, Ly, Lz), topology=topology)
                fts_grid = RectilinearGrid(arch, size=( 8,  8, 4),
                                           x=(-δ_pad, Lx + δ_pad),
                                           y=(-δ_pad, Ly + δ_pad),
                                           z=(-Lz, 0),               # match ocean-facing z of sim_grid
                                           topology=topology)

                f(x, y, z) = x                                    # affine → trilinear-exact
                fts = spatial_test_fts(fts_grid, [0.0, 1.0], f)

                Δ_fringe = Lx / 4
                # NOTE: single-sided fringe; switch to MaximumMask of west + east
                # CosineRampMask{:x} once #5576 lands for a true Davies two-sided fringe.
                mask = CosineRampMask{:x}(start=Δ_fringe, stop=0)  # full at x=0, off at x=Δ_fringe

                τ  = 100.0
                Δt = 0.1                                          # w·Δt/τ ≤ 1e-3 → tight linearization
                forcing = Relaxation(rate=1/τ, mask=mask, target=fts)
                model = NonhydrostaticModel(sim_grid; tracers=:c, forcing=(; c=forcing))
                # model.tracers.c is zero-initialized by NonhydrostaticModel.
                time_step!(model, Δt)

                c = Array(interior(model.tracers.c))

                interior_max_abs = 0.0
                fringe_count = 0
                for k in 1:size(sim_grid, 3), j in 1:size(sim_grid, 2), i in 1:size(sim_grid, 1)
                    x, y, z = node(i, j, k, sim_grid, Center(), Center(), Center())
                    w = mask(x, y, z)
                    if w ≈ 0
                        interior_max_abs = max(interior_max_abs, abs(c[i, j, k]))
                    else
                        fringe_count += 1
                        # Exact one-step solution of dc/dt = (w/τ)·(f(x,y,z) - c), c(0)=0:
                        # c(Δt) = f(x,y,z) · (1 - exp(-w·Δt/τ))
                        expected = f(x, y, z) * (1 - exp(-w * Δt / τ))
                        @test c[i, j, k] ≈ expected rtol=1e-3
                    end
                end
                @test interior_max_abs < 1e-12                    # no leakage into interior
                @test fringe_count > 0                            # sanity: mask is non-trivial
            end

            @testset "Relaxation FTS-target temporal interp [$A]" begin
                @info "      Testing Relaxation FTS-target temporal interp [$A]..."

                Lx, Ly, Lz = 1000.0, 1000.0, 100.0
                # Bounded x mirrors the spatial-interp test for consistency.
                topology = (Bounded, Bounded, Bounded)
                grid = RectilinearGrid(arch, size=(8, 8, 4), extent=(Lx, Ly, Lz), topology=topology)

                g(t) = t                                          # affine in t → linear-interp exact
                times = [0.0, 1.0, 2.0]
                fts = temporal_test_fts(grid, times, g)           # FTS on sim grid → identity spatial

                Δ_fringe = Lx / 4
                mask = CosineRampMask{:x}(start=Δ_fringe, stop=0)

                τ  = 1.0
                Δt = 0.05
                Nsteps = 10                                       # step to t = 0.5, well inside [0, 1]
                forcing = Relaxation(rate=1/τ, mask=mask, target=fts)
                model = NonhydrostaticModel(grid; tracers=:c, forcing=(; c=forcing))
                for _ in 1:Nsteps
                    time_step!(model, Δt)
                end
                t_end = model.clock.time

                c = Array(interior(model.tracers.c))

                interior_max_abs = 0.0
                fringe_count = 0
                for k in 1:size(grid, 3), j in 1:size(grid, 2), i in 1:size(grid, 1)
                    x, y, z = node(i, j, k, grid, Center(), Center(), Center())
                    w = mask(x, y, z)
                    if w ≈ 0
                        interior_max_abs = max(interior_max_abs, abs(c[i, j, k]))
                    else
                        fringe_count += 1
                        # Exact solution of dc/dt = (w/τ)·(t - c), c(0)=0:
                        # c(t) = t - τ_eff + τ_eff · exp(-t/τ_eff),  τ_eff = τ/w
                        τ_eff = τ / w
                        expected = t_end - τ_eff + τ_eff * exp(-t_end / τ_eff)
                        @test c[i, j, k] ≈ expected rtol=1e-3
                    end
                end
                @test interior_max_abs < 1e-12
                @test fringe_count > 0
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
                    scheme = Oceananigans.Advection.materialize_advection(scheme, MockGrid(arch))
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
