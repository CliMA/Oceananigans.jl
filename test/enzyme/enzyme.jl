include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Enzyme
using Oceananigans.TimeSteppers: reset!

# Required presently
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

# OceananigansLogger doesn't work here -- not sure why
Logging.global_logger(TestLogger())

f(grid) = CenterField(grid)
const maximum_diffusivity = 100

"""
    set_diffusivity!(model, diffusivity)

Change diffusivity of model to `diffusivity`.
"""
function set_diffusivity!(model, diffusivity)
    closure = VerticalScalarDiffusivity(; κ=diffusivity)
    names = tuple(:c) # tracernames(model.tracers)
    closure = with_tracers(names, closure)
    model.closure = closure
    return nothing
end

function set_initial_condition!(model, amplitude)
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    cᵢ(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    set!(model, c=cᵢ)

    return nothing
end

function stable_diffusion!(model, amplitude, diffusivity)
    reset!(model.clock)
    set_diffusivity!(model, diffusivity)
    set_initial_condition!(model, amplitude)

    # Do time-stepping
    Nx, Ny, Nz = size(model.grid)
    κ_max = maximum_diffusivity
    Δz = 1 / Nz
    Δt = 1e-1 * Δz^2 / κ_max

    for _ = 1:10
        time_step!(model, Δt; euler=true)
    end

    # Compute scalar metric
    c = model.tracers.c

    # Hard way (for enzyme - the sum function sometimes errors with AD)
    # c² = c^2
    # sum_c² = sum(c²)

    # Another way to compute it
    sum_c² = 0.0
    for k = 1:Nz, j = 1:Ny,  i = 1:Nx
        sum_c² += c[i, j, k]^2
    end

    # Need the ::Float64 for type inference with automatic differentiation
    return sum_c²::Float64
end

@testset "Enzyme unit tests" begin
    arch = CPU()
    FT = Float64

    N = 100
    topo = (Periodic, Flat, Flat)
    grid = RectilinearGrid(arch, FT, topology=topo, size=N, halo=2, x=(-1, 1), y=(-1, 1), z=(-1, 1))
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, typeof(Const(grid)))
    tape, primal, shadowp = fwd(Const(f), Const(grid))

    # @show tape primal shadowp

    shadow = if shadowp isa Base.RefValue
        shadowp[]
    else
        shadowp
    end

    @test size(primal) == size(shadow)
end

function set_initial_condition_via_launch!(model_tracer, amplitude)
    # Set initial condition
    amplitude = Ref(amplitude)
    cᵢ(x, y, z) = amplitude[]

    temp = Base.broadcasted(Base.identity, FunctionField((Center, Center, Center), cᵢ, model_tracer.grid))

    temp = convert(Base.Broadcast.Broadcasted{Nothing}, temp)
    grid = model_tracer.grid
    arch = architecture(model_tracer)

    param = Oceananigans.Utils.KernelParameters(size(model_tracer), map(Oceananigans.Fields.offset_index, model_tracer.indices))
    Oceananigans.Utils.launch!(arch, grid, param, Oceananigans.Fields._broadcast_kernel!, model_tracer, temp)

    return nothing
end

function momentum_equation!(model)

    # Do time-stepping
    Nx, Ny, Nz = size(model.grid)
    Δz = 1 / Nz
    Δt = 1e-1 * Δz^2

    model.clock.time = 0
    model.clock.iteration = 0

    for _ = 1:100
        time_step!(model, Δt; euler=true)
    end

    # Compute scalar metric
    u = model.velocities.u

    # Hard way (for enzyme - the sum function sometimes errors with AD)
    # c² = c^2
    # sum_c² = sum(c²)

    # Another way to compute it
    sum_u² = 0.0
    for k = 1:Nz, j = 1:Ny,  i = 1:Nx
        sum_u² += u[i, j, k]^2
    end

    # Need the ::Float64 for type inference with automatic differentiation
    return sum_u²::Float64
end

@testset "Enzyme + Oceananigans Initialization Broadcast Kernel" begin
    Nx = Ny = 64
    Nz = 8

    x = y = (-π, π)
    z = (-0.5, 0.5)
    topology = (Periodic, Periodic, Bounded)

    grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)
    model = HydrostaticFreeSurfaceModel(grid; tracers=:c)
    model_tracer = model.tracers.c

    amplitude = 1.0
    amplitude = Ref(amplitude)
    cᵢ(x, y, z) = amplitude[]
    temp = Base.broadcasted(Base.identity, FunctionField((Center, Center, Center), cᵢ, model_tracer.grid))

    temp = convert(Base.Broadcast.Broadcasted{Nothing}, temp)
    grid = model_tracer.grid
    arch = architecture(model_tracer)

    if arch == CPU()
        param = Oceananigans.Utils.KernelParameters(size(model_tracer),
                                                    map(Oceananigans.Fields.offset_index, model_tracer.indices))
        dmodel_tracer = Enzyme.make_zero(model_tracer)

        # Test the individual kernel launch
        autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
                 Oceananigans.Utils.launch!,
                 Const(arch),
                 Const(grid),
                 Const(param),
                 Const(Oceananigans.Fields._broadcast_kernel!),
                 Duplicated(model_tracer, dmodel_tracer),
                 Const(temp))

        # Test out differentiation of the broadcast infrastructure
        autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
                 set_initial_condition_via_launch!,
                 Duplicated(model_tracer, dmodel_tracer),
                 Active(1.0))

        # Test differentiation of the high-level set interface
        dmodel = Enzyme.make_zero(model)
        autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
                 set_initial_condition!,
                 Duplicated(model, dmodel),
                 Active(1.0))
    end
end

@testset "Enzyme for advection and diffusion with various boundary conditions" begin
    Nx = Ny = 64
    Nz = 8

    Lx = Ly = L = 2π
    Lz = 1

    x = y = (-L/2, L/2)
    z = (-Lz/2, Lz/2)
    topology = (Periodic, Periodic, Bounded)

    grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)
    diffusion = VerticalScalarDiffusivity(κ=0.1)

    u = XFaceField(grid)
    v = YFaceField(grid)

    U = 1
    u₀(x, y, z) = - U * cos(x + L/8) * sin(y) * (z + L/2)
    v₀(x, y, z) = + U * sin(x + L/8) * cos(y) * (z + L/2)

    set!(u, u₀)
    set!(v, v₀)
    fill_halo_regions!(u)
    fill_halo_regions!(v)

    @inline function tracer_flux(i, j, grid, clock, model_fields, p)
        c₀ = p.surface_tracer_concentration
        u★ = p.piston_velocity
        return - u★ * (c₀ - model_fields.c[i, j, p.level])
    end

    parameters = (surface_tracer_concentration = 1,
                  piston_velocity = 0.1,
                  level = Nz)

    top_c_bc = FluxBoundaryCondition(tracer_flux; discrete_form=true, parameters)
    c_bcs = FieldBoundaryConditions(top=top_c_bc)

    # TODO:
    # 1. Make the velocity fields evolve
    # 2. Add surface fluxes
    # 3. Do a problem where we invert for the tracer fluxes (maybe with CATKE)

    model_no_bc = HydrostaticFreeSurfaceModel(grid;
                                              tracer_advection = WENO(),
                                              tracers = :c,
                                              velocities = PrescribedVelocityFields(; u, v),
                                              closure = diffusion)

    model_bc = HydrostaticFreeSurfaceModel(grid;
                                           tracer_advection = WENO(),
                                           tracers = :c,
                                           velocities = PrescribedVelocityFields(; u, v),
                                           boundary_conditions = (; c=c_bcs),
                                           closure = diffusion)

    models = [model_no_bc, model_bc]

    @info "Advection-diffusion results, first without then with flux BC"

    for i in 1:2
        # Compute derivative by hand
        κ₁, κ₂ = 0.99, 1.01
        c²₁ = stable_diffusion!(models[i], 1, κ₁)
        c²₂ = stable_diffusion!(models[i], 1, κ₂)
        dc²_dκ_fd = (c²₂ - c²₁) / (κ₂ - κ₁)

        # Now for real
        amplitude = 1.0
        κ = 1.0
        dmodel = Enzyme.make_zero(models[i])
        set_diffusivity!(dmodel, 0)

        dc²_dκ = autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
                          stable_diffusion!,
                          Duplicated(models[i], dmodel),
                          Const(amplitude),
                          Active(κ))

        @info """ \n
        Advection-diffusion:
        Enzyme computed $dc²_dκ
        Finite differences computed $dc²_dκ_fd
        """

        tol = 0.01
        rel_error = abs(dc²_dκ[1][3] - dc²_dκ_fd) / abs(dc²_dκ_fd)
        @test rel_error < tol
    end
end

function set_viscosity!(model, viscosity)
    new_closure = ScalarDiffusivity(ν=viscosity)
    names = ()
    new_closure = with_tracers(names, new_closure)
    model.closure = new_closure
    return nothing
end

function viscous_hydrostatic_turbulence(ν, model, u_init, v_init, Δt, u_truth, v_truth)
    # Initialize the model
    reset!(model.clock)
    set_viscosity!(model, ν)
    set!(model, u=u_init, v=v_init)
    fill!(model.free_surface.displacement, 0)

    # Step it forward
    for n = 1:10
        time_step!(model, Δt)
    end

    # Compute the sum square error
    u, v, w = model.velocities
    Nx, Ny, Nz = size(model.grid)
    err = 0.0
    for j = 1:Ny, i = 1:Nx
        err += @inbounds (u[i, j, 1] - u_truth[i, j, 1])^2 +
                         (v[i, j, 1] - v_truth[i, j, 1])^2
    end

    return err::Float64
end

@testset "Enzyme autodifferentiation of hydrostatic turbulence" begin
    Random.seed!(123)
    arch = CPU()
    Nx = Ny = 32
    Nz = 1
    x = y = (0, 2π)
    z = (0, 1)
    ν₀ = 1e-2

    grid = RectilinearGrid(arch, size=(Nx, Ny, 1); x, y, z, topology=(Periodic, Periodic, Bounded))
    closure = ScalarDiffusivity(ν=ν₀)
    momentum_advection = Centered(order=2)

    g = 4^2
    c = sqrt(g)
    free_surface = ExplicitFreeSurface(gravitational_acceleration=g)
    model = HydrostaticFreeSurfaceModel(grid; momentum_advection, free_surface, closure)

    ϵ(x, y, z) = 2randn() - 1
    set!(model, u=ϵ, v=ϵ)

    u_init = deepcopy(model.velocities.u)
    v_init = deepcopy(model.velocities.v)

    Δx = minimum_xspacing(grid)
    Δt = 0.01 * Δx / c
    for n = 1:10
        time_step!(model, Δt)
    end

    u_truth = deepcopy(model.velocities.u)
    v_truth = deepcopy(model.velocities.v)

    # Use a manual finite difference (central difference) to compute the gradient at ν1 = ν₀ + Δν
    Δν = 1e-6
    ν0 = ν₀
    ν1 = ν₀ + Δν
    ν2 = ν₀ + 2Δν
    e0 = viscous_hydrostatic_turbulence(ν0, model, u_init, v_init, Δt, u_truth, v_truth)
    e2 = viscous_hydrostatic_turbulence(ν2, model, u_init, v_init, Δt, u_truth, v_truth)
    ΔeΔν = (e2 - e0) / 2Δν

    @info "Finite difference computed: $ΔeΔν"

    @info "Now with autodiff..."
    start_time = time_ns()

    # Use autodiff to compute a gradient at ν1 = ν₀ + Δν
    dmodel = Enzyme.make_zero(model)
    dedν = @with_stack_size autodiff(set_runtime_activity(Enzyme.Reverse),
                                     viscous_hydrostatic_turbulence,
                                     Active(ν1),
                                     Duplicated(model, dmodel),
                                     Const(u_init),
                                     Const(v_init),
                                     Const(Δt),
                                     Const(u_truth),
                                     Const(v_truth))

    @info "Automatically computed: $dedν."
    @info "Elapsed time: " * prettytime(1e-9 * (time_ns() - start_time))

    tol = 1e-1
    rel_error = abs(dedν[1][1] - ΔeΔν) / abs(ΔeΔν)
    @test rel_error < tol
end

@testset "Enzyme rule for the implicit free-surface PCG solve" begin
    using Oceananigans.BoundaryConditions: fill_halo_regions!
    using Oceananigans.Solvers: solve!
    using Oceananigans.Models.HydrostaticFreeSurfaceModels: implicit_free_surface_linear_operation!
    using LinearAlgebra: I, dot, norm

    Nx, Ny = 12, 8
    underlying = LatitudeLongitudeGrid(size=(Nx, Ny, 3), halo=(4, 4, 4), longitude=(0, 360),
                                       latitude=(-60, 60), z=(-3000, 0))
    depth = -1000 .* [1 + mod(i + 2j, 3) for i in 1:Nx, j in 1:Ny]
    depth[3:4, 2:5] .= 0 # a continent
    depth[9, 6] = 0      # an island
    grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(depth))
    model = HydrostaticFreeSurfaceModel(grid; free_surface=ImplicitFreeSurface(), buoyancy=nothing, tracers=())

    solver = model.free_surface.implicit_step_solver
    g, Δt = model.free_surface.gravitational_acceleration, 600.0
    η = model.free_surface.displacement
    rhs = solver.right_hand_side
    ∫ᶻA = solver.vertically_integrated_lateral_areas
    wet = vec(depth .< 0)
    N = Nx * Ny

    set_interior!(f, x) = (fill!(parent(f), 0); interior(f) .= reshape(x, Nx, Ny, 1); fill_halo_regions!(f); f)
    interior_vector(f) = vec(Array(interior(f)))

    # Dense matrix of the linear operation, restricted to the wet columns
    function apply_A(x)
        p = set_interior!(similar(η), x)
        q = similar(η)
        fill!(parent(q), 0)
        implicit_free_surface_linear_operation!(q, p, ∫ᶻA.xᶠᶜᶜ, ∫ᶻA.yᶜᶠᶜ, g, Δt)
        return interior_vector(q)
    end
    A = reduce(hcat, apply_A(Matrix(1.0I, N, N)[:, n]) for n in 1:N)
    @test A ≈ A' # the rules rely on the symmetry of A
    Aʷ = A[wet, wet]

    b = [sin(n) for n in 1:N] .* wet # primal right-hand side
    y = [cos(n) for n in 1:N] .* wet # adjoint seed η̄ and tangent rhṡ

    # Reverse: rhs̄ = A⁻¹ η̄
    set_interior!(rhs, b)
    set_interior!(η, zeros(N))
    dη = set_interior!(similar(η), y)
    drhs = set_interior!(similar(rhs), zeros(N))
    autodiff(Enzyme.Reverse, solve!, Const, Duplicated(η, dη), Const(solver), Duplicated(rhs, drhs), Const(g), Const(Δt))
    r̄ = interior_vector(drhs)
    @test norm(r̄[wet] - Aʷ \ y[wet]) < 1e-9 * norm(Aʷ \ y[wet])
    @test all(iszero, r̄[.!wet])
    @test all(iszero, parent(dη))
    # Dot test
    @test dot(Aʷ \ b[wet], y[wet]) ≈ dot(b, r̄) rtol=1e-9

    # Forward: η̇ = A⁻¹ rhṡ
    set_interior!(rhs, b)
    set_interior!(η, zeros(N))
    dη = set_interior!(similar(η), zeros(N))
    drhs = set_interior!(similar(rhs), y)
    autodiff(Enzyme.Forward, solve!, Const, Duplicated(η, dη), Const(solver), Duplicated(rhs, drhs), Const(g), Const(Δt))
    η̇ = interior_vector(dη)
    @test norm(η̇[wet] - Aʷ \ y[wet]) < 1e-9 * norm(Aʷ \ y[wet])
    @test all(iszero, η̇[.!wet])
end
