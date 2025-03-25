include("dependencies_for_runtests.jl")

using Enzyme
using Oceananigans.TimeSteppers: reset!
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

# Required presently
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.strictAliasing!(false)
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
    model = HydrostaticFreeSurfaceModel(; grid, tracers=:c)
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

    model_no_bc = HydrostaticFreeSurfaceModel(; grid,
                                              tracer_advection = WENO(),
                                              tracers = :c,
                                              velocities = PrescribedVelocityFields(; u, v),
                                              closure = diffusion)

    model_bc = HydrostaticFreeSurfaceModel(; grid,
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
    fill!(model.free_surface.η, 0)

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
    model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, free_surface, closure)

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
    dedν = autodiff(set_runtime_activity(Enzyme.Reverse),
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

function time_step_with_buoyancy(simulation, Tᵢ, Sᵢ, wind_stress)
    model = simulation.model

    model.tracers.T .= Tᵢ
    model.tracers.S .= Sᵢ
    model.velocities.u.boundary_conditions.top.condition .= wind_stress

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0
    model.clock.last_Δt = Inf

    # Step it forward
    #run!(simulation)
    for n = 1:10
        time_step!(model, 20minutes; euler=true)
    end

    return nothing
end

function compute_forward_u(simulation, initial_temperature, initial_salinity, wind_stress, i, j, k)
    time_step_with_buoyancy(simulation, initial_temperature, initial_salinity, wind_stress)
    
    # Another way to compute it
    forward_u = simulation.model.velocities.u[i, j, k]
    
    return forward_u::Float64
end

@testset "Enzyme autodifferentiation of turbulence with buoyancy on a LatLongGrid" begin
    arch = CPU()
    Nλ = 32
    Nφ = 32
    Nz = 2
    
    Lλ = 60
    Lφ = 60
    Lz = 1800
    
    φ₀ = 15
    
    grid = LatitudeLongitudeGrid(arch,
                                 size = (Nλ, Nφ, Nz),
                                 halo = (7, 7, 7),
                                 longitude = (-Lλ/2, Lλ/2),
                                 latitude = (φ₀, φ₀ + Lφ),
                                 z = [-Lz, -450, 0])
    
    horizontal_closure = HorizontalScalarDiffusivity(ν = 5000.0, κ = 1000.0)
    vertical_closure   = VerticalScalarDiffusivity(ν = 1e-2, κ = 1e-5)

    closure  = (horizontal_closure, vertical_closure)
    coriolis = HydrostaticSphericalCoriolis()

    # Setting wind stress boundary condition
    ρₒ          = 1026.0 # kg m⁻³, average density at the surface of the world ocean
    τ₀          = 0.1 / ρₒ # N m⁻² / density of seawater
    no_slip_bc  = ValueBoundaryCondition(0.0)
    wind_stress = Field{Face, Center, Nothing}(grid)

    u_top_bc    = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))
    @inline τx(λ, φ) = τ₀ * cos(2π * (φ - φ₀) / Lφ)
    set!(wind_stress, τx)

    set!(u_top_bc.condition, wind_stress)

    # Setting overall boundary conditions:
    u_bcs = FieldBoundaryConditions(north=no_slip_bc, south=no_slip_bc, top=u_top_bc)
    v_bcs = FieldBoundaryConditions(east=no_slip_bc, west=no_slip_bc)

    momentum_advection = VectorInvariant()
    tracer_advection   = Centered(order=2)

    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())

    g = 4^2
    c = sqrt(g)
    free_surface = ExplicitFreeSurface(gravitational_acceleration=g)

    model = HydrostaticFreeSurfaceModel(; grid,
                                    coriolis = coriolis,
                                    momentum_advection,
                                    tracer_advection,
                                    closure = closure,
                                    tracers = (:T, :S),
                                    boundary_conditions = (u=u_bcs, v=v_bcs),
                                    buoyancy = buoyancy)

    simulation = Simulation(model, Δt=20minutes, stop_iteration=10)

    # Set initial temperature and salinity
    dTdz = 30.0 / 1900.0
    # Temperature initial condition: a stable density gradient with random noise superposed.
    fₜ(λ, φ, z) = 30 + dTdz * z
    fₛ(λ, φ, z) = 35

    Tᵢ = Field{Center, Center, Center}(grid)
    Sᵢ = Field{Center, Center, Center}(grid)

    set!(Tᵢ, fₜ)
    set!(Sᵢ, fₛ)
    
    # Use a manual finite difference (central difference) to compute the gradient at ν1 = ν₀ + Δν
    i  = 10
    j  = 10
    k  = 1
    J0 = deepcopy(wind_stress)
    J1 = deepcopy(wind_stress)
    J2 = deepcopy(wind_stress)

    ΔJ = 1e-3

    J1[i, j, k] = J0[i, j, k] + ΔJ
    J2[i, j, k] = J2[i, j, k] + 2ΔJ
    e0 = compute_forward_u(simulation, Tᵢ, Sᵢ, J0, i+1, j, k)
    set!(simulation.model, u=0, v=0, T=0, S=0)
    e2 = compute_forward_u(simulation, Tᵢ, Sᵢ, J2, i+1, j, k)
    set!(simulation.model, u=0, v=0, T=0, S=0)
    ΔeΔJ = (e2 - e0) / 2ΔJ

    @info "Finite difference computed for wind stress at index $i, $j, $k: $ΔeΔJ"

    @info "Now with autodiff..."
    start_time = time_ns()
    
    dmodel = Enzyme.make_zero(model)
    dsim   = Simulation(dmodel, Δt=20minutes, stop_iteration=10)

    dTᵢ  = Enzyme.make_zero(Tᵢ)
    dSᵢ  = Enzyme.make_zero(Sᵢ)
    dJ1  = Enzyme.make_zero(J1)

    # Use autodiff to compute a gradient at J1 = wind_stress with permutation
    dmodel = Enzyme.make_zero(model)
    dedJ = autodiff(set_runtime_activity(Enzyme.Reverse),
                    compute_forward_u, Active,
                    Duplicated(simulation, dsim),
                    Duplicated(Tᵢ, dTᵢ),
                    Duplicated(Sᵢ, dSᵢ),
                    Duplicated(J1, dJ1),
                    Const(i+1),
                    Const(j),
                    Const(k))

    @info "Automatically computed: $dedJ."
    @info "Elapsed time: " * prettytime(1e-9 * (time_ns() - start_time))

    tol = 1e-1
    rel_error = abs(dJ1[i, j, k] - ΔeΔJ) / abs(ΔeΔJ)
    @test rel_error < tol
end