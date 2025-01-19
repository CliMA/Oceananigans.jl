include("dependencies_for_runtests.jl")

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
    set_diffusivity!(model, diffusivity)
    set_initial_condition!(model, amplitude)
    
    # Do time-stepping
    Nx, Ny, Nz = size(model.grid)
    κ_max = maximum_diffusivity
    Δz = 1 / Nz
    Δt = 1e-1 * Δz^2 / κ_max

    model.clock.time = 0
    model.clock.iteration = 0

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

    @show tape primal shadowp

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

@testset "Enzyme on advection and diffusion" begin
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

    @inline function tracer_flux(x, y, t, c, p)
        c₀ = p.surface_tracer_concentration
        u★ = p.piston_velocity
        return - u★ * (c₀ - c)
    end

    parameters = (surface_tracer_concentration = 1,
                  piston_velocity = 0.1)

    top_c_bc = FluxBoundaryCondition(tracer_flux, field_dependencies=:c; parameters)
    c_bcs = FieldBoundaryConditions(top=top_c_bc)

    # TODO:
    # 1. Make the velocity fields evolve
    # 2. Add surface fluxes
    # 3. Do a problem where we invert for the tracer fluxes (maybe with CATKE)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        tracer_advection = WENO(),
                                        tracers = :c,
                                        velocities = PrescribedVelocityFields(; u, v),
                                        closure = diffusion)

    # Compute derivative by hand
    κ₁, κ₂ = 0.9, 1.1
    c²₁ = stable_diffusion!(model, 1, κ₁)
    c²₂ = stable_diffusion!(model, 1, κ₂)
    dc²_dκ_fd = (c²₂ - c²₁) / (κ₂ - κ₁)

    # Now for real
    amplitude = 1.0
    κ = 1.0
    dmodel = Enzyme.make_zero(model)
    set_diffusivity!(dmodel, 0)

    dc²_dκ = autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
                      stable_diffusion!,
                      Duplicated(model, dmodel),
                      Const(amplitude),
                      Active(κ))

    @info """ \n
        Enzyme computed $dc²_dκ
        Finite differences computed $dc²_dκ_fd
    """

    tol = 0.01
    rel_error = abs(dc²_dκ[1][3] - dc²_dκ_fd) / abs(dc²_dκ_fd)
    @test rel_error < tol
end

