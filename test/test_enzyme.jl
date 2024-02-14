using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Enzyme

# Required presently
Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true
Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing

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
    # Set initial condition
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
    Δz = 2π / Nz
    Δt = 1e-1 * Δz^2 / κ_max

    model.clock.time = 0
    model.clock.iteration = 0

    for n = 1:10
        time_step!(model, Δt; euler=true)
    end

    # Compute scalar metric
    c = model.tracers.c

    # Hard way
    # c² = c^2
    # sum_c² = sum(c²)

    # Another way to compute it
    sum_c² = 0.0
    for i = 1:Nx, j = 1:Ny, k = 1:Nz
        sum_c² += c[i, j, k]^2
    end

    return sum_c²::Float64
end

@testset "Enzyme Unit Tests" begin
    arch=CPU()
    FT=Float64

    N = 100
    topo = (Periodic, Flat, Flat)
    grid = RectilinearGrid(arch, FT, topology=topo, size=N, halo=2, x=(-1, 1), y=(-1, 1), z=(-1, 1))
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, typeof(Const(grid)))

    tape, primal, shadow = fwd(Const(f), Const(grid) )

    @show tape, primal, shadow

    @test size(primal) == size(shadow)
end


@testset "Enzyme on advection and diffusion" begin
    Nx = Ny = 64
    Nz = 8

    x = y = (-π, π)
    z = (-0.5, 0.5)
    topology = (Periodic, Periodic, Bounded)

    grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)
    diffusion = VerticalScalarDiffusivity(κ=0.1)

    u = XFaceField(grid)
    v = YFaceField(grid)

    U = 1
    u₀(x, y, z) = - U * cos(x + π/4) * sin(y) * (z + 0.5)
    v₀(x, y, z) = + U * sin(x + π/4) * cos(y) * (z + 0.5)

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
                                        buoyancy = nothing,
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

    dc²_dκ = autodiff(Enzyme.Reverse,
                    stable_diffusion!,
                    Duplicated(model, dmodel),
                    Const(amplitude),
                    Active(κ))

    @info """ \n
    Enzyme computed $dc²_dκ
    Finite differences computed $dc²_dκ_fd
    """

    tol = 0.1
    @test abs((dc²_dκ - dc²_dκ_fd) / dc²_dκ_fd) < tol
end