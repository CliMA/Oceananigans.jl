include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans
using Oceananigans.Models.VarianceDissipationComputations
using KernelAbstractions: @kernel, @index

@kernel function _compute_dissipation!(Δtc², c⁻, c, Δtd², d⁻, d, grid, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Δtc²[i, j, k] = (c[i, j, k]^2 - c⁻[i, j, k]^2) / Δt * Vᶜᶜᶜ(i, j, k, grid)
        c⁻[i, j, k]   = c[i, j, k]
        Δtd²[i, j, k] = (d[i, j, k]^2 - d⁻[i, j, k]^2) / Δt * Vᶜᶜᶜ(i, j, k, grid)
        d⁻[i, j, k]   = d[i, j, k]
    end
end

function compute_tracer_dissipation!(sim)
    c    = sim.model.tracers.c
    d    = sim.model.tracers.d
    c⁻   = sim.model.auxiliary_fields.c⁻
    d⁻   = sim.model.auxiliary_fields.d⁻
    Δtc² = sim.model.auxiliary_fields.Δtc²
    Δtd² = sim.model.auxiliary_fields.Δtd²
    grid = sim.model.grid
    Oceananigans.Utils.launch!(architecture(grid), grid, :xyz,
                               _compute_dissipation!,
                               Δtc², c⁻, c, Δtd², d⁻, d, grid, sim.Δt)

    return nothing
end

periodic_grid(arch, ::Val{:x}) = RectilinearGrid(arch; size=20, x=(-1, 1), halo=5, topology = (Periodic, Flat, Flat))
periodic_grid(arch, ::Val{:y}) = RectilinearGrid(arch; size=20, y=(-1, 1), halo=5, topology = (Flat, Periodic, Flat))
periodic_grid(arch, ::Val{:z}) = RectilinearGrid(arch; size=20, z=(-1, 1), halo=5, topology = (Flat, Flat, Periodic))

get_advection_dissipation(filepath, dim, t) = FieldTimeSeries(filepath, "A$(t)$(dim)")
get_diffusion_dissipation(filepath, dim, t) = FieldTimeSeries(filepath, "D$(t)$(dim)")

advecting_velocity(::Val{:x}) = PrescribedVelocityFields(u = 1)
advecting_velocity(::Val{:y}) = PrescribedVelocityFields(v = 1)
advecting_velocity(::Val{:z}) = PrescribedVelocityFields(w = 1)

function test_implicit_diffusion_diagnostic(arch, dim, timestepper, schedule)

    # 1D grid constructions
    grid = periodic_grid(arch, Val(dim))

    # Change to test pure advection schemes
    tracer_advection = (c=WENO(order=7), d = Centered(order=4))
    closure = (ScalarDiffusivity(κ=(c=1e-3, d=1e-5)), ScalarDiffusivity(κ=1e-4))
    velocities = advecting_velocity(Val(dim))

    c⁻   = CenterField(grid)
    d⁻   = CenterField(grid)
    Δtc² = CenterField(grid)
    Δtd² = CenterField(grid)

    model = HydrostaticFreeSurfaceModel(grid;
                                        timestepper,
                                        velocities,
                                        tracer_advection,
                                        closure,
                                        tracers=(:c, :d),
                                        auxiliary_fields=(; Δtc², c⁻, Δtd², d⁻))

    c₀(x) = sin(2π  * x)
    d₀(x) = cos(10π * x)

    set!(model, c=c₀, d=d₀)
    set!(model.auxiliary_fields.c⁻, c₀)
    set!(model.auxiliary_fields.d⁻, d₀)

    sim = Simulation(model; Δt=0.01, stop_time=1, verbose=false)

    ϵc = VarianceDissipation(:c, grid)
    ϵd = VarianceDissipation(:d, grid)

    fc = flatten_dissipation_fields(ϵc)
    fd = flatten_dissipation_fields(ϵd)

    outputs = merge(model.tracers, model.auxiliary_fields, fd, fc)

    # Add both callbacks to the simulation with a schedule
    schedule_logs = schedule == IterationInterval(1) ? () :
        ((:warn, "VarianceDissipation callback must be called every Iteration or on `ConsecutiveIterations`. \n" *
                 "Changing `schedule` to `ConsecutiveIterations(schedule)`."),)

    @test_logs schedule_logs... add_callback!(sim, ϵc, schedule)
    @test_logs schedule_logs... add_callback!(sim, ϵd, schedule)

    dir = mktempdir()
    sim.output_writers[:solution] = JLD2Writer(model, outputs;
                                               dir,
                                               filename="one_d_simulation_$(dim).jld2",
                                               schedule, # Make sure it is the same schedule as the one where we compute the dissipation
                                               overwrite_files=true,
                                               array_type = Array{Float64})

    sim.callbacks[:compute_tracer_dissipation] = Callback(compute_tracer_dissipation!, IterationInterval(1))

    run!(sim)

    filepath = sim.output_writers[:solution].filepath
    Δtc² = FieldTimeSeries(filepath, "Δtc²")
    Ac   = get_advection_dissipation(filepath, dim, :c)
    Dc   = get_diffusion_dissipation(filepath, dim, :c)

    Δtd² = FieldTimeSeries(filepath, "Δtd²")
    Ad   = get_advection_dissipation(filepath, dim, :d)
    Dd   = get_diffusion_dissipation(filepath, dim, :d)

    Nt = length(Ac.times)

    ∫closs = [sum(interior(Δtc²[i]))  for i in 1:Nt]
    ∫Ac    = [sum(interior(Ac[i]))    for i in 1:Nt]
    ∫Dc    = [sum(interior(Dc[i]))    for i in 1:Nt]

    ∫dloss = [sum(interior(Δtd²[i]))  for i in 1:Nt]
    ∫Ad    = [sum(interior(Ad[i]))    for i in 1:Nt]
    ∫Dd    = [sum(interior(Dd[i]))    for i in 1:Nt]

    for i in 3:Nt-1
        @test abs(∫closs[i] - ∫Ac[i] - ∫Dc[i]) < 2e-13 # Arbitrary tolerance, not exactly machine precision
        @test abs(∫dloss[i] - ∫Ad[i] - ∫Dd[i]) < 2e-13 # Arbitrary tolerance, not exactly machine precision
    end

    rm(dir; recursive=true)
end

# Σ(A + K) against the change of Σ V c² over one time step, relative to Σ(|A| + |K|)
function relative_budget_residual(model, Δt; Nsteps=5)
    grid = model.grid
    simulation = Simulation(model; Δt, stop_iteration=Nsteps, verbose=false)
    ϵ = VarianceDissipation(:c, grid)
    add_callback!(simulation, ϵ)

    V = KernelFunctionOperation{Center, Center, Center}(Vᶜᶜᶜ, grid)
    variance = Field(V * model.tracers.c^2)
    productions = (ϵ.advective_production..., ϵ.diffusive_production...)

    residual = 0.0
    for _ in 1:Nsteps
        before = sum(compute!(variance))
        time_step!(simulation)
        after = sum(compute!(variance))
        total = sum(sum(Array(interior(P))) for P in productions)
        scale = sum(sum(abs, Array(interior(P))) for P in productions)
        residual = max(residual, abs(total - (after - before) / Δt) / scale)
    end

    return residual
end

function zstar_lock_exchange(arch, timestepper, free_surface)
    z = MutableVerticalDiscretization((-1000, 0))
    grid = RectilinearGrid(arch; size=(16, 8), x=(0, 16e3), z, topology=(Bounded, Flat, Bounded), halo=(4, 4))
    model = HydrostaticFreeSurfaceModel(grid; timestepper, free_surface, tracers=(:b, :c), buoyancy=BuoyancyTracer(),
                                        momentum_advection=WENO(), tracer_advection=WENO(order=7))
    set!(model, u=(x, z) -> 0.1 * sin(π * x / 16e3),
                b=(x, z) -> 0.01 * tanh((x - 8e3) / 2e3),
                c=(x, z) -> exp(-(x - 6e3)^2 / 2e3^2 - (z + 500)^2 / 200^2))
    return model
end

function implicitly_diffused_tracer(arch, timestepper)
    grid = RectilinearGrid(arch; size=(16, 8), x=(0, 1), z=(-1, 0), topology=(Periodic, Flat, Bounded), halo=(4, 4))
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ=1e-2)
    model = HydrostaticFreeSurfaceModel(grid; timestepper, closure, tracers=:c, buoyancy=nothing,
                                        velocities=PrescribedVelocityFields(u=1), tracer_advection=WENO(order=7))
    set!(model, c=(x, z) -> exp(-(x - 0.5)^2 / 0.01 - (z + 0.5)^2 / 0.01))
    return model
end

@testset "Closed variance budget" begin
    for arch in archs
        @testset "Z-star, SplitRungeKutta3 and split-explicit free surface [$(typeof(arch))]" begin
            model = zstar_lock_exchange(arch, :SplitRungeKutta3, SplitExplicitFreeSurface(substeps=20))
            @test relative_budget_residual(model, 60) < 1e-6
        end
        @testset "Z-star, QuasiAdamsBashforth2 and explicit free surface [$(typeof(arch))]" begin
            model = zstar_lock_exchange(arch, :QuasiAdamsBashforth2, ExplicitFreeSurface())
            @test relative_budget_residual(model, 4) < 1e-6
        end
        @testset "Vertically implicit diffusion and $timestepper [$(typeof(arch))]" for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
            @test relative_budget_residual(implicitly_diffused_tracer(arch, timestepper), 2e-3) < 1e-6
        end
    end
end

@testset "Implicit Diffusion Diagnostic" begin
    schedules = [IterationInterval(1), IterationInterval(10), IterationInterval(100)]
    timesteppers = (:QuasiAdamsBashforth2, :SplitRungeKutta2, :SplitRungeKutta3, :SplitRungeKutta5)
    for arch in archs, schedule in schedules, timestepper in timesteppers
        @testset "Implicit Diffusion on $schedule schedule and $timestepper, in $dim-direction [$(typeof(arch))]" for dim in (:x, :y, :z)
            test_implicit_diffusion_diagnostic(arch, dim, timestepper, schedule)
        end
    end
end
