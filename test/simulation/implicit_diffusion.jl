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

    Uⁿ⁻¹ = Oceananigans.Fields.VelocityFields(grid)
    Uⁿ   = Oceananigans.Fields.VelocityFields(grid)

    sim = Simulation(model; Δt=0.01, stop_time=1, verbose=false)

    ϵc = VarianceDissipation(:c, grid; Uⁿ⁻¹, Uⁿ)
    ϵd = VarianceDissipation(:d, grid; Uⁿ⁻¹, Uⁿ)

    # Check that the advecting velocities are the same field
    @test ϵc.previous_state.Uⁿ   === ϵd.previous_state.Uⁿ
    @test ϵc.previous_state.Uⁿ⁻¹ === ϵd.previous_state.Uⁿ⁻¹

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

@testset "Implicit Diffusion Diagnostic" begin
    schedules = [IterationInterval(1), IterationInterval(10), IterationInterval(100)]
    timesteppers = (:QuasiAdamsBashforth2, :SplitRungeKutta2, :SplitRungeKutta3, :SplitRungeKutta5)
    for arch in archs, schedule in schedules, timestepper in timesteppers
        @testset "Implicit Diffusion on $schedule schedule and $timestepper, in $dim-direction [$(typeof(arch))]" for dim in (:x, :y, :z)
            test_implicit_diffusion_diagnostic(arch, dim, timestepper, schedule)
        end
    end
end
