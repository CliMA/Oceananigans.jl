include("dependencies_for_runtests.jl")

using Oceananigans
using Oceananigans.Simulations: VarianceDissipation
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

get_advection_dissipation(::Val{:x}, t) = FieldTimeSeries("one_d_simulation_x.jld2", "A$(t)x")
get_advection_dissipation(::Val{:y}, t) = FieldTimeSeries("one_d_simulation_y.jld2", "A$(t)y")
get_advection_dissipation(::Val{:z}, t) = FieldTimeSeries("one_d_simulation_z.jld2", "A$(t)z")

get_diffusion_dissipation(::Val{:x}, t) = FieldTimeSeries("one_d_simulation_x.jld2", "D$(t)x")
get_diffusion_dissipation(::Val{:y}, t) = FieldTimeSeries("one_d_simulation_y.jld2", "D$(t)y")
get_diffusion_dissipation(::Val{:z}, t) = FieldTimeSeries("one_d_simulation_z.jld2", "D$(t)z")

advecting_velocity(::Val{:x}) = PrescribedVelocityFields(u = 1)
advecting_velocity(::Val{:y}) = PrescribedVelocityFields(v = 1)
advecting_velocity(::Val{:z}) = PrescribedVelocityFields(w = 1)

function test_implicit_diffusion_diagnostic(arch, dim, schedule)

    # 1D grid constructions
    grid = periodic_grid(arch, Val(dim))

    # Change to test pure advection schemes
    tracer_advection = (c=WENO(order=7), d = Centered(order=4))
    closure = ScalarDiffusivity(κ=(c=1e-3, d=1e-5))
    velocities = advecting_velocity(Val(dim))

    c⁻   = CenterField(grid)
    d⁻   = CenterField(grid)
    Δtc² = CenterField(grid)
    Δtd² = CenterField(grid)

    model = HydrostaticFreeSurfaceModel(; grid, 
                                        timestepper=:QuasiAdamsBashforth2, 
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

    sim = Simulation(model; Δt=0.01, stop_time=1)

    ϵc = VarianceDissipation(:c, grid; Uⁿ⁻¹, Uⁿ)
    ϵd = VarianceDissipation(:d, grid; Uⁿ⁻¹, Uⁿ)

    # Check that the advecting velocities are the same field
    @test ϵc.previous_state.Uⁿ   === ϵd.previous_state.Uⁿ
    @test ϵc.previous_state.Uⁿ⁻¹ === ϵd.previous_state.Uⁿ⁻¹

    fc = Oceananigans.Simulations.VarianceDissipationComputations.flatten_dissipation_fields(ϵc)
    fd = Oceananigans.Simulations.VarianceDissipationComputations.flatten_dissipation_fields(ϵd)

    outputs = merge(model.tracers, model.auxiliary_fields, fd, fc)
    
    # Add both callbacks to the simulation with a schedule
    add_callback!(sim, ϵc, schedule)    
    add_callback!(sim, ϵd, schedule)

    sim.output_writers[:solution] = JLD2Writer(model, outputs;
                                               filename="one_d_simulation_$(dim).jld2",
                                               schedule, # Make sure it is the same schedule as the one where we compute the dissipation
                                               overwrite_existing=true,
                                               array_type = Array{Float64})

    sim.callbacks[:compute_tracer_dissipation] = Callback(compute_tracer_dissipation!, IterationInterval(1))

    run!(sim)

    Δtc² = FieldTimeSeries("one_d_simulation_$(dim).jld2", "Δtc²") 
    Ac   = get_advection_dissipation(Val(dim), :c)
    Dc   = get_diffusion_dissipation(Val(dim), :c)

    Δtd² = FieldTimeSeries("one_d_simulation_$(dim).jld2", "Δtd²") 
    Ad   = get_advection_dissipation(Val(dim), :d)
    Dd   = get_diffusion_dissipation(Val(dim), :d)

    Nt = length(Ac.times)

    ∫closs = [sum(interior(Δtc²[i]))  for i in 1:Nt]
    ∫Ac    = [sum(interior(Ac[i]))    for i in 1:Nt]
    ∫Dc    = [sum(interior(Dc[i]))    for i in 1:Nt] 

    ∫dloss = [sum(interior(Δtd²[i]))  for i in 1:Nt]
    ∫Ad    = [sum(interior(Ad[i]))    for i in 1:Nt]
    ∫Dd    = [sum(interior(Dd[i]))    for i in 1:Nt] 

    for i in 1:Nt-1
        @test abs(∫closs[i] - ∫Ac[i] - ∫Dc[i]) < 2e-14 # Arbitrary tolerance, not exactly machine precision
        @test abs(∫dloss[i] - ∫Ad[i] - ∫Dd[i]) < 2e-14 # Arbitrary tolerance, not exactly machine precision
    end
end

@testset "Implicit Diffusion Diagnostic" begin
    @info "Testing implicit diffusion diagnostic..."
    for arch in archs
        schedules = [IterationInterval(1), IterationInterval(10), IterationInterval(100)]
        for schedule in schedules
            @testset "Implicit Diffusion on $schedule schedule [$(typeof(arch))]" begin
                @info "  Testing implicit diffusion diagnostic [$(typeof(arch))] with $schedule in x-direction..."
                test_implicit_diffusion_diagnostic(arch, :x, schedule)
                @info "  Testing implicit diffusion diagnostic [$(typeof(arch))] with $schedule in y-direction..."
                test_implicit_diffusion_diagnostic(arch, :y, schedule)
                @info "  Testing implicit diffusion diagnostic [$(typeof(arch))] with $schedule in z-direction..."
                test_implicit_diffusion_diagnostic(arch, :z, schedule)
            end
        end
    end
end
