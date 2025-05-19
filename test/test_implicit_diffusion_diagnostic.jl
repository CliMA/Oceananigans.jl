include("dependencies_for_runtests.jl")

using Oceananigans
using Oceananigans.Simulations: VarianceDissipation
using KernelAbstractions: @kernel, @index

@kernel function _compute_dissipation!(Δtc², c⁻, c, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Δtc²[i, j, k] = (c[i, j, k]^2 - c⁻[i, j, k]^2) / Δt
        c⁻[i, j, k]   = c[i, j, k]
    end
end

function compute_tracer_dissipation!(sim)
    c    = sim.model.tracers.c
    c⁻   = sim.model.auxiliary_fields.c⁻
    Δtc² = sim.model.auxiliary_fields.Δtc²
    grid = sim.model.grid
    Oceananigans.Utils.launch!(architecture(grid), grid, :xyz, 
                               _compute_dissipation!, 
                               Δtc², c⁻, c, sim.Δt)

    return nothing
end

periodic_grid(arch, ::Val{:x}) = RectilinearGrid(arch; size=20, x=(-1, 1), halo=5, topology = (Periodic, Flat, Flat))
periodic_grid(arch, ::Val{:y}) = RectilinearGrid(arch; size=20, y=(-1, 1), halo=5, topology = (Flat, Periodic, Flat))
periodic_grid(arch, ::Val{:z}) = RectilinearGrid(arch; size=20, z=(-1, 1), halo=5, topology = (Flat, Flat, Periodic))

get_advection_dissipation(::Val{:x}) = FieldTimeSeries("one_d_simulation_x.jld2", "Acx")
get_advection_dissipation(::Val{:y}) = FieldTimeSeries("one_d_simulation_y.jld2", "Acy")
get_advection_dissipation(::Val{:z}) = FieldTimeSeries("one_d_simulation_z.jld2", "Acz")

get_diffusion_dissipation(::Val{:x}) = FieldTimeSeries("one_d_simulation_x.jld2", "Dcx")
get_diffusion_dissipation(::Val{:y}) = FieldTimeSeries("one_d_simulation_y.jld2", "Dcy")
get_diffusion_dissipation(::Val{:z}) = FieldTimeSeries("one_d_simulation_z.jld2", "Dcz")

advecting_velocity(::Val{:x}) = PrescribedVelocityFields(u = 1)
advecting_velocity(::Val{:y}) = PrescribedVelocityFields(v = 1)
advecting_velocity(::Val{:z}) = PrescribedVelocityFields(w = 1)

function test_implicit_diffusion_diagnostic(arch, dim, schedule)

    # 1D grid constructions
    grid = periodic_grid(arch, Val(dim))

    # Change to test pure advection schemes
    tracer_advection = WENO(order=5)
    closure = ScalarDiffusivity(κ=1e-3)
    velocities = advecting_velocity(Val(dim))

    c⁻   = CenterField(grid)
    Δtc² = CenterField(grid)

    model = HydrostaticFreeSurfaceModel(; grid, 
                                        timestepper=:QuasiAdamsBashforth2, 
                                        velocities, 
                                        tracer_advection, 
                                        closure, 
                                        tracers=:c,
                                        auxiliary_fields=(; Δtc², c⁻))

    c₀(x) = sin(2π * x)
    set!(model, c=c₀)
    set!(model.auxiliary_fields.c⁻, c₀)

    sim = Simulation(model; Δt=0.01, stop_time=1)

    ϵ = VarianceDissipation(:c, grid)
    f = Oceananigans.Simulations.VarianceDissipationComputations.flatten_dissipation_fields(ϵ)

    outputs = merge((; c = model.tracers.c, Δtc² = model.auxiliary_fields.Δtc²), f)
    
    add_callback!(sim, ϵ, schedule)

    sim.output_writers[:solution] = JLD2Writer(model, outputs;
                                               filename="one_d_simulation_$(dim).jld2",
                                               schedule, # Make sure it is the same schedule as the one where we compute the dissipation
                                               overwrite_existing=true,
                                               array_type = Array{Float64})

    sim.callbacks[:compute_tracer_dissipation] = Callback(compute_tracer_dissipation!, IterationInterval(1))

    run!(sim)

    Δtc² = FieldTimeSeries("one_d_simulation_$(dim).jld2", "Δtc²") 
    Ac   = get_advection_dissipation(Val(dim))
    Dc   = get_diffusion_dissipation(Val(dim))

    Nt = length(Ac.times)

    ∫closs = [sum(interior(Δtc²[i]))  for i in 1:Nt]
    ∫A     = [sum(interior(Ac[i]))    for i in 1:Nt]
    ∫D     = [sum(interior(Dc[i]))    for i in 1:Nt] 

    Δ = min(grid.Δxᶜᵃᵃ, grid.Δyᵃᶜᵃ, grid.z.Δᵃᵃᶜ)

    for i in 1:Nt-1
        @test abs(∫closs[i] * Δ - ∫A[i] - ∫D[i]) < 1e-14 # Arbitrary tolerance, not exactly machine precision
    end
end
@testset "Implicit Diffusion Diagnostic" begin
    @info "Testing implicit diffusion diagnostic..."
    for arch in archs
        schedules = [IterationInterval(1), IterationInterval(10)]
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
