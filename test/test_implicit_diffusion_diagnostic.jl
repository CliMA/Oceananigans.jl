include("dependencies_for_runtests.jl")

using Oceananigans
using Oceananigans.Diagnostics: VarianceDissipation
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
    Oceananigans.Utils.launch!(CPU(), sim.model.grid, :xyz, 
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

function test_implicit_diffusion_diagnostic(arch, dim)

    # 1D grid constructions
    grid = periodic_grid(arch, Val(dim))

    # Change to test pure advection schemes
    tracer_advection = WENO(order=5)
    closure = ScalarDiffusivity(κ=1e-3)
    velocities = PrescribedVelocityFields(u=1)

    c⁻    = CenterField(grid)
    Δtc²  = CenterField(grid)

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

    ϵ = VarianceDissipation(model)
    f = Oceananigans.Diagnostics.VarianceDissipationComputations.flatten_dissipation_fields(ϵ)
    outputs = merge((; c = model.tracers.c, Δtc² = model.auxiliary_fields.Δtc²), f)
    sim.diagnostics[:variance_dissipation] = ϵ
    sim.output_writers[:solution] = JLD2Writer(model, outputs;
                                            filename="one_d_simulation_$(dim).jld2",
                                            schedule=IterationInterval(10),
                                            overwrite_existing=true)

    sim.callbacks[:compute_tracer_dissipation] = Callback(compute_tracer_dissipation!, IterationInterval(1))

    run!(sim)

    Δtc² = FieldTimeSeries("one_d_simulation_$(dim).jld2", "Δtc²") 
    Ac   = get_advection_dissipation(Val(dim))
    Dc   = get_diffusion_dissipation(Val(dim))

    Nt = length(Ac.times)

    ∫closs = [sum(interior(Δtc²[i], :, 1, 1))  for i in 1:Nt]
    ∫A     = [sum(interior(Ac[i],   :, 1, 1))  for i in 1:Nt]
    ∫D     = [sum(interior(Dc[i],   :, 1, 1))  for i in 1:Nt] 

    Δ = min(grid.Δxᶜᵃᵃ, grid.Δyᵃᶜᵃ, grid.z.Δᵃᵃᶜ)

    for i in 1:Nt-1
        @test abs(∫closs[i] * Δ - ∫A[i] - ∫D[i]) < 1e-6
    end
end

@testset "Implicit Diffusion Diagnostic" begin
    @info "Testing implicit diffusion diagnostic..."
    for arch in archs
        @testset "Implicit Diffusion [$(typeof(arch))]" begin
            @info "  Testing implicit diffusion diagnostic [$(typeof(arch))]..."
            test_implicit_diffusion_diagnostic(arch, :x)
            test_implicit_diffusion_diagnostic(arch, :y)
            test_implicit_diffusion_diagnostic(arch, :z)
        end
    end
end
