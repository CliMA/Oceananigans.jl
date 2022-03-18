include("dependencies_for_runtests.jl")

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity

for closure in closures
    @eval begin
        using Oceananigans.TurbulenceClosures: $closure
    end
end

function closure_instantiation(closurename)
    closure = getproperty(TurbulenceClosures, closurename)()
    return true
end

function constant_isotropic_diffusivity_basic(T=Float64; ν=T(0.3), κ=T(0.7))
    closure = ScalarDiffusivity(T; κ=(T=κ, S=κ), ν=ν)
    return closure.ν == ν && closure.κ.T == κ
end

function tracer_specific_horizontal_diffusivity(T=Float64; νh=T(0.3), κh=T(0.7))
    closure = HorizontalScalarDiffusivity(κ=(T=κh, S=κh), ν=νh)
    return closure.ν == νh && closure.κ.T == κh && closure.κ.T == κh
end

function run_constant_isotropic_diffusivity_fluxdiv_tests(FT=Float64; ν=FT(0.3), κ=FT(0.7))
          arch = CPU()
       closure = ScalarDiffusivity(FT, κ=(T=κ, S=κ), ν=ν)
          grid = RectilinearGrid(FT, size=(3, 1, 4), extent=(3, 1, 4))
    velocities = VelocityFields(grid)
       tracers = TracerFields((:T, :S), grid)
         clock = Clock(time=0.0)

    u, v, w = velocities
       T, S = tracers

    for k in 1:4
        interior(u)[:, 1, k] .= [0, -1/2, 0]
        interior(v)[:, 1, k] .= [0, -2,   0]
        interior(w)[:, 1, k] .= [0, -3,   0]
        interior(T)[:, 1, k] .= [0, -1,   0]
    end

    model_fields = merge(datatuple(velocities), datatuple(tracers))
    fill_halo_regions!(merge(velocities, tracers), nothing, model_fields)

    U, C = velocities, tracers

    K, b = nothing, nothing
    closure_args = (U, C, clock, b)

    @test ∇_dot_qᶜ(2, 1, 3, grid, closure, K, Val(1), closure_args...) == - 2κ
    @test ∂ⱼ_τ₁ⱼ(2, 1, 3, grid, closure, K, closure_args...) == - 2ν
    @test ∂ⱼ_τ₂ⱼ(2, 1, 3, grid, closure, K, closure_args...) == - 4ν
    @test ∂ⱼ_τ₃ⱼ(2, 1, 3, grid, closure, K, closure_args...) == - 6ν

    return nothing
end

function horizontal_diffusivity_fluxdiv(FT=Float64; νh=FT(0.3), κh=FT(0.7), νz=FT(0.1), κz=FT(0.5))
    arch       = CPU()
    closureh   = HorizontalScalarDiffusivity(FT, ν=νh, κ=(T=κh, S=κh))
    closurez   = VerticalScalarDiffusivity(FT, ν=νz, κ=(T=κz, S=κz))
    grid       = RectilinearGrid(arch, FT, size=(3, 1, 4), extent=(3, 1, 4))
    eos        = LinearEquationOfState(FT)
    buoyancy   = SeawaterBuoyancy(FT, gravitational_acceleration=1, equation_of_state=eos)
    velocities = VelocityFields(grid)
    tracers    = TracerFields((:T, :S), grid)
    clock      = Clock(time=0.0)

    u, v, w, T, S = merge(velocities, tracers)

    interior(u)[:, 1, 2] .= [0,  1, 0]
    interior(u)[:, 1, 3] .= [0, -1, 0]
    interior(u)[:, 1, 4] .= [0,  1, 0]

    interior(v)[:, 1, 2] .= [0,  1, 0]
    interior(v)[:, 1, 3] .= [0, -2, 0]
    interior(v)[:, 1, 4] .= [0,  1, 0]

    interior(w)[:, 1, 2] .= [0,  1, 0]
    interior(w)[:, 1, 3] .= [0, -3, 0]
    interior(w)[:, 1, 4] .= [0,  1, 0]

    interior(T)[:, 1, 2] .= [0,  1, 0]
    interior(T)[:, 1, 3] .= [0, -4, 0]
    interior(T)[:, 1, 4] .= [0,  1, 0]

    model_fields = merge(datatuple(velocities), datatuple(tracers))
    fill_halo_regions!(merge(velocities, tracers), nothing, model_fields)

    U, C = velocities, tracers

    K, b = nothing, nothing
    closure_args = (U, C, clock, b)

    return (∇_dot_qᶜ(2, 1, 3, grid, closureh, K, Val(1), closure_args...) == -  8κh &&
            ∇_dot_qᶜ(2, 1, 3, grid, closurez, K, Val(1), closure_args...) == - 10κz &&
              ∂ⱼ_τ₁ⱼ(2, 1, 3, grid, closureh, K, closure_args...) == - (2νh) &&
              ∂ⱼ_τ₁ⱼ(2, 1, 3, grid, closurez, K, closure_args...) == - (4νz) &&
              ∂ⱼ_τ₂ⱼ(2, 1, 3, grid, closureh, K, closure_args...) == - (4νh) &&
              ∂ⱼ_τ₂ⱼ(2, 1, 3, grid, closurez, K, closure_args...) == - (6νz) &&
              ∂ⱼ_τ₃ⱼ(2, 1, 3, grid, closureh, K, closure_args...) == - (6νh) &&
              ∂ⱼ_τ₃ⱼ(2, 1, 3, grid, closurez, K, closure_args...) == - (8νz))
end

function time_step_with_variable_isotropic_diffusivity(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 2, 3))
    closure = ScalarDiffusivity(ν = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t),
                                κ = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t))

    model = NonhydrostaticModel(; grid, closure)
    time_step!(model, 1, euler=true)
    return true
end

function time_step_with_variable_anisotropic_diffusivity(arch)
    clov = VerticalScalarDiffusivity(ν = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t),
                                     κ = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t))

    cloh = HorizontalScalarDiffusivity(ν = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t),
                                       κ = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t))
    for clo in (clov, cloh)
        model = NonhydrostaticModel(grid=RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 2, 3)), closure=clo)
        time_step!(model, 1, euler=true)
    end
    return true
end

function time_step_with_tupled_closure(FT, arch)
    closure_tuple = (AnisotropicMinimumDissipation(FT), ScalarDiffusivity(FT))

    model = NonhydrostaticModel(closure=closure_tuple,
                                grid=RectilinearGrid(arch, FT, size=(1, 1, 1), extent=(1, 2, 3)))

    time_step!(model, 1, euler=true)
    return true
end

function run_time_step_with_catke_tests(arch, closure)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 2, 3))
    buoyancy = BuoyancyTracer()

    # These shouldn't work (need :e in tracers)
    @test_throws ArgumentError HydrostaticFreeSurfaceModel(; grid, closure, buoyancy, tracers=:b)
    @test_throws ArgumentError HydrostaticFreeSurfaceModel(; grid, closure, buoyancy, tracers=(:b, :E))

    # CATKE isn't supported with NonhydrostaticModel (we don't diffuse vertical velocity)
    @test_throws ErrorException NonhydrostaticModel(; grid, closure, buoyancy, tracers=(:b, :c, :e))

    model = HydrostaticFreeSurfaceModel(; grid, closure, buoyancy, tracers = (:b, :c, :e))

    # Default boundary condition is Flux, Nothing... with CATKE this has to change.
    @test !(model.tracers.e.boundary_conditions.top.condition isa BoundaryCondition{Flux, Nothing})

    # Can we time-step?
    time_step!(model, 1)
    @test true

    # Once more for good measure 
    time_step!(model, 1)
    @test true

    # Return model if we want to do more tests
    return model
end

function compute_closure_specific_diffusive_cfl(closure)
    grid = RectilinearGrid(CPU(), size=(1, 1, 1), extent=(1, 2, 3))

    model = NonhydrostaticModel(; grid, closure)
    dcfl = DiffusiveCFL(0.1)
    @test dcfl(model) isa Number

    tracerless_model = NonhydrostaticModel(; grid, closure, buoyancy=nothing, tracers=nothing)
    dcfl = DiffusiveCFL(0.2)
    @test dcfl(tracerless_model) isa Number

    return nothing
end

@testset "Turbulence closures" begin
    @info "Testing turbulence closures..."

    @testset "Closure instantiation" begin
        @info "  Testing closure instantiation..."
        for closure in closures
            @test closure_instantiation(closure)
        end
    end

    @testset "ScalarDiffusivity" begin
        @info "  Testing ScalarDiffusivity..."
        for T in float_types
            @test constant_isotropic_diffusivity_basic(T)
            run_constant_isotropic_diffusivity_fluxdiv_tests(T)
        end
    end

    @testset "HorizontalScalarDiffusivity" begin
        @info "  Testing HorizontalScalarDiffusivity..."
        for T in float_types
            @test tracer_specific_horizontal_diffusivity(T)
            @test horizontal_diffusivity_fluxdiv(T, νz=zero(T), νh=zero(T))
            @test horizontal_diffusivity_fluxdiv(T)
        end
    end

    @testset "Time-stepping with variable diffusivities" begin
        @info "  Testing time-stepping with presribed variable diffusivities..."
        for arch in archs
            @test time_step_with_variable_isotropic_diffusivity(arch)
            @test time_step_with_variable_anisotropic_diffusivity(arch)
        end
    end

    @testset "Time-stepping with CATKE closure" begin
        @info "  Testing time-stepping with CATKE closure and closure tuples with CATKE..."
        for arch in archs
            warning = false

            @info "    Testing time-stepping CATKE by itself..."
            closure = CATKEVerticalDiffusivity(; warning)
            run_time_step_with_catke_tests(arch, closure)

            @info "    Testing time-stepping CATKE in a 2-tuple with HorizontalScalarDiffusivity..."
            closure = (CATKEVerticalDiffusivity(; warning), HorizontalScalarDiffusivity())
            model = run_time_step_with_catke_tests(arch, closure)
            @test first(model.closure) === closure[1]

            # Test that closure tuples with CATKE are correctly reordered
            @info "    Testing time-stepping CATKE in a 2-tuple with HorizontalScalarDiffusivity..."
            closure = (HorizontalScalarDiffusivity(), CATKEVerticalDiffusivity(; warning))
            model = run_time_step_with_catke_tests(arch, closure)
            @test first(model.closure) === closure[2]

            # These are slow to compile...
            @info "    Testing time-stepping CATKE in a 3-tuple..."
            closure = (HorizontalScalarDiffusivity(), CATKEVerticalDiffusivity(; warning), VerticalScalarDiffusivity())
            model = run_time_step_with_catke_tests(arch, closure)
            @test first(model.closure) === closure[2]
        end
    end

    @testset "Closure tuples" begin
        @info "  Testing time-stepping with a tuple of closures..."
        for arch in archs
            for FT in float_types
                @test time_step_with_tupled_closure(FT, arch)
            end
        end
    end

    @testset "Diagnostics" begin
        @info "  Testing turbulence closure diagnostics..."
        for closurename in closures
            closure = getproperty(TurbulenceClosures, closurename)()
            compute_closure_specific_diffusive_cfl(closure)
        end

        # now test also a case for a tuple of closures
        compute_closure_specific_diffusive_cfl((ScalarDiffusivity(), ScalarBiharmonicDiffusivity()))
    end
end
