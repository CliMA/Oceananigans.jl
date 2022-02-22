include("dependencies_for_runtests.jl")

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity
using Oceananigans.Grids: ZDirection, XYDirections

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

function anisotropic_diffusivity_convenience_kwarg(T=Float64; νh=T(0.3), κh=T(0.7))
    closure = ScalarDiffusivity(κ=(T=κh, S=κh), ν=νh, isotropy=XYDirections())
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
    fill_halo_regions!(merge(velocities, tracers), arch, nothing, model_fields)

    U, C = datatuples(velocities, tracers)

    @test ∇_dot_qᶜ(2, 1, 3, grid, closure, C.T, Val(1), clock, nothing) == - 2κ
    @test ∂ⱼ_τ₁ⱼ(2, 1, 3, grid, closure, clock, U, nothing) == - 2ν
    @test ∂ⱼ_τ₂ⱼ(2, 1, 3, grid, closure, clock, U, nothing) == - 4ν
    @test ∂ⱼ_τ₃ⱼ(2, 1, 3, grid, closure, clock, U, nothing) == - 6ν

    return nothing
end

function anisotropic_diffusivity_fluxdiv(FT=Float64; νh=FT(0.3), κh=FT(0.7), νz=FT(0.1), κz=FT(0.5))
          arch = CPU()
      closureh = ScalarDiffusivity(FT, ν=νh, κ=(T=κh, S=κh), isotropy=XYDirections())
      closurez = ScalarDiffusivity(FT, ν=νz, κ=(T=κz, S=κz), isotropy=ZDirection())
          grid = RectilinearGrid(arch, FT, size=(3, 1, 4), extent=(3, 1, 4))
           eos = LinearEquationOfState(FT)
      buoyancy = SeawaterBuoyancy(FT, gravitational_acceleration=1, equation_of_state=eos)
    velocities = VelocityFields(grid)
       tracers = TracerFields((:T, :S), grid)
         clock = Clock(time=0.0)

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
    fill_halo_regions!(merge(velocities, tracers), arch, nothing, model_fields)

    U, C = datatuples(velocities, tracers)

    return (∇_dot_qᶜ(2, 1, 3, grid, closureh, C.T, Val(1), clock, nothing) == -  8κh &&
            ∇_dot_qᶜ(2, 1, 3, grid, closurez, C.T, Val(1), clock, nothing) == - 10κz &&
              ∂ⱼ_τ₁ⱼ(2, 1, 3, grid, closureh, clock, U, nothing) == - (2νh) &&
              ∂ⱼ_τ₁ⱼ(2, 1, 3, grid, closurez, clock, U, nothing) == - (4νz) &&
              ∂ⱼ_τ₂ⱼ(2, 1, 3, grid, closureh, clock, U, nothing) == - (4νh) &&
              ∂ⱼ_τ₂ⱼ(2, 1, 3, grid, closurez, clock, U, nothing) == - (6νz) &&
              ∂ⱼ_τ₃ⱼ(2, 1, 3, grid, closureh, clock, U, nothing) == - (6νh) &&
              ∂ⱼ_τ₃ⱼ(2, 1, 3, grid, closurez, clock, U, nothing) == - (8νz))
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

    for dir in (XYDirections(), ZDirection())
        closure = ScalarDiffusivity(ν = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t),
                                    κ = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t),
                                    isotropy = dir)
        model = NonhydrostaticModel(grid=RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 2, 3)), closure=closure)

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
    @test_throws ErrorException NonhydrostaticModel(; grid, closure, buoyancy, tracers=(:b, :e))

    model = HydrostaticFreeSurfaceModel(; grid, closure, buoyancy, tracers = (:b, :e))

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

function compute_closure_specific_diffusive_cfl(closurename)
    grid = RectilinearGrid(CPU(), size=(1, 1, 1), extent=(1, 2, 3))
    closure = getproperty(TurbulenceClosures, closurename)()

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

    @testset "Constant isotropic diffusivity" begin
        @info "  Testing constant isotropic diffusivity..."
        for T in float_types
            @test constant_isotropic_diffusivity_basic(T)
            run_constant_isotropic_diffusivity_fluxdiv_tests(T)
        end
    end

    @testset "Constant anisotropic diffusivity" begin
        @info "  Testing constant anisotropic diffusivity..."
        for T in float_types
            @test anisotropic_diffusivity_convenience_kwarg(T)
            @test anisotropic_diffusivity_fluxdiv(T, νz=zero(T), νh=zero(T))
            @test anisotropic_diffusivity_fluxdiv(T)
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

            @info "    Testing time-stepping CATKE in a 2-tuple with ScalarDiffusivity..."
            closure = (CATKEVerticalDiffusivity(; warning), ScalarDiffusivity())
            model = run_time_step_with_catke_tests(arch, closure)
            @test first(model.closure) === closure[1]

            # Test that closure tuples with CATKE are correctly reordered
            @info "    Testing time-stepping CATKE in a 2-tuple with ScalarDiffusivity..."
            closure = (ScalarDiffusivity(), CATKEVerticalDiffusivity(; warning))
            model = run_time_step_with_catke_tests(arch, closure)
            @test first(model.closure) === closure[2]

            # These are slow to compile...
            @info "    Testing time-stepping CATKE in a 3-tuple..."
            closure = (ScalarDiffusivity(), CATKEVerticalDiffusivity(; warning), ScalarDiffusivity())
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
        for closure in closures
            compute_closure_specific_diffusive_cfl(closure)
        end
    end
end
