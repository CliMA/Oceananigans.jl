include("dependencies_for_runtests.jl")

using Random
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, RiBasedVerticalDiffusivity, DiscreteDiffusionFunction

using Oceananigans.TurbulenceClosures: viscosity_location, diffusivity_location,
                                       required_halo_size_x, required_halo_size_y, required_halo_size_z,
                                       cell_diffusion_timescale, formulation, min_Δxyz

using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z,
                                       viscous_flux_ux, viscous_flux_uy, viscous_flux_uz

using Oceananigans.TurbulenceClosures: ScalarDiffusivity,
                                       ScalarBiharmonicDiffusivity,
                                       TwoDimensionalLeith,
                                       ConvectiveAdjustmentVerticalDiffusivity,
                                       Smagorinsky,
                                       DynamicSmagorinsky,
                                       SmagorinskyLilly,
                                       LagrangianAveraging,
                                       AnisotropicMinimumDissipation

using Oceananigans.Grids: znode

ConstantSmagorinsky(FT=Float64) = Smagorinsky(FT, coefficient=0.16)
DirectionallyAveragedDynamicSmagorinsky(FT=Float64) = DynamicSmagorinsky(FT, averaging=(1, 2))
LagrangianAveragedDynamicSmagorinsky(FT=Float64) = DynamicSmagorinsky(FT, averaging=LagrangianAveraging())

function tracer_specific_horizontal_diffusivity(T=Float64; νh=T(0.3), κh=T(0.7))
    closure = HorizontalScalarDiffusivity(κ=(T=κh, S=κh), ν=νh)
    return closure.ν == νh && closure.κ.T == κh && closure.κ.T == κh
end

function run_constant_isotropic_diffusivity_fluxdiv_tests(FT=Float64; ν=FT(0.3), κ=FT(0.7))
    arch       = CPU()
    closure    = ScalarDiffusivity(FT, κ=(T=κ, S=κ), ν=ν)
    grid       = RectilinearGrid(FT, size=(3, 1, 4), extent=(3, 1, 4))
    velocities = VelocityFields(grid)
    tracers    = TracerFields((:T, :S), grid)
    clock      = Clock(time=0.0)

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

    K, b = nothing, nothing
    closure_args = (clock, model_fields, b)

    @test ∇_dot_qᶜ(2, 1, 3, grid, closure, K, Val(1), tracers[1], closure_args...) == - 2κ
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

    K, b = nothing, nothing
    closure_args = (clock, model_fields, b)

    return (∇_dot_qᶜ(2, 1, 3, grid, closureh, K, Val(1), T, closure_args...) == -  8κh &&
            ∇_dot_qᶜ(2, 1, 3, grid, closurez, K, Val(1), T, closure_args...) == - 10κz &&
              ∂ⱼ_τ₁ⱼ(2, 1, 3, grid, closureh, K, closure_args...) == - 2νh &&
              ∂ⱼ_τ₁ⱼ(2, 1, 3, grid, closurez, K, closure_args...) == - 4νz &&
              ∂ⱼ_τ₂ⱼ(2, 1, 3, grid, closureh, K, closure_args...) == - 4νh &&
              ∂ⱼ_τ₂ⱼ(2, 1, 3, grid, closurez, K, closure_args...) == - 6νz &&
              ∂ⱼ_τ₃ⱼ(2, 1, 3, grid, closureh, K, closure_args...) == - 6νh &&
              ∂ⱼ_τ₃ⱼ(2, 1, 3, grid, closurez, K, closure_args...) == - 8νz)
end

function time_step_with_variable_isotropic_diffusivity(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 2, 3))
    closure = ScalarDiffusivity(ν = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t),
                                κ = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t))

    model = NonhydrostaticModel(grid; closure)
    time_step!(model, 1)
    return true
end

function time_step_with_field_isotropic_diffusivity(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 2, 3))
    ν = CenterField(grid)
    κ = CenterField(grid)
    closure = ScalarDiffusivity(; ν, κ)
    model = NonhydrostaticModel(grid; closure)
    time_step!(model, 1)
    return true
end

function time_step_with_variable_anisotropic_diffusivity(arch)
    clov = VerticalScalarDiffusivity(ν = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t),
                                     κ = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t))

    cloh = HorizontalScalarDiffusivity(ν = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t),
                                       κ = (x, y, z, t) -> exp(z) * cos(x) * cos(y) * cos(t))
    for clo in (clov, cloh)
        grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 2, 3))
        model = NonhydrostaticModel(grid, closure=clo)
        time_step!(model, 1)
    end

    return true
end

function time_step_with_variable_discrete_diffusivity(arch)
    @inline νd(i, j, k, grid, clock, fields) = 1 + fields.u[i, j, k] * 5
    @inline κd(i, j, k, grid, clock, fields) = 1 + fields.v[i, j, k] * 5

    closure_ν = ScalarDiffusivity(ν = νd, discrete_form=true, loc = (Face, Center, Center))
    closure_κ = ScalarDiffusivity(κ = κd, discrete_form=true, loc = (Center, Face, Center))

    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 2, 3))
    model = NonhydrostaticModel(grid, tracers = (:T, :S), closure = (closure_ν, closure_κ))

    time_step!(model, 1)
    return true
end

function time_step_with_variable_AMD_coefficient(arch; use_field_coefficient=false)
    grid = RectilinearGrid(arch, size=(4, 5, 6), extent=(1, 2, 3))

    Cν_func(x, y, z) = exp(z) * cos(x) * cos(y)
    Cκ_func(x, y, z) = exp(z) * cos(x) * cos(y)

    if use_field_coefficient
        Cν = CenterField(grid)
        Cκ = CenterField(grid)
        set!(Cν, Cν_func)
        set!(Cκ, Cκ_func)
    else
        Cν = Cν_func
        Cκ = Cκ_func
    end

    closure = AnisotropicMinimumDissipation(; Cν, Cκ)
    model = NonhydrostaticModel(grid; closure)
    time_step!(model, 1)
    return true
end

function time_step_with_tupled_closure(FT, arch)
    closure_tuple = (AnisotropicMinimumDissipation(FT), ScalarDiffusivity(FT))
    grid = RectilinearGrid(arch, FT, size=(2, 2, 2), extent=(1, 2, 3))

    model = NonhydrostaticModel(grid; closure=closure_tuple)

    time_step!(model, 1)
    return true
end

function run_catke_tke_substepping_tests(arch, closure)
    # A large domain to make sure we do not have viscous CFL problems
    # with the explicit CATKE time-stepping necessary for this test
    grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(100, 200, 300))

    model = HydrostaticFreeSurfaceModel(grid;
                                        momentum_advection = nothing,
                                        tracer_advection = nothing,
                                        closure,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b))

    # set random velocities
    Random.seed!(1234)
    set!(model, u = (x, y, z) -> rand(), v = (x, y, z) -> rand())

    # time step the model
    time_step!(model, 1)

    # Check that eⁿ⁺¹ == Δt * Gⁿ.e with Δt = 1 (euler step)
    @test model.tracers.e ≈ model.timestepper.G⁻.e

    eⁿ  = deepcopy(model.tracers.e)
    G⁻⁻ = deepcopy(model.timestepper.G⁻.e)

    # time step the model again
    time_step!(model, 1)
    G⁻ = model.timestepper.G⁻.e

    C₁ = 1.5 + model.timestepper.χ
    C₂ = 0.5 + model.timestepper.χ

    eⁿ⁺¹ = compute!(Field(eⁿ + C₁ * G⁻ - C₂ * G⁻⁻))

    # Check that eⁿ⁺¹ == eⁿ + Δt * (C₁ Gⁿ.e - C₂ G⁻.e)
    @test model.tracers.e ≈ eⁿ⁺¹

    return model
end

function run_time_step_with_catke_tests(arch, closure, timestepper)
    grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 2, 3))
    buoyancy = BuoyancyTracer()

    @test HydrostaticFreeSurfaceModel(grid; closure, buoyancy, tracers = :b) isa HydrostaticFreeSurfaceModel
    @test HydrostaticFreeSurfaceModel(grid; closure, buoyancy, tracers = (:b, :E)) isa HydrostaticFreeSurfaceModel

    # CATKE isn't supported with NonhydrostaticModel (we don't diffuse vertical velocity)
    @test_throws ErrorException NonhydrostaticModel(grid; closure, buoyancy, tracers = (:b, :c, :e))

    # Supplying closure tracers explicitly should error
    @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; closure, buoyancy, tracers = (:b, :c, :e))

    model = HydrostaticFreeSurfaceModel(grid; closure, buoyancy, tracers = (:b, :c))

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

function compute_closure_specific_diffusive_cfl(arch, closure)
    grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 2, 3))

    model = NonhydrostaticModel(grid;  closure,  buoyancy = BuoyancyTracer(),  tracers = :b)
    args = (model.closure, model.closure_fields, Val(1), model.tracers.b, model.clock, fields(model), model.buoyancy)
    dcfl = DiffusiveCFL(0.1)
    @test dcfl(model) isa Number

    @allowscalar begin
        @test diffusive_flux_x(1, 1, 1, grid, args...) == 0
        @test diffusive_flux_y(1, 1, 1, grid, args...) == 0
        @test diffusive_flux_z(1, 1, 1, grid, args...) == 0
    end

    tracerless_model = NonhydrostaticModel(grid; closure)
    args = (model.closure, model.closure_fields, model.clock, fields(model), model.buoyancy)
    dcfl = DiffusiveCFL(0.2)
    @test dcfl(tracerless_model) isa Number
    @allowscalar begin
        @test viscous_flux_ux(1, 1, 1, grid, args...) == 0
        @test viscous_flux_uy(1, 1, 1, grid, args...) == 0
        @test viscous_flux_uz(1, 1, 1, grid, args...) == 0
    end

    return nothing
end

function test_function_scalar_diffusivity()

    depth_scale = 120
    @inline ν(x, y, z, t) = 2000 * exp(z / depth_scale)
    @inline κ(x, y, z, t) = 2000 * exp(z / depth_scale)

    closure = ScalarDiffusivity(; ν, κ)

    grid = RectilinearGrid(CPU(), size=(2, 2, 2), extent=(1, 2, 3))
    model = NonhydrostaticModel(grid; closure, tracers = :b, buoyancy = BuoyancyTracer())
    max_diffusivity = maximum(2000 * exp.(znodes(model.grid, Center()) / depth_scale))
    Δ = min_Δxyz(model.grid, formulation(model.closure))

    τκ = Δ^2 / max_diffusivity
    return cell_diffusion_timescale(model) == τκ
end

function test_discrete_function_scalar_diffusivity()

    @inline function ν(i, j, k, grid, clock, fields, p)
        z = znode(i, j, k, grid, Center(), Center(), Center())
        return 2000 * exp(z / p.depth_scale_ν)
    end
    @inline function κ(i, j, k, grid, clock, fields, p)
        z = znode(i, j, k, grid, Center(), Center(), Center())
        return 2000 * exp(z / p.depth_scale_κ)
    end

    closure = ScalarDiffusivity(; ν, κ, discrete_form=true,
                                  loc=(Center, Center, Center),
                                  parameters = (;depth_scale_ν = 100, depth_scale_κ = 100))

    grid = RectilinearGrid(CPU(), size=(2, 2, 2), extent=(1, 2, 3))
    model = NonhydrostaticModel(grid; closure, tracers = :b, buoyancy = BuoyancyTracer())
    max_diffusivity = maximum(2000 * exp.(znodes(model.grid, Center()) / 100))
    Δ = min_Δxyz(model.grid, formulation(model.closure))
    τκ = Δ^2 / max_diffusivity
    return cell_diffusion_timescale(model) == τκ
end

@testset "Turbulence closures" begin
    @info "Testing turbulence closures..."

    @testset "Closure instantiation" begin
        for closurename in closures
            closure = @eval $closurename()
            @test closure isa TurbulenceClosures.AbstractTurbulenceClosure

            for arch in archs
                @info "  Testing the instantiation of NonhydrostaticModel with $closurename on $arch..."
                grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 2, 3))
                model = NonhydrostaticModel(grid; closure, tracers = :c)
                c = model.tracers.c
                u = model.velocities.u

                κ = diffusivity(model.closure, model.closure_fields, Val(:c))
                @test diffusivity(model, Val(:c)) == diffusivity(model.closure, model.closure_fields, Val(:c))
                κ_dx_c = κ * ∂x(c)

                ν = viscosity(model.closure, model.closure_fields)
                @test viscosity(model) == viscosity(model.closure, model.closure_fields)
                ν_dx_u = ν * ∂x(u)
                @test ν_dx_u[1, 1, 1] == 0
                @test κ_dx_c[1, 1, 1] == 0
            end
        end

        c = Center()
        f = Face()
        ri_based = RiBasedVerticalDiffusivity()
        @test viscosity_location(ri_based) == (c, c, f)
        @test diffusivity_location(ri_based) == (c, c, f)

        catke = CATKEVerticalDiffusivity()
        @test viscosity_location(catke) == (c, c, f)
        @test diffusivity_location(catke) == (c, c, f)
    end

    @testset "ScalarDiffusivity" begin
        @info "  Testing ScalarDiffusivity..."
        for T in float_types
            ν, κ = 0.3, 0.7
            closure = ScalarDiffusivity(T; κ=(T=κ, S=κ), ν=ν)
            @test closure.ν == T(ν)
            @test closure.κ.T == T(κ)
            run_constant_isotropic_diffusivity_fluxdiv_tests(T)
        end

        @info "  Testing ScalarDiffusivity with different halo requirements..."
        closure = ScalarDiffusivity(ν=0.3)
        @test required_halo_size_x(closure) == 1
        @test required_halo_size_y(closure) == 1
        @test required_halo_size_z(closure) == 1

        closure = ScalarBiharmonicDiffusivity(ν=0.3)
        @test required_halo_size_x(closure) == 2
        @test required_halo_size_y(closure) == 2
        @test required_halo_size_z(closure) == 2

        @inline ν(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields) = ℑxᶠᵃᵃ(i, j, k, grid, ℑxᶜᵃᵃ, fields.u)
        closure = ScalarDiffusivity(; ν, discrete_form=true, required_halo_size=2)

        @test closure.ν isa DiscreteDiffusionFunction
        @test required_halo_size_x(closure) == 2
        @test required_halo_size_y(closure) == 2
        @test required_halo_size_z(closure) == 2

        @info "  Testing cell_diffusion_timescale for ScalarDiffusivity with FunctionDiffusion"
        @test test_function_scalar_diffusivity()
        @test test_discrete_function_scalar_diffusivity()
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
        @info "  Testing time-stepping with prescribed variable diffusivities..."
        for arch in archs
            @test time_step_with_variable_isotropic_diffusivity(arch)
            @test time_step_with_field_isotropic_diffusivity(arch)
            @test time_step_with_variable_anisotropic_diffusivity(arch)
            @test time_step_with_variable_discrete_diffusivity(arch)
        end
    end

    @testset "AnisotropicMinimumDissipation with variable coefficients" begin
        @info "  Testing AnisotropicMinimumDissipation time stepping with variable coefficients..."
        for arch in archs
            @test time_step_with_variable_AMD_coefficient(arch, use_field_coefficient=false)
            @test time_step_with_variable_AMD_coefficient(arch, use_field_coefficient=true)
        end
    end

    @testset "Dynamic Smagorinsky closures" begin
        @info "  Testing that dynamic Smagorinsky closures produce diffusivity fields of correct sizes..."
        for arch in archs
            grid = RectilinearGrid(arch, size=(2, 3, 4), extent=(1, 2, 3))

            closure = Smagorinsky(coefficient=DynamicCoefficient(averaging=1))
            model = NonhydrostaticModel(grid; closure)
            @test size(model.closure_fields.𝒥ᴸᴹ) == (1, grid.Ny, grid.Nz)
            @test size(model.closure_fields.𝒥ᴹᴹ) == (1, grid.Ny, grid.Nz)
            @test size(model.closure_fields.LM)  == size(grid)
            @test size(model.closure_fields.MM)  == size(grid)
            @test size(model.closure_fields.Σ)   == size(grid)
            @test size(model.closure_fields.Σ̄)   == size(grid)

            closure = DynamicSmagorinsky(averaging=(1, 2))
            model = NonhydrostaticModel(grid; closure)
            @test size(model.closure_fields.𝒥ᴸᴹ) == (1, 1, grid.Nz)
            @test size(model.closure_fields.𝒥ᴹᴹ) == (1, 1, grid.Nz)

            closure = DynamicSmagorinsky(averaging=(2, 3))
            model = NonhydrostaticModel(grid; closure)
            @test size(model.closure_fields.𝒥ᴸᴹ) == (grid.Nx, 1, 1)
            @test size(model.closure_fields.𝒥ᴹᴹ) == (grid.Nx, 1, 1)

            closure = DynamicSmagorinsky(averaging=Colon())
            model = NonhydrostaticModel(grid; closure)
            @test size(model.closure_fields.𝒥ᴸᴹ) == (1, 1, 1)
            @test size(model.closure_fields.𝒥ᴹᴹ) == (1, 1, 1)

            closure = DynamicSmagorinsky(averaging=LagrangianAveraging())
            model = NonhydrostaticModel(grid; closure)
            @test size(model.closure_fields.𝒥ᴸᴹ)  == size(grid)
            @test size(model.closure_fields.𝒥ᴹᴹ)  == size(grid)
            @test size(model.closure_fields.𝒥ᴸᴹ⁻) == size(grid)
            @test size(model.closure_fields.𝒥ᴹᴹ⁻) == size(grid)
            @test size(model.closure_fields.Σ)    == size(grid)
            @test size(model.closure_fields.Σ̄)    == size(grid)
        end
    end

    @testset "Time-stepping with CATKE closure" begin
        @info "  Testing time-stepping with CATKE closure and closure tuples with CATKE..."
        for arch in archs
            @info "    Testing time-stepping CATKE by itself..."
            catke = CATKEVerticalDiffusivity()
            explicit_catke = CATKEVerticalDiffusivity(ExplicitTimeDiscretization())

            for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
                run_time_step_with_catke_tests(arch, catke, timestepper)
            end

            run_catke_tke_substepping_tests(arch, explicit_catke)

            @info "    Testing time-stepping CATKE in a 2-tuple with HorizontalScalarDiffusivity..."
            closure = (catke, HorizontalScalarDiffusivity())
            model = run_time_step_with_catke_tests(arch, closure, :QuasiAdamsBashforth2)
            @test first(model.closure) === closure[1]
            closure = (explicit_catke, HorizontalScalarDiffusivity())
            run_catke_tke_substepping_tests(arch, closure)


            # Test that closure tuples with CATKE are correctly reordered
            @info "    Testing time-stepping CATKE in a 2-tuple with HorizontalScalarDiffusivity..."
            closure = (HorizontalScalarDiffusivity(), catke)
            model = run_time_step_with_catke_tests(arch, closure, :QuasiAdamsBashforth2)
            @test first(model.closure) === closure[2]
            closure = (HorizontalScalarDiffusivity(), explicit_catke)
            run_catke_tke_substepping_tests(arch, closure)

            # These are slow to compile...
            @info "    Testing time-stepping CATKE in a 3-tuple..."
            closure = (HorizontalScalarDiffusivity(), catke, VerticalScalarDiffusivity())
            model = run_time_step_with_catke_tests(arch, closure, :QuasiAdamsBashforth2)
            @test first(model.closure) === closure[2]
            closure = (HorizontalScalarDiffusivity(), explicit_catke, VerticalScalarDiffusivity())
            run_catke_tke_substepping_tests(arch, closure)
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
        for arch in archs
            @info "  Testing turbulence closure diagnostics..."
            for closurename in closures
                @info "    Testing turbulence closure diagnostics for $closurename on $arch"
                closure = @eval $closurename()
                compute_closure_specific_diffusive_cfl(arch, closure)
            end

            # now test also a case for a tuple of closures
            @info "    Testing turbulence closure diagnostics for a Tuple closure on $arch"
            compute_closure_specific_diffusive_cfl(arch, (ScalarDiffusivity(),
                                                          ScalarBiharmonicDiffusivity(),
                                                          SmagorinskyLilly(),
                                                          AnisotropicMinimumDissipation()))
        end
    end

    @testset "Vertical mixing closures with various buoyancy models" begin
        grid = RectilinearGrid(CPU(), size=(4, 4, 4), extent=(1, 1, 1))

        buoyancy_configs = [
            ("BuoyancyTracer",       BuoyancyTracer(),                                                   :b,       (b = (x, y, z) -> z,)),
            ("LinearEquationOfState", SeawaterBuoyancy(equation_of_state=LinearEquationOfState()),        (:T, :S), (T = (x, y, z) -> 20 + z, S = (x, y, z) -> 35 - z)),
            ("TEOS10EquationOfState", SeawaterBuoyancy(equation_of_state=SeawaterPolynomials.TEOS10EquationOfState()), (:T, :S), (T = (x, y, z) -> 20 + z, S = (x, y, z) -> 35 - z)),
        ]

        closures = [
            ("RiBasedVerticalDiffusivity", RiBasedVerticalDiffusivity(warning=false)),
            ("CATKEVerticalDiffusivity",   CATKEVerticalDiffusivity()),
        ]

        for (closure_name, closure) in closures
            @testset "$closure_name" begin
                @info "  Testing $closure_name with different buoyancy formulations..."
                for (buoyancy_name, buoyancy, tracers, initial_conditions) in buoyancy_configs
                    @testset "$buoyancy_name" begin
                        @info "    Testing $closure_name with $buoyancy_name..."
                        model = HydrostaticFreeSurfaceModel(grid; closure, buoyancy, tracers)
                        set!(model; u = (x, y, z) -> z, initial_conditions...)
                        time_step!(model, 1)
                        @test model isa HydrostaticFreeSurfaceModel
                    end
                end
            end
        end
    end
end
