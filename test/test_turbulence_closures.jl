for closure in closures
    @eval begin
        using Oceananigans.TurbulenceClosures: $closure
    end
end

datatuple(args, names) = NamedTuple{names}(a.data for a in args)

function test_closure_instantiation(FT, closurename)
    closure = getproperty(TurbulenceClosures, closurename)(FT)
    return eltype(closure) == FT
end

function test_calculate_diffusivities(arch, closurename, FT=Float64; kwargs...)
      tracernames = (:b,)
          closure = getproperty(TurbulenceClosures, closurename)(FT; kwargs...)
          closure = with_tracers(tracernames, closure)
             grid = RegularCartesianGrid(FT; size=(3, 3, 3), length=(3, 3, 3))
    diffusivities = TurbulentDiffusivities(arch, grid, tracernames, closure)
         buoyancy = BuoyancyTracer()
       velocities = Oceananigans.VelocityFields(arch, grid)
          tracers = Oceananigans.TracerFields(arch, grid, tracernames)

    U, C, K = datatuples(velocities, tracers, diffusivities)

    calculate_diffusivities!(K, arch, grid, closure, buoyancy, U, C)

    return true
end

function test_constant_isotropic_diffusivity_basic(T=Float64; ν=T(0.3), κ=T(0.7))
    closure = ConstantIsotropicDiffusivity(T; κ=(T=κ, S=κ), ν=ν)
    return closure.ν == ν && closure.κ.T == κ
end

function test_constant_isotropic_diffusivity_fluxdiv(FT=Float64; ν=FT(0.3), κ=FT(0.7))
          arch = CPU()
       closure = ConstantIsotropicDiffusivity(FT, κ=(T=κ, S=κ), ν=ν)
          grid = RegularCartesianGrid(FT; size=(3, 1, 4), length=(3, 1, 4))
           bcs = SolutionBoundaryConditions((:T, :S), HorizontallyPeriodicSolutionBCs())
    velocities = Oceananigans.VelocityFields(arch, grid)
       tracers = Oceananigans.TracerFields(arch, grid, (:T, :S))

    u, v, w = velocities
       T, S = tracers

    for k in 1:4
        interior(u)[:, 1, k] .= [0, -1, 0]
        interior(v)[:, 1, k] .= [0, -2, 0]
        interior(w)[:, 1, k] .= [0, -3, 0]
        interior(T)[:, 1, k] .= [0, -1, 0]
    end

    U, C = datatuples(velocities, tracers)
    fill_halo_regions!(merge(U, C), bcs, arch, grid)

    return (   ∇_κ_∇c(2, 1, 3, grid, closure, C.T, Val(1)) == 2κ &&
            ∂ⱼ_2ν_Σ₁ⱼ(2, 1, 3, grid, closure, U) == 2ν &&
            ∂ⱼ_2ν_Σ₂ⱼ(2, 1, 3, grid, closure, U) == 4ν &&
            ∂ⱼ_2ν_Σ₃ⱼ(2, 1, 3, grid, closure, U) == 6ν )
end

function test_anisotropic_diffusivity_fluxdiv(FT=Float64; νh=FT(0.3), κh=FT(0.7), νv=FT(0.1), κv=FT(0.5))
          arch = CPU()
       closure = ConstantAnisotropicDiffusivity(FT, νh=νh, νv=νv, κh=(T=κh, S=κh), κv=(T=κv, S=κv))
          grid = RegularCartesianGrid(FT; size=(3, 1, 4), length=(3, 1, 4))
           bcs = SolutionBoundaryConditions((:T, :S), HorizontallyPeriodicSolutionBCs())
      buoyancy = SeawaterBuoyancy(FT, gravitational_acceleration=1, equation_of_state=LinearEquationOfState(FT))
    velocities = Oceananigans.VelocityFields(arch, grid)
       tracers = Oceananigans.TracerFields(arch, grid, (:T, :S))

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

    U, C = datatuples(velocities, tracers)
    fill_halo_regions!(merge(U, C), bcs, arch, grid)

    return (   ∇_κ_∇c(2, 1, 3, grid, closure, C.T, Val(1)) == 8κh + 10κv &&
            ∂ⱼ_2ν_Σ₁ⱼ(2, 1, 3, grid, closure, U) == 2νh + 4νv &&
            ∂ⱼ_2ν_Σ₂ⱼ(2, 1, 3, grid, closure, U) == 4νh + 6νv &&
            ∂ⱼ_2ν_Σ₃ⱼ(2, 1, 3, grid, closure, U) == 6νh + 8νv)
end

function time_step_with_tupled_closure(FT, arch)
    closure_tuple = (AnisotropicMinimumDissipation(FT), ConstantAnisotropicDiffusivity(FT))

    model = Model(architecture=arch, float_type=FT, closure=closure_tuple,
                  grid=RegularCartesianGrid(FT; size=(16, 16, 16), length=(1, 2, 3)))

    time_step!(model, 1, 1)
    return true
end

@testset "Turbulence closures" begin
    println("Testing turbulence closures...")

    @testset "Closure instantiation" begin
        println("  Testing closure instantiation...")
        for T in float_types
            for closure in closures
                @test test_closure_instantiation(T, closure)
            end
        end
    end

    @testset "Calculation of nonlinear diffusivities" begin
        println("  Testing calculation of nonlinear diffusivities...")
        for T in float_types
            for arch in archs
                for closure in closures
                    println("    Calculating diffusivities for $closure [$T, $arch]")
                    @test test_calculate_diffusivities(arch, closure, T)
                end
            end
        end
    end

    @testset "Constant isotropic diffusivity" begin
        println("  Testing constant isotropic diffusivity...")
        for T in float_types
            @test test_constant_isotropic_diffusivity_basic(T)
            @test test_constant_isotropic_diffusivity_fluxdiv(T)
        end
    end

    @testset "Constant anisotropic diffusivity" begin
        println("  Testing constant anisotropic diffusivity...")
        for T in float_types
            @test test_anisotropic_diffusivity_fluxdiv(T, νv=zero(T), νh=zero(T))
            @test test_anisotropic_diffusivity_fluxdiv(T)
        end
    end

    @testset "Closure tuples" begin
        println("  Testing time-stepping with a tuple of closures...")
        for arch in archs
            for FT in float_types
                @test time_step_with_tupled_closure(FT, arch)
            end
        end
    end
end
