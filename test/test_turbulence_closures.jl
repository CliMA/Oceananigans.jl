using Oceananigans.TurbulenceClosures: ∂x_caa, ∂x_faa, ∂x²_caa, ∂x²_faa,
                                       ∂y_aca, ∂y_afa, ∂y²_aca, ∂y²_afa,
                                       ∂z_aac, ∂z_aaf, ∂z²_aac, ∂z²_aaf,
                                       ▶x_caa, ▶x_faa, ▶y_aca, ▶y_afa,
                                       ▶z_aac, ▶z_aaf

using GPUifyLoops: @launch
using Oceananigans: launch_config, datatuples, device
import Oceananigans: datatuple

closures = (
            :ConstantIsotropicDiffusivity,
            :ConstantAnisotropicDiffusivity,
            :SmagorinskyLilly,
            :BlasiusSmagorinsky,
            :RozemaAnisotropicMinimumDissipation,
            :VerstappenAnisotropicMinimumDissipation
           )

for closure in closures
    @eval begin
        using Oceananigans.TurbulenceClosures: $closure
    end
end

datatuple(args, names) = NamedTuple{names}(a.data for a in args)

function test_closure_instantiation(T, closurename)
    closure = getproperty(TurbulenceClosures, closurename)(T)
    return eltype(closure) == T
end

function test_calc_diffusivities(arch, closurename, FT=Float64; kwargs...)
    closure = getproperty(TurbulenceClosures, closurename)(FT; kwargs...)
    grid = RegularCartesianGrid(FT, (3, 3, 3), (3, 3, 3))
    diffusivities = TurbulentDiffusivities(arch, grid, closure)
    eos = LinearEquationOfState{FT}()
    grav = one(FT)
    velocities = Oceananigans.VelocityFields(arch, grid)
    tracers = Oceananigans.TracerFields(arch, grid)

    U, Φ, K = datatuples(velocities, tracers, diffusivities)

    @launch device(arch) config=launch_config(grid, 3) calc_diffusivities!(
        K, grid, closure, eos, grav, U, Φ)

    return true
end

function test_constant_isotropic_diffusivity_basic(T=Float64; ν=T(0.3), κ=T(0.7))
    closure = ConstantIsotropicDiffusivity(T, κ=κ, ν=ν)
    return closure.ν == ν && closure.κ == κ
end

function test_constant_isotropic_diffusivity_fluxdiv(FT=Float64;
                                                     ν=FT(0.3), κ=FT(0.7))

    arch = CPU()
    closure = ConstantIsotropicDiffusivity(FT, κ=κ, ν=ν)
    grid = RegularCartesianGrid(FT, (3, 1, 4), (3, 1, 4))
    bcs = HorizontallyPeriodicSolutionBCs()
    eos = LinearEquationOfState()
    grav = one(FT)

    velocities = Oceananigans.VelocityFields(arch, grid)
    tracers = Oceananigans.TracerFields(arch, grid)
    u, v, w = velocities
    T, S = tracers

    for k = 1:4
        data(u)[:, 1, k] .= [0, -1, 0]
        data(v)[:, 1, k] .= [0, -2, 0]
        data(w)[:, 1, k] .= [0, -3, 0]
        data(T)[:, 1, k] .= [0, -1, 0]
    end

    U, Φ = datatuples(velocities, tracers)
    fill_halo_regions!(merge(U, Φ), bcs, arch, grid)

    return (   ∇_κ_∇c(2, 1, 3, grid, Φ.T, closure) == 2κ &&
            ∂ⱼ_2ν_Σ₁ⱼ(2, 1, 3, grid, closure, U...) == 2ν &&
            ∂ⱼ_2ν_Σ₂ⱼ(2, 1, 3, grid, closure, U...) == 4ν &&
            ∂ⱼ_2ν_Σ₃ⱼ(2, 1, 3, grid, closure, U...) == 6ν
            )
end

function test_anisotropic_diffusivity_fluxdiv(FT=Float64; νh=FT(0.3), κh=FT(0.7),
                                              νv=FT(0.1), κv=FT(0.5))

    arch = CPU()
    closure = ConstantAnisotropicDiffusivity(FT, κh=κh, νh=νh, κv=κv, νv=νv)
    grid = RegularCartesianGrid(FT, (3, 1, 4), (3, 1, 4))
    bcs = HorizontallyPeriodicSolutionBCs()
    eos = LinearEquationOfState()
    grav = one(FT)
    velocities = Oceananigans.VelocityFields(arch, grid)
    tracers = Oceananigans.TracerFields(arch, grid)
    u, v, w = velocities
    T, S = tracers

    data(u)[:, 1, 2] .= [0,  1, 0]
    data(u)[:, 1, 3] .= [0, -1, 0]
    data(u)[:, 1, 4] .= [0,  1, 0]

    data(v)[:, 1, 2] .= [0,  1, 0]
    data(v)[:, 1, 3] .= [0, -2, 0]
    data(v)[:, 1, 4] .= [0,  1, 0]

    data(w)[:, 1, 2] .= [0,  1, 0]
    data(w)[:, 1, 3] .= [0, -3, 0]
    data(w)[:, 1, 4] .= [0,  1, 0]

    data(T)[:, 1, 2] .= [0,  1, 0]
    data(T)[:, 1, 3] .= [0, -4, 0]
    data(T)[:, 1, 4] .= [0,  1, 0]

    U, Φ = datatuples(velocities, tracers)
    fill_halo_regions!(merge(U, Φ), bcs, arch, grid)

    return (   ∇_κ_∇c(2, 1, 3, grid, Φ.T, closure, eos, grav, U..., Φ...) == 8κh + 10κv &&
            ∂ⱼ_2ν_Σ₁ⱼ(2, 1, 3, grid, closure, eos, grav, U..., Φ...) == 2νh + 4νv &&
            ∂ⱼ_2ν_Σ₂ⱼ(2, 1, 3, grid, closure, eos, grav, U..., Φ...) == 4νh + 6νv &&
            ∂ⱼ_2ν_Σ₃ⱼ(2, 1, 3, grid, closure, eos, grav, U..., Φ...) == 6νh + 8νv
            )
end

function test_function_interpolation(T=Float64)
    grid = RegularCartesianGrid(T, (3, 3, 3), (3, 3, 3))
    ϕ = rand(T, 3, 3, 3)
    ϕ² = ϕ.^2

    ▶x_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[1, 2, 2]) / 2
    ▶x_ϕ_c = (ϕ²[3, 2, 2] + ϕ²[2, 2, 2]) / 2

    ▶y_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[2, 1, 2]) / 2
    ▶y_ϕ_c = (ϕ²[2, 3, 2] + ϕ²[2, 2, 2]) / 2

    ▶z_ϕ_f = (ϕ²[2, 2, 2] + ϕ²[2, 2, 1]) / 2
    ▶z_ϕ_c = (ϕ²[2, 2, 3] + ϕ²[2, 2, 2]) / 2

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    return (
        ▶x_caa(2, 2, 2, grid, f, ϕ) == ▶x_ϕ_c &&
        ▶x_faa(2, 2, 2, grid, f, ϕ) == ▶x_ϕ_f &&

        ▶y_aca(2, 2, 2, grid, f, ϕ) == ▶y_ϕ_c &&
        ▶y_afa(2, 2, 2, grid, f, ϕ) == ▶y_ϕ_f &&

        ▶z_aac(2, 2, 2, grid, f, ϕ) == ▶z_ϕ_c &&
        ▶z_aaf(2, 2, 2, grid, f, ϕ) == ▶z_ϕ_f
        )
end

function test_function_differentiation(T=Float64)
    grid = RegularCartesianGrid(T, (3, 3, 3), (3, 3, 3))
    ϕ = rand(T, 3, 3, 3)
    ϕ² = ϕ.^2

    ∂x_ϕ_f = ϕ²[2, 2, 2] - ϕ²[1, 2, 2]
    ∂x_ϕ_c = ϕ²[3, 2, 2] - ϕ²[2, 2, 2]

    ∂y_ϕ_f = ϕ²[2, 2, 2] - ϕ²[2, 1, 2]
    ∂y_ϕ_c = ϕ²[2, 3, 2] - ϕ²[2, 2, 2]

    # Note reverse indexing here!
    ∂z_ϕ_f = ϕ²[2, 2, 1] - ϕ²[2, 2, 2]
    ∂z_ϕ_c = ϕ²[2, 2, 2] - ϕ²[2, 2, 3]

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    return (
        ∂x_caa(2, 2, 2, grid, f, ϕ) == ∂x_ϕ_c &&
        ∂x_faa(2, 2, 2, grid, f, ϕ) == ∂x_ϕ_f &&

        ∂y_aca(2, 2, 2, grid, f, ϕ) == ∂y_ϕ_c &&
        ∂y_afa(2, 2, 2, grid, f, ϕ) == ∂y_ϕ_f &&

        ∂z_aac(2, 2, 2, grid, f, ϕ) == ∂z_ϕ_c &&
        ∂z_aaf(2, 2, 2, grid, f, ϕ) == ∂z_ϕ_f
        )
end


@testset "Turbulence closures" begin
    println("Testing turbulence closures...")

    @testset "Closure operators" begin
        println("  Testing closure operators...")
        @test test_function_interpolation()
        @test test_function_differentiation()
    end

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
                    println("    Calculating diffusivities for $closure ($T, $arch)")
                    @test test_calc_diffusivities(arch, closure, T)
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
end
