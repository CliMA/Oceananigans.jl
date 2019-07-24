function test_closure_instantiation(T, closurename)
    closure = getproperty(TurbulenceClosures, closurename)(T)
    return eltype(closure) == T
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

function test_constant_isotropic_diffusivity_basic(T=Float64; ν=T(0.3), κ=T(0.7))
    closure = ConstantIsotropicDiffusivity(T, κ=κ, ν=ν)
    return closure.ν == ν && closure.κ == κ
end

function test_tensor_diffusivity_tuples(T=Float64; ν=T(0.3), κ=T(0.7))
    closure = ConstantIsotropicDiffusivity(T, κ=κ, ν=ν)
    return (
            κ₁₁.ccc(nothing, nothing, nothing, nothing, closure) == κ &&
            κ₂₂.ccc(nothing, nothing, nothing, nothing, closure) == κ &&
            κ₃₃.ccc(nothing, nothing, nothing, nothing, closure) == κ &&

            ν₁₁.ccc(nothing, nothing, nothing, nothing, closure) == ν &&
            ν₂₂.ccc(nothing, nothing, nothing, nothing, closure) == ν &&
            ν₃₃.ccc(nothing, nothing, nothing, nothing, closure) == ν &&

            ν₁₁.ffc(nothing, nothing, nothing, nothing, closure) == ν &&
            ν₂₂.ffc(nothing, nothing, nothing, nothing, closure) == ν &&
            ν₃₃.ffc(nothing, nothing, nothing, nothing, closure) == ν &&

            ν₁₁.fcf(nothing, nothing, nothing, nothing, closure) == ν &&
            ν₂₂.fcf(nothing, nothing, nothing, nothing, closure) == ν &&
            ν₃₃.fcf(nothing, nothing, nothing, nothing, closure) == ν &&

            ν₁₁.cff(nothing, nothing, nothing, nothing, closure) == ν &&
            ν₂₂.cff(nothing, nothing, nothing, nothing, closure) == ν &&
            ν₃₃.cff(nothing, nothing, nothing, nothing, closure) == ν
    )
end

datatuple(args, names) = NamedTuple{names}(a.data for a in args)

function test_constant_isotropic_diffusivity_fluxdiv(FT=Float64; ν=FT(0.3), κ=FT(0.7))
    arch = CPU()
    closure = ConstantIsotropicDiffusivity(FT, κ=κ, ν=ν)
    grid = RegularCartesianGrid(FT, (3, 1, 1), (3, 1, 1))
    diffusivities = TurbulentDiffusivities(arch, grid, closure)
    fbcs = DoublyPeriodicBCs()
    eos = LinearEquationOfState()
    grav = 1.0

    u = FaceFieldX(FT, arch, grid)
    v = FaceFieldY(FT, arch, grid)
    w = FaceFieldZ(FT, arch, grid)
    T =  CellField(FT, arch, grid)
    S =  CellField(FT, arch, grid)

    u_ft = (:u, fbcs, u.data)
    v_ft = (:v, fbcs, v.data)
    w_ft = (:w, fbcs, w.data)
    T_ft = (:T, fbcs, T.data)
    S_ft = (:S, fbcs, S.data)
    uvwTS_ft = (u_ft, v_ft, w_ft, T_ft, S_ft)

    data(u)[:, 1, 1] .= [0, -1, 0]
    data(v)[:, 1, 1] .= [0, -2, 0]
    data(w)[:, 1, 1] .= [0, -3, 0]
    data(T)[:, 1, 1] .= [0, -1, 0]

    U = datatuple((u, v, w), (:u, :v, :w))
    Φ = datatuple((T, S), (:T, :S))
    fill_halo_regions!(grid, uvwTS_ft...)
    calc_diffusivities!(diffusivities, grid, closure, eos, grav, U, Φ)

    return (   ∇_κ_∇c(2, 1, 1, grid, T.data, closure) == 2κ &&
            ∂ⱼ_2ν_Σ₁ⱼ(2, 1, 1, grid, closure, U...) == 2ν &&
            ∂ⱼ_2ν_Σ₂ⱼ(2, 1, 1, grid, closure, U...) == 4ν &&
            ∂ⱼ_2ν_Σ₃ⱼ(2, 1, 1, grid, closure, U...) == 6ν
            )
end

function test_calc_diffusivities(arch, closurename, FT=Float64)

    closure = getproperty(TurbulenceClosures, closurename)(FT)
    grid = RegularCartesianGrid(FT, (3, 1, 3), (3, 1, 3))
    diffusivities = TurbulentDiffusivities(arch, grid, closure)
    eos = LinearEquationOfState(FT)
    grav = 1.0
    u = FaceFieldX(FT, arch, grid)
    v = FaceFieldY(FT, arch, grid)
    w = FaceFieldZ(FT, arch, grid)
    T = CellField(FT, arch, grid)
    S = CellField(FT, arch, grid)

    U = datatuple((u, v, w), (:u, :v, :w))
    Φ = datatuple((T, S), (:T, :S))
    calc_diffusivities!(diffusivities, grid, closure, eos, grav, U, Φ)

    return true
end

function test_anisotropic_diffusivity_fluxdiv(FT=Float64; νh=FT(0.3), κh=FT(0.7), νv=FT(0.1), κv=FT(0.5))
    closure = ConstantAnisotropicDiffusivity(FT, κh=κh, νh=νh, κv=κv, νv=νv)
    grid = RegularCartesianGrid(FT, (3, 1, 3), (3, 1, 3))
    fbcs = DoublyPeriodicBCs()
    eos = LinearEquationOfState()
    g = 1.0

    arch = CPU()
    u = FaceFieldX(FT, arch, grid)
    v = FaceFieldY(FT, arch, grid)
    w = FaceFieldZ(FT, arch, grid)
    T =  CellField(FT, arch, grid)
    S =  CellField(FT, arch, grid)

    u_ft = (:u, fbcs, u.data)
    v_ft = (:v, fbcs, v.data)
    w_ft = (:w, fbcs, w.data)
    T_ft = (:T, fbcs, T.data)
    S_ft = (:S, fbcs, S.data)
    uvwTS_ft = (u_ft, v_ft, w_ft, T_ft, S_ft)

    data(u)[:, 1, 1] .= [0,  1, 0]
    data(u)[:, 1, 2] .= [0, -1, 0]
    data(u)[:, 1, 3] .= [0,  1, 0]

    data(v)[:, 1, 1] .= [0,  1, 0]
    data(v)[:, 1, 2] .= [0, -2, 0]
    data(v)[:, 1, 3] .= [0,  1, 0]

    data(w)[:, 1, 1] .= [0,  1, 0]
    data(w)[:, 1, 2] .= [0, -3, 0]
    data(w)[:, 1, 3] .= [0,  1, 0]

    data(T)[:, 1, 1] .= [0,  1, 0]
    data(T)[:, 1, 2] .= [0, -4, 0]
    data(T)[:, 1, 3] .= [0,  1, 0]

    fill_halo_regions!(grid, uvwTS_ft...)

    return (∇_κ_∇c(2, 1, 2, grid, T.data, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 8κh + 10κv &&
            ∂ⱼ_2ν_Σ₁ⱼ(2, 1, 2, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 2νh + 4νv &&
            ∂ⱼ_2ν_Σ₂ⱼ(2, 1, 2, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 4νh + 6νv &&
            ∂ⱼ_2ν_Σ₃ⱼ(2, 1, 2, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 6νh + 8νv
            )
end

function test_smag_divflux_finiteness(FT=Float64)
    arch = CPU()
    closure = ConstantSmagorinsky(FT)
    grid = RegularCartesianGrid(FT, (4, 4, 4), (4, 4, 4))
    fbcs = DoublyPeriodicBCs()
    eos = LinearEquationOfState(FT)
    diffusivities = TurbulentDiffusivities(arch, grid, closure)
    grav = FT(1.0)

    u = FaceFieldX(FT, arch, grid)
    v = FaceFieldY(FT, arch, grid)
    w = FaceFieldZ(FT, arch, grid)
    T =  CellField(FT, arch, grid)
    S =  CellField(FT, arch, grid)

    underlying_data(u) .= 0
    underlying_data(v) .= 0
    underlying_data(w) .= 0
    underlying_data(T) .= eos.T₀
    underlying_data(S) .= eos.S₀

    U = datatuple((u, v, w), (:u, :v, :w))
    Φ = datatuple((T, S), (:T, :S))
    calc_diffusivities!(diffusivities, grid, closure, eos, grav, U, Φ)

    u_ft = (:u, fbcs, u.data)
    v_ft = (:v, fbcs, v.data)
    w_ft = (:w, fbcs, w.data)
    T_ft = (:T, fbcs, T.data)
    S_ft = (:S, fbcs, S.data)
    uvwTS_ft = (u_ft, v_ft, w_ft, T_ft, S_ft)

    fill_halo_regions!(grid, uvwTS_ft...)

    return (isfinite(∇_κ_∇c(2, 1, 2, grid, T.data, closure, diffusivities)) &&
        isfinite(∂ⱼ_2ν_Σ₁ⱼ(2, 1, 2, grid, closure, u.data, v.data, w.data, diffusivities)) &&
        isfinite(∂ⱼ_2ν_Σ₂ⱼ(2, 1, 2, grid, closure, u.data, v.data, w.data, diffusivities)) &&
        isfinite(∂ⱼ_2ν_Σ₃ⱼ(2, 1, 2, grid, closure, u.data, v.data, w.data, diffusivities))
        )
end

@testset "Turbulence closures" begin
    println("Testing turbulence closures...")

    @testset "Closure operators" begin
        println("  Testing closure operators...")
        @test test_function_interpolation()
        @test test_function_differentiation()
    end

    @testset "Closure instantiation and basic operation" begin
        println("  Testing closure instantiation...")
        for T in float_types
            for closure in (:ConstantIsotropicDiffusivity,
                            :ConstantAnisotropicDiffusivity,
                            :ConstantSmagorinsky,
                            :AnisotropicMinimumDissipation)

                @test test_closure_instantiation(T, closure)
                for arch in archs
                    @test test_calc_diffusivities(arch, closure, T)
                end
            end
        end
    end

    @testset "Constant isotropic diffusivity" begin
        println("  Testing constant isotropic diffusivity...")
        for T in float_types
            @test test_constant_isotropic_diffusivity_basic(T)
            @test test_tensor_diffusivity_tuples(T)
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

    @testset "Constant Smagorinsky" begin
        println("  Testing constant Smagorinsky...")
        for T in float_types
            @test_skip test_smag_divflux_finiteness(T)
        end
    end
end
