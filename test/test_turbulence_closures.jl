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

function test_constant_isotropic_diffusivity_fluxdiv(TF=Float64; ν=TF(0.3), κ=TF(0.7))
    closure = ConstantIsotropicDiffusivity(TF, κ=κ, ν=ν)
    grid = RegularCartesianGrid(TF, (3, 1, 1), (3, 1, 1))
    eos = LinearEquationOfState()
    g = 1.0

    arch = CPU()
    u = FaceFieldX(TF, arch, grid)
    v = FaceFieldY(TF, arch, grid)
    w = FaceFieldZ(TF, arch, grid)
    T =  CellField(TF, arch, grid)
    S =  CellField(TF, arch, grid)

    data(u)[:, 1, 1] .= [0, -1, 0]
    data(v)[:, 1, 1] .= [0, -2, 0]
    data(w)[:, 1, 1] .= [0, -3, 0]
    data(T)[:, 1, 1] .= [0, -1, 0]

    fill_halo_regions!(arch, grid, u.data, v.data, w.data, T.data, S.data)

    return (∇_κ_∇ϕ(2, 1, 1, grid, T.data, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 2κ &&
            ∂ⱼ_2ν_Σ₁ⱼ(2, 1, 1, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 2ν &&
            ∂ⱼ_2ν_Σ₂ⱼ(2, 1, 1, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 4ν &&
            ∂ⱼ_2ν_Σ₃ⱼ(2, 1, 1, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 6ν
            )
end

function test_anisotropic_diffusivity_fluxdiv(TF=Float64; νh=TF(0.3), κh=TF(0.7), νv=TF(0.1), κv=TF(0.5))
    closure = ConstantAnisotropicDiffusivity(TF, κh=κh, νh=νh, κv=κv, νv=νv)
    grid = RegularCartesianGrid(TF, (3, 1, 3), (3, 1, 3))
    eos = LinearEquationOfState()
    g = 1.0

    arch = CPU()
    u = FaceFieldX(TF, arch, grid)
    v = FaceFieldY(TF, arch, grid)
    w = FaceFieldZ(TF, arch, grid)
    T =  CellField(TF, arch, grid)
    S =  CellField(TF, arch, grid)

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

    fill_halo_regions!(arch, grid, u.data, v.data, w.data, T.data, S.data)

    return (∇_κ_∇ϕ(2, 1, 2, grid, T.data, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 8κh + 10κv &&
            ∂ⱼ_2ν_Σ₁ⱼ(2, 1, 2, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 2νh + 4νv &&
            ∂ⱼ_2ν_Σ₂ⱼ(2, 1, 2, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 4νh + 6νv &&
            ∂ⱼ_2ν_Σ₃ⱼ(2, 1, 2, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data) == 6νh + 8νv
            )
end

function test_smag_divflux_finiteness(TF=Float64)
    closure = ConstantSmagorinsky(TF)
    grid = RegularCartesianGrid(TF, (3, 3, 3), (3, 3, 3))
    eos = LinearEquationOfState()
    g = 1.0

    arch = CPU()
    u = FaceFieldX(TF, arch, grid)
    v = FaceFieldY(TF, arch, grid)
    w = FaceFieldZ(TF, arch, grid)
    T =  CellField(TF, arch, grid)
    S =  CellField(TF, arch, grid)

    fill_halo_regions!(arch, grid, u.data, v.data, w.data, T.data, S.data)

    return (
        isfinite(∇_κ_∇ϕ(2, 1, 2, grid, T.data, closure, eos, g, u.data, v.data, w.data, T.data, S.data)) &&
        isfinite(∂ⱼ_2ν_Σ₁ⱼ(2, 1, 2, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data)) &&
        isfinite(∂ⱼ_2ν_Σ₂ⱼ(2, 1, 2, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data)) &&
        isfinite(∂ⱼ_2ν_Σ₃ⱼ(2, 1, 2, grid, closure, eos, g, u.data, v.data, w.data, T.data, S.data))
        )
end
