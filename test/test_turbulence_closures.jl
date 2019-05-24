function test_closure_instantiation(; closures=(
        :ConstantSmagorinsky,
        :ConstantIsotropicDiffusivity,
        :DirectionalDiffusivity)
        )

    results = Bool[]

    for closurename in closures
        closure64 = getproperty(TurbulenceClosures, closurename)()
        closure32 = getproperty(TurbulenceClosures, closurename)(Float32)

        push!(results, eltype(closure64) == Float64)
        push!(results, eltype(closure32) == Float32)
    end

    return all(results)
end

function test_function_interpolation()
    grid = RegularCartesianGrid((3, 3, 3), (3, 3, 3))
    ϕ = rand(3, 3, 3)
    ϕ² = ϕ.^2

    ▶x_ϕ_c = (ϕ²[2, 2, 2] + ϕ²[1, 2, 2]) / 2
    ▶x_ϕ_f = (ϕ²[3, 2, 2] + ϕ²[2, 2, 2]) / 2

    ▶y_ϕ_c = (ϕ²[2, 2, 2] + ϕ²[2, 1, 2]) / 2
    ▶y_ϕ_f = (ϕ²[2, 3, 2] + ϕ²[2, 2, 2]) / 2

    ▶z_ϕ_c = (ϕ²[2, 2, 2] + ϕ²[2, 2, 1]) / 2
    ▶z_ϕ_f = (ϕ²[2, 2, 3] + ϕ²[2, 2, 2]) / 2

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    return (
        TurbulenceClosures.▶x_caa(2, 2, 2, grid, f, ϕ) == ▶x_ϕ_c &&
        TurbulenceClosures.▶x_faa(2, 2, 2, grid, f, ϕ) == ▶x_ϕ_f &&

        TurbulenceClosures.▶y_aca(2, 2, 2, grid, f, ϕ) == ▶y_ϕ_c &&
        TurbulenceClosures.▶y_afa(2, 2, 2, grid, f, ϕ) == ▶y_ϕ_f &&

        TurbulenceClosures.▶z_aac(2, 2, 2, grid, f, ϕ) == ▶z_ϕ_c &&
        TurbulenceClosures.▶z_aaf(2, 2, 2, grid, f, ϕ) == ▶z_ϕ_f
        )
end

function test_function_differentiation()
    grid = RegularCartesianGrid((3, 3, 3), (3, 3, 3))
    ϕ = rand(3, 3, 3)
    ϕ² = ϕ.^2

    ∂x_ϕ_c = ϕ²[2, 2, 2] - ϕ²[1, 2, 2]
    ∂x_ϕ_f = ϕ²[3, 2, 2] - ϕ²[2, 2, 2]

    ∂y_ϕ_c = ϕ²[2, 2, 2] - ϕ²[2, 1, 2]
    ∂y_ϕ_f = ϕ²[2, 3, 2] - ϕ²[2, 2, 2]

    # Note reverse indexing here!
    ∂z_ϕ_c = ϕ²[2, 2, 1] - ϕ²[2, 2, 2]
    ∂z_ϕ_f = ϕ²[2, 2, 2] - ϕ²[2, 2, 3]

    f(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

    return (
        TurbulenceClosures.∂x_caa(2, 2, 2, grid, f, ϕ) == ∂x_ϕ_c &&
        TurbulenceClosures.∂x_faa(2, 2, 2, grid, f, ϕ) == ∂x_ϕ_f &&

        TurbulenceClosures.∂y_aca(2, 2, 2, grid, f, ϕ) == ∂y_ϕ_c &&
        TurbulenceClosures.∂y_afa(2, 2, 2, grid, f, ϕ) == ∂y_ϕ_f &&

        TurbulenceClosures.∂z_aac(2, 2, 2, grid, f, ϕ) == ∂z_ϕ_c &&
        TurbulenceClosures.∂z_aaf(2, 2, 2, grid, f, ϕ) == ∂z_ϕ_f
        )
end

function test_constant_isotropic_diffusivity_basic(; ν=0.3, κ=0.7)
    closure = TurbulenceClosures.ConstantIsotropicDiffusivity(κ=κ, ν=ν)
    return (                    closure.ν == ν &&
                                closure.κ == κ &&
        TurbulenceClosures.κ_ccc(nothing, nothing, nothing, nothing, closure) == κ
    )
end

function test_constant_isotropic_diffusivity_fluxdiv(; ν=0.3, κ=0.7)
    closure = TurbulenceClosures.ConstantIsotropicDiffusivity(κ=κ, ν=ν)
    grid = RegularCartesianGrid((3, 1, 1), (3, 1, 1))
    u = zeros(3, 1, 1); v = zeros(3, 1, 1); w = zeros(3, 1, 1)
    T = zeros(3, 1, 1); S = zeros(3, 1, 1)

    u[:, 1, 1] .= [0, -1, 0]
    v[:, 1, 1] .= [0, -2, 0]
    w[:, 1, 1] .= [0, -3, 0]
    T[:, 1, 1] .= [0, -1, 0]

    return (TurbulenceClosures.∇_κ_∇ϕ(2, 1, 1, grid, T, closure, u, v, w, T, S) == 2κ &&
            TurbulenceClosures.∂ⱼ_2ν_Σ₁ⱼ(2, 1, 1, grid, closure, u, v, w, T, S) == 2ν &&
            TurbulenceClosures.∂ⱼ_2ν_Σ₂ⱼ(2, 1, 1, grid, closure, u, v, w, T, S) == 4ν &&
            TurbulenceClosures.∂ⱼ_2ν_Σ₃ⱼ(2, 1, 1, grid, closure, u, v, w, T, S) == 6ν
            )
end

function test_directional_diffusivity_fluxdiv(; νh=0.3, κh=0.7, νv=0.1, κv=0.5)
    closure = TurbulenceClosures.DirectionalDiffusivity(κh=κh, νh=νh, κv=κv, νv=νv)
    grid = RegularCartesianGrid((3, 1, 3), (3, 1, 3))

    u = zeros(3, 1, 3); v = zeros(3, 1, 3); w = zeros(3, 1, 3)
    T = zeros(3, 1, 3); S = zeros(3, 1, 3)

    u[:, 1, 1] .= [0,  1, 0]
    u[:, 1, 2] .= [0, -1, 0]
    u[:, 1, 3] .= [0,  1, 0]

    v[:, 1, 1] .= [0,  1, 0]
    v[:, 1, 2] .= [0, -2, 0]
    v[:, 1, 3] .= [0,  1, 0]
    w[:, 1, 1] .= [0,  1, 0]
    w[:, 1, 2] .= [0, -3, 0]
    w[:, 1, 3] .= [0,  1, 0]

    T[:, 1, 1] .= [0,  1, 0]
    T[:, 1, 2] .= [0, -4, 0]
    T[:, 1, 3] .= [0,  1, 0]

    return (TurbulenceClosures.∇_κ_∇ϕ(2, 1, 2, grid, T, closure, u, v, w, T, S) == 8κh + 10κv &&
            TurbulenceClosures.∂ⱼ_2ν_Σ₁ⱼ(2, 1, 2, grid, closure, u, v, w, T, S) == 2νh + 4νv &&
            TurbulenceClosures.∂ⱼ_2ν_Σ₂ⱼ(2, 1, 2, grid, closure, u, v, w, T, S) == 4νh + 6νv &&
            TurbulenceClosures.∂ⱼ_2ν_Σ₃ⱼ(2, 1, 2, grid, closure, u, v, w, T, S) == 6νh + 8νv
            )
end

function test_smag_divflux_finiteness()
    closure = TurbulenceClosures.ConstantSmagorinsky()
    grid = RegularCartesianGrid((3, 3, 3), (3, 3, 3))
    u, v, w = rand(size(grid)...), rand(size(grid)...), rand(size(grid)...)
    T, S = rand(size(grid)...), rand(size(grid)...)

    return (
        isfinite(TurbulenceClosures.∇_κ_∇ϕ(2, 1, 2, grid, T, closure, u, v, w, T, S)) &&
        isfinite(TurbulenceClosures.∂ⱼ_2ν_Σ₁ⱼ(2, 1, 2, grid, closure, u, v, w, T, S)) &&
        isfinite(TurbulenceClosures.∂ⱼ_2ν_Σ₂ⱼ(2, 1, 2, grid, closure, u, v, w, T, S)) &&
        isfinite(TurbulenceClosures.∂ⱼ_2ν_Σ₃ⱼ(2, 1, 2, grid, closure, u, v, w, T, S))
        )
end
