Base.@kwdef struct ConstantSmagorinsky{T} <: IsotropicDiffusivity{T}
               C :: T = 0.23
              Pr :: T = 1.0
    ν_background :: T = 0.0
    κ_background :: T = 0.0
end

"""
    ConstantSmagorinsky(T=Float64; C=0.23, Pr=1.0, ν_background=1e-6,
                            κ_background=1e-7)

Returns a `ConstantSmagorinsky` closure object of type `T` with

* `C`            : Smagorinsky constant
* `Pr`           : Prandtl number
* `ν_background` : background viscosity
* `κ_background` : background diffusivity
"""
ConstantSmagorinsky(T; kwargs...) =
      typed_keyword_constructor(T, ConstantSmagorinsky; kwargs...)

"Return the filter width for Constant Smagorinsky on a Regular Cartesian grid."
Δ(i, j, k, grid::RegularCartesianGrid, ::ConstantSmagorinsky) = geo_mean_Δ(grid)

# tr_Σ² : ccc
#   Σ₁₂ : ffc
#   Σ₁₃ : fcf
#   Σ₂₃ : cff

"Return the double dot product of strain at `ccc`."
function ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)
    return (
                    tr_Σ²(i, j, k, grid, u, v, w)
            + 2 * ▶xy_cca(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ▶xz_cac(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ▶yz_acc(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `ffc`."
function ΣᵢⱼΣᵢⱼ_ffc(i, j, k, grid, u, v, w)
    return (
                  ▶xy_ffa(i, j, k, grid, tr_Σ², u, v, w)
            + 2 *    Σ₁₂²(i, j, k, grid, u, v, w)
            + 2 * ▶yz_afc(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ▶xz_fac(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `fcf`."
function ΣᵢⱼΣᵢⱼ_fcf(i, j, k, grid, u, v, w)
    return (
                  ▶xz_faf(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ▶yz_acf(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *    Σ₁₃²(i, j, k, grid, u, v, w)
            + 2 * ▶xy_fca(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `cff`."
function ΣᵢⱼΣᵢⱼ_cff(i, j, k, grid, u, v, w)
    return (
                  ▶yz_aff(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ▶xz_caf(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ▶xy_cfa(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *    Σ₂₃²(i, j, k, grid, u, v, w)
            )
end

# Temporarily set filter widths to cell-size (rather than distance between cell centers, etc.)
Δ_ccc = Δ
Δ_ffc = Δ
Δ_fcf = Δ
Δ_cff = Δ

ν_ccc(i, j, k, grid, clo::ConstantSmagorinsky, u, v, w, T, S) =
    (clo.C * Δ_ccc(i, j, k, grid, clo))^2 * sqrt(2 * ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)) + clo.ν_background

ν_ffc(i, j, k, grid, clo::ConstantSmagorinsky, u, v, w, T, S) =
    (clo.C * Δ_ffc(i, j, k, grid, clo))^2 * sqrt(2 * ΣᵢⱼΣᵢⱼ_ffc(i, j, k, grid, u, v, w)) + clo.ν_background

ν_fcf(i, j, k, grid, clo::ConstantSmagorinsky, u, v, w, T, S) =
    (clo.C * Δ_fcf(i, j, k, grid, clo))^2 * sqrt(2 * ΣᵢⱼΣᵢⱼ_fcf(i, j, k, grid, u, v, w)) + clo.ν_background

ν_cff(i, j, k, grid, clo::ConstantSmagorinsky, u, v, w, T, S) =
    (clo.C * Δ_cff(i, j, k, grid, clo))^2 * sqrt(2 * ΣᵢⱼΣᵢⱼ_cff(i, j, k, grid, u, v, w)) + clo.ν_background

κ_ccc(i, j, k, grid, clo::ConstantSmagorinsky, u, v, w, T, S) =
    (clo.C * Δ_ccc(i, j, k, grid, clo))^2 * sqrt(2 * ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)) / clo.Pr + clo.κ_background
