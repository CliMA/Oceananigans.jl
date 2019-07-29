Base.@kwdef struct ConstantSmagorinsky{T} <: IsotropicDiffusivity{T}
    Cs :: T = 0.2
    Cb :: T = 1.0
    Pr :: T = 1.0
     ν :: T = 1e-6
     κ :: T = 1e-7
end

"""
    ConstantSmagorinsky(T=Float64; C=0.23, Pr=1.0, ν_background=1e-6,
                            κ_background=1e-7)

Returns a `ConstantSmagorinsky` closure object of type `T` with

*  `C` : Smagorinsky constant
* `Pr` : Prandtl number
*  `ν` : background viscosity
*  `κ` : background diffusivity
"""
ConstantSmagorinsky(T; kwargs...) =
      typed_keyword_constructor(T, ConstantSmagorinsky; kwargs...)

function TurbulentDiffusivities(arch::Architecture, grid::Grid, ::ConstantSmagorinsky)
    νₑ = zeros(arch, grid)
    return (νₑ=νₑ,)
end

"Return the filter width for Constant Smagorinsky on a Regular Cartesian grid."
@inline Δ(i, j, k, grid::RegularCartesianGrid, ::ConstantSmagorinsky) = geo_mean_Δ(grid)

# tr_Σ² : ccc
#   Σ₁₂ : ffc
#   Σ₁₃ : fcf
#   Σ₂₃ : cff

"Return the double dot product of strain at `ccc`."
@inline function ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)
    return (
                    tr_Σ²(i, j, k, grid, u, v, w)
            + 2 * ▶xy_cca(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ▶xz_cac(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ▶yz_acc(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

# Temporarily set filter widths to cell-size (rather than distance between cell centers, etc.)
const Δ_ccc = Δ
const Δ_ffc = Δ
const Δ_fcf = Δ
const Δ_cff = Δ

"""
    stability(N², Σ², Pr, Cb)

Return the stability function

``1 - \\sqrt( Cb N^2 / (Pr Σ^2) )``

when ``N^2 > 0``, and 1 otherwise.
"""
@inline stability(N²::T, Σ²::T, Pr::T, Cb::T) where T = one(T) - sqrt(stability_factor(N², Σ², Pr, Cb))
@inline stability_factor(N²::T, Σ²::T, Pr::T, Cb::T) where T = min(one(T), max(zero(T), Cb * N² / (Pr*Σ²)))

"""
    νₑ(ς, Cs, Δ, Σ²)

Return the eddy viscosity for constant Smagorinsky
given the stability `ς`, model constant `Cs`,
filter with `Δ`, and strain tensor dot product `Σ²`.
"""
@inline νₑ(ς, Cs, Δ, Σ²) = ς * (Cs*Δ)^2 * sqrt(2Σ²)

@inline function ν_ccc(i, j, k, grid, clo::ConstantSmagorinsky, c, eos, grav, u, v, w, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)
    N² = ▶z_aac(i, j, k, grid, ∂z_aaf, buoyancy, eos, grav, T, S)
     Δ = Δ_ccc(i, j, k, grid, clo)
     ς = stability(N², Σ², clo.Pr, clo.Cb)

    return νₑ(ς, clo.Cs, Δ, Σ²) + clo.ν
end

@inline function κ_ccc(i, j, k, grid, clo::ConstantSmagorinsky, c, eos, grav, u, v, w, T, S)
    νₑ = ν_ccc(i, j, k, grid, clo, c, eos, grav, u, v, w, T, S)
    return (νₑ - clo.ν) / clo.Pr + clo.κ
end

"""
    κ_∂x_c(i, j, k, grid, c, κ, closure, eos, g, u, v, w, T, S)

Return `κ ∂x c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂x_c(i, j, k, grid, c, νₑ, closure::ConstantSmagorinsky)
    νₑ = ▶x_faa(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / closure.Pr + closure.κ
    ∂x_c = ∂x_faa(i, j, k, grid, c)
    return κₑ * ∂x_c
end

"""
    κ_∂y_c(i, j, k, grid, c, κ, closure::ConstantSmagorinsky, eos, g, u, v, w, T, S)

Return `κ ∂y c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂y_c(i, j, k, grid, c, νₑ, closure::ConstantSmagorinsky)
    νₑ = ▶y_afa(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / closure.Pr + closure.κ
    ∂y_c = ∂y_afa(i, j, k, grid, c)
    return κₑ * ∂y_c
end

"""
    κ_∂z_c(i, j, k, grid, c, κ, closure::ConstantSmagorinsky, eos, g, u, v, w, T, S)

Return `κ ∂z c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂z_c(i, j, k, grid, c, νₑ, closure::ConstantSmagorinsky)
    νₑ = ▶z_aaf(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / closure.Pr + closure.κ
    ∂z_c = ∂z_aaf(i, j, k, grid, c)
    return κₑ * ∂z_c
end

"""
    ∇_κ_∇c(i, j, k, grid, c, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ c)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇c(i, j, k, grid, c, closure::ConstantSmagorinsky, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, c, diffusivities.νₑ, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, c, diffusivities.νₑ, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, c, diffusivities.νₑ, closure)
)

function calc_diffusivities!(diffusivities, grid, closure::ConstantSmagorinsky, eos, grav, U, Φ)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds diffusivities.νₑ[i, j, k] =
                    ν_ccc(i, j, k, grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
            end
        end
    end
end

#
# Double dot product of strain on cell edges (currently unused)
#

"Return the double dot product of strain at `ffc`."
@inline function ΣᵢⱼΣᵢⱼ_ffc(i, j, k, grid, u, v, w)
    return (
                  ▶xy_ffa(i, j, k, grid, tr_Σ², u, v, w)
            + 2 *    Σ₁₂²(i, j, k, grid, u, v, w)
            + 2 * ▶yz_afc(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ▶xz_fac(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `fcf`."
@inline function ΣᵢⱼΣᵢⱼ_fcf(i, j, k, grid, u, v, w)
    return (
                  ▶xz_faf(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ▶yz_acf(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *    Σ₁₃²(i, j, k, grid, u, v, w)
            + 2 * ▶xy_fca(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `cff`."
@inline function ΣᵢⱼΣᵢⱼ_cff(i, j, k, grid, u, v, w)
    return (
                  ▶yz_aff(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ▶xz_caf(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ▶xy_cfa(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *    Σ₂₃²(i, j, k, grid, u, v, w)
            )
end
