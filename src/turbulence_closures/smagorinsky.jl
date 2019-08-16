abstract type AbstractSmagorinsky{T} <: IsotropicDiffusivity{T} end

@inline geo_mean_Δᶠ(i, j, k, grid::RegularCartesianGrid{T}) where T = 
    (grid.Δx * grid.Δy * grid.Δz)^T(1/3)

#####
##### The "Deardorff" version of the Smagorinsky turbulence closure.
##### We also call this 'Constant Smagorinsky'.
#####

"""
    DeardorffSmagorinsky(T=Float64; C=0.23, Pr=1.0, ν_background=1e-6,
                            κ_background=1e-7)

Returns a `DeardorffSmagorinsky` closure object of type `T` with

*  `C` : Smagorinsky constant
* `Pr` : Prandtl number
*  `ν` : background viscosity
*  `κ` : background diffusivity
"""
Base.@kwdef struct DeardorffSmagorinsky{T} <: AbstractSmagorinsky{T}
    Cs :: T = 0.2
    Cb :: T = 1.0
    Pr :: T = 1.0
     ν :: T = 1e-6
     κ :: T = 1e-7
end

const ConstantSmagorinsky = DeardorffSmagorinsky

DeardorffSmagorinsky(T; kwargs...) = 
    typed_keyword_constructor(T, DeardorffSmagorinsky; kwargs...)

"""
    stability(N², Σ², Pr, Cb)

Return the stability function

``1 - \\sqrt( Cb N^2 / (Pr Σ^2) )``

when ``N^2 > 0``, and 1 otherwise.
"""
@inline stability(N²::T, Σ²::T, Pr::T, Cb::T) where T = 
    ifelse(Σ²==0, zero(T), one(T) - sqrt(stability_factor(N², Σ², Pr, Cb)))

@inline stability_factor(N²::T, Σ²::T, Pr::T, Cb::T) where T = 
    min(one(T), max(zero(T), Cb * N² / (Pr*Σ²)))

"""
    νₑ(ς, Cs, Δᶠ, Σ²)

Return the eddy viscosity for constant Smagorinsky
given the stability `ς`, model constant `Cs`,
filter width `Δᶠ`, and strain tensor dot product `Σ²`.
"""
@inline νₑ_deardorff(ς, Cs, Δᶠ, Σ²) = ς * (Cs*Δᶠ)^2 * sqrt(2Σ²)

@inline function ν_ccc(i, j, k, grid, clo::DeardorffSmagorinsky, c, eos, grav, u, v, w, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)
    N² = ▶z_aac(i, j, k, grid, ∂z_aaf, buoyancy, eos, grav, T, S)
     Δᶠ = Δᶠ_ccc(i, j, k, grid, clo)
     ς = stability(N², Σ², clo.Pr, clo.Cb)

    return νₑ_deardorff(ς, clo.Cs, Δᶠ, Σ²) + clo.ν
end

@inline function ν_ccf(i, j, k, grid, clo::DeardorffSmagorinsky, c, eos, grav, u, v, w, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_ccf(i, j, k, grid, u, v, w)
    N² = ∂z_aaf(i, j, k, grid, buoyancy, eos, grav, T, S)
     Δᶠ = Δᶠ_ccf(i, j, k, grid, clo)
     ς = stability(N², Σ², clo.Pr, clo.Cb)

    return νₑ_deardorff(ς, clo.Cs, Δᶠ, Σ²) + clo.ν
end


#####
##### Blasius Smagorinsky: the version of the Smagorinsky turbulence closure
##### used in the UK Met Office code "Blasius" (see Polton and Belcher 2007).
####

"""
    BlasiusSmagorinsky(T=Float64; Pr=1.0, ν=1e-6, κ=1e-7)

Returns a `BlasiusSmagorinsky` closure object of type `T` with

    * `Pr` : Prandtl number
    *  `ν` : background viscosity
    *  `κ` : background diffusivity
"""
struct BlasiusSmagorinsky{ML, T} <: AbstractSmagorinsky{T}
              Pr :: T
               ν :: T
               κ :: T
   mixing_length :: ML
end

BlasiusSmagorinsky(T=Float64; Pr=1.0, ν=1e-6, κ=1e-7, mixing_length=1.0) = 
    BlasiusSmagorinsky{typeof(mixing_length), T}(Pr, ν, κ, mixing_length)

const BS = BlasiusSmagorinsky

@inline Lm²(i, j, k, grid, closure::BS{<:AbstractFloat}, args...) = 
    closure.mixing_length^2

@inline Lm²(i, j, k, grid, closure::BS{<:Nothing}, args...) = 
    geo_mean_Δᶠ(i, j, k, grid)^2

#####
##### Mixing length via Monin-Obukhov theory
#####

struct MoninObukhovMixingLength{L, T, U, B}
    Cκ :: T
    z₀ :: T
    L₀ :: L
    Qu :: U
    Qb :: U
end

MoninObukhovMixingLength(T=Float64; Qu, Qb, Cκ=0.4, z₀=0.0, L₀=nothing) = 
    MoninObukhovMixingLength(T(Cκ), T(z₀), L₀, Qu, Qb)

const MO = MoninObukhovMixingLength

@inline ϕm(z, Lm) = ϕm(z, Lm.Qu, Lm.Qb, Lm.Cκ)

@inline function ϕm(z::T, Qu, Qb, Cκ) where T
    if Qu == 0
        return zero(T)
    elseif Qb > 0 # unstable
        return ( 1 - 15Cκ * Qb * z / Qu^T(1.5) )^T(-0.25)
    else # neutral or stable
        return 1 + 5Cκ * Qb * z / Qu^T(1.5)
    end
end

@inline function Lm²(i, j, k, grid, clo::BlasiusSmagorinsky{<:MO, T}, Σ², N²) where T
    Lm = clo.mixing_length
    z = @inbounds grid.zC[i, j, k]
    Lm⁻² = ( 
             ( ϕm(z + Lm.z₀, Lm) * buoyancy_factor(Σ², N²)^T(0.25) / (Lm.Cκ * (z + Lm.z₀)) )^2 
            + 1 / L₀(i, j, k, grid, Lm)^2
           )
    return 1 / Lm⁻²
end

@inline L₀(i, j, k, grid, mixing_length::MO{<:Number}) = mixing_length.L₀

@inline L₀(i, j, k, grid, mixing_length::MO{<:Nothing}) = 
    geo_mean_Δᶠ(i, j, k, grid)

@inline buoyancy_factor(Σ², N²::T) where T = 
    ifelse(Σ²==0, zero(T), max(zero(T), (one(T) - N² / Σ²)))

"""
    νₑ_blasius(Lm², Σ², N²)

Return the eddy viscosity for the BLASIUS version of constant Smagorinsky
(Polton and Belcher, 2007) given the squared mixing length scale `Lm²`,
strain tensor dot product `Σ²`, and buoyancy gradient / squared buoyancy
frequency `N²`.
"""
@inline νₑ_blasius(Lm², Σ², N²::T) where T = Lm² * sqrt(Σ²/2 * buoyancy_factor(Σ², N²))

@inline function ν_ccc(i, j, k, grid, clo::BlasiusSmagorinsky, c, eos, grav, u, v, w, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, u, v, w)
    N² = ▶z_aac(i, j, k, grid, ∂z_aaf, buoyancy, eos, grav, T, S)
    Lm_sq = Lm²(i, j, k, grid, clo, Σ², N²)

    return νₑ_blasius(Lm_sq, Σ², N²) + clo.ν
end

@inline function ν_ccf(i, j, k, grid, clo::BlasiusSmagorinsky, c, eos, grav, u, v, w, T, S)
    Σ² = ΣᵢⱼΣᵢⱼ_ccf(i, j, k, grid, u, v, w)
    N² = ∂z_aaf(i, j, k, grid, buoyancy, eos, grav, T, S)
    Lm_sq = Lm²(i, j, k, grid, clo, Σ², N²)

    return νₑ_blasius(Lm_sq, Σ², N²) + clo.ν
end

#####
##### Abstract Smagorinsky functionality
#####

function TurbulentDiffusivities(arch::Architecture, grid::Grid, ::AbstractSmagorinsky)
    νₑ = CellField(arch, grid)
    return (νₑ=νₑ,)
end

"Return the filter width for Constant Smagorinsky on a Regular Cartesian grid."
@inline Δᶠ(i, j, k, grid::RegularCartesianGrid, ::AbstractSmagorinsky) = geo_mean_Δᶠ(i, j, k, grid)

# Temporarily set filter widths to cell-size (rather than distance between cell centers, etc.)
const Δᶠ_ccc = Δᶠ
const Δᶠ_ccf = Δᶠ
const Δᶠ_ffc = Δᶠ
const Δᶠ_fcf = Δᶠ
const Δᶠ_cff = Δᶠ

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

@inline function ΣᵢⱼΣᵢⱼ_ccf(i, j, k, grid, u, v, w)
    return (
                    ▶z_aaf(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ▶xyz_ccf(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *   ▶x_caa(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *   ▶y_aca(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

@inline function κ_ccc(i, j, k, grid, clo::AbstractSmagorinsky, c, eos, grav, u, v, w, T, S)
    νₑ = ν_ccc(i, j, k, grid, clo, c, eos, grav, u, v, w, T, S)
    return (νₑ - clo.ν) / clo.Pr + clo.κ
end

@inline function κ_ccf(i, j, k, grid, clo::AbstractSmagorinsky, c, eos, grav, u, v, w, T, S)
    νₑ = ν_ccf(i, j, k, grid, clo, c, eos, grav, u, v, w, T, S)
    return (νₑ - clo.ν) / clo.Pr + clo.κ
end

"""
    κ_∂x_c(i, j, k, grid, c, κ, closure, eos, g, u, v, w, T, S)

Return `κ ∂x c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂x_c(i, j, k, grid, c, νₑ, closure::AbstractSmagorinsky)
    νₑ = ▶x_faa(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / closure.Pr + closure.κ
    ∂x_c = ∂x_faa(i, j, k, grid, c)
    return κₑ * ∂x_c
end

"""
    κ_∂y_c(i, j, k, grid, c, κ, closure::AbstractSmagorinsky, eos, g, u, v, w, T, S)

Return `κ ∂y c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂y_c(i, j, k, grid, c, νₑ, closure::AbstractSmagorinsky)
    νₑ = ▶y_afa(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / closure.Pr + closure.κ
    ∂y_c = ∂y_afa(i, j, k, grid, c)
    return κₑ * ∂y_c
end

"""
    κ_∂z_c(i, j, k, grid, c, κ, closure::AbstractSmagorinsky, eos, g, u, v, w, T, S)

Return `κ ∂z c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂z_c(i, j, k, grid, c, νₑ, closure::AbstractSmagorinsky)
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
@inline ∇_κ_∇c(i, j, k, grid, c, closure::AbstractSmagorinsky, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, c, diffusivities.νₑ, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, c, diffusivities.νₑ, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, c, diffusivities.νₑ, closure)
)

# This function assumes rigid lids on the top and bottom.
function calc_diffusivities!(diffusivities, grid, closure::AbstractSmagorinsky, eos, grav, U, Φ)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if k == 1
                    @inbounds diffusivities.νₑ[i, j, k] =
                        ν_ccf(i, j, 2, grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                elseif k == grid.Nz
                    @inbounds diffusivities.νₑ[i, j, k] =
                        ν_ccf(i, j, k, grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                else
                    @inbounds diffusivities.νₑ[i, j, k] =
                        ν_ccc(i, j, k, grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                end
            end
        end
    end
end

#####
##### Double dot product of strain on cell edges (currently unused)
#####

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
