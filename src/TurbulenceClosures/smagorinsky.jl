@inline geo_mean_Δᶠ(i, j, k, grid::RegularCartesianGrid{T}) where T =
    (grid.Δx * grid.Δy * grid.Δz)^T(1/3)

#####
##### The turbulence closure proposed by Smagorinsky and Lilly.
##### We also call this 'Constant Smagorinsky'.
#####

"""
    SmagorinskyLilly{T} <: AbstractSmagorinsky{T}

Parameters for the Smagorinsky-Lilly turbulence closure.
"""
struct SmagorinskyLilly{T, P, K} <: AbstractSmagorinsky{T}
     C :: T
    Cb :: T
    Pr :: P
     ν :: T
     κ :: K
    function SmagorinskyLilly{T}(C, Cb, Pr, ν, κ) where T
        Pr = convert_diffusivity(T, Pr)
         κ = convert_diffusivity(T, κ)
        return new{T, typeof(Pr), typeof(κ)}(C, Cb, Pr, ν, κ)
    end
end

"""
    SmagorinskyLilly([T=Float64;] C=0.23, Pr=1, ν=1.05e-6, κ=1.46e-7)

Return a `SmagorinskyLilly` type associated with the turbulence closure proposed by
Lilly (1962) and Smagorinsky (1958, 1963), which has an eddy viscosity of the form

    `νₑ = (C * Δᶠ)² * √(2Σ²) * √(1 - Cb * N² / (Pr * Σ²)) + ν`,

and an eddy diffusivity of the form

    `κₑ = (νₑ - ν) / Pr + κ`

where `Δᶠ` is the filter width, `Σ² = ΣᵢⱼΣᵢⱼ` is the double dot product of
the strain tensor `Σᵢⱼ`, and `N²` is the total buoyancy gradient.

Keyword arguments
=================
    - `C::T`: Model constant
    - `Cb::T`: Buoyancy term multipler (`Cb = 0` turns it off, `Cb = 1` turns it on)
    - `Pr`: Turbulent Prandtl numbers for each tracer (a single `Pr` is applied to every tracer)
    - `ν::T`: background viscosity
    - `κ`: background diffusivities for each tracer (a single `κ` is applied to every tracer)

References
==========

* Smagorinsky, J. "On the numerical integration of the primitive equations of motion for 
    baroclinic flow in a closed region." Monthly Weather Review (1958)
* Lilly, D. K. "On the numerical simulation of buoyant convection." Tellus (1962)
* Smagorinsky, J. "General circulation experiments with the primitive equations: I. 
    The basic experiment." Monthly weather review (1963)
"""
SmagorinskyLilly(T=Float64; C=0.23, Cb=1.0, Pr=1.0, ν=ν₀, κ=κ₀) =
    SmagorinskyLilly{T}(C, Cb, Pr, ν, κ)

function with_tracers(tracers, closure::SmagorinskyLilly{T}) where T
    Pr = tracer_diffusivities(tracers, closure.Pr)
     κ = tracer_diffusivities(tracers, closure.κ)
    return SmagorinskyLilly{T}(closure.C, closure.Cb, Pr, closure.ν, κ)
end

"""
    stability(N², Σ², Pr, Cb)

Return the stability function

    ``\$ \\sqrt(1 - Cb N^2 / (Pr Σ^2) ) \$``

when ``N^2 > 0``, and 1 otherwise.
"""
@inline stability(N²::T, Σ²::T, Pr::T, Cb::T) where T =
    ifelse(Σ²==0, zero(T), sqrt(one(T) - stability_factor(N², Σ², Pr, Cb)))

@inline stability_factor(N²::T, Σ²::T, Pr::T, Cb::T) where T =
    min(one(T), max(zero(T), Cb * N² / (Pr * Σ²)))

"""
    νₑ(ς, C, Δᶠ, Σ²)

Return the eddy viscosity for constant Smagorinsky
given the stability `ς`, model constant `C`,
filter width `Δᶠ`, and strain tensor dot product `Σ²`.
"""
@inline νₑ_deardorff(ς, C, Δᶠ, Σ²) = ς * (C*Δᶠ)^2 * sqrt(2Σ²)

@inline function ν_ccc(i, j, k, grid, clo::SmagorinskyLilly, c, buoyancy_params, U, Φ)
    Σ² = ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, U.u, U.v, U.w)
    N² = ▶z_aac(i, j, k, grid, buoyancy_frequency_squared, buoyancy_params, Φ)
    Δᶠ = Δᶠ_ccc(i, j, k, grid, clo)
     ς = stability(N², Σ², clo.Pr, clo.Cb)

    return νₑ_deardorff(ς, clo.C, Δᶠ, Σ²) + clo.ν
end

#####
##### Blasius Smagorinsky: the version of the Smagorinsky turbulence closure
##### used in the UK Met Office code "Blasius" (see Polton and Belcher 2007).
####

"""
    BlasiusSmagorinsky{ML, T}

Parameters for the version of the Smagorinsky closure used in the UK Met Office code
Blasius, according to Polton and Belcher (2007).
"""
struct BlasiusSmagorinsky{ML, T, P, K} <: AbstractSmagorinsky{T}
               Pr :: P
                ν :: T
                κ :: K
    mixing_length :: ML

    function ConstantIsotropicDiffusivity{T}(Pr, ν, κ, mixing_length) where T
        Pr = convert_diffusivity(T, Pr)
         κ = convert_diffusivity(T, κ)
        return new{typeof(mixing_length), T, typeof(Pr), typeof(κ)}(Pr, ν, κ, mixing_length)
    end
end

"""
    BlasiusSmagorinsky(T=Float64; Pr=1.0, ν=1.05e-6, κ=1.46e-7)

Returns a `BlasiusSmagorinsky` closure object of type `T` with

    * `Pr` : Prandtl number
    *  `ν` : background viscosity
    *  `κ` : background diffusivity

"""
BlasiusSmagorinsky(T=Float64; Pr=1.0, ν=ν₀, κ=κ₀, mixing_length=nothing) =
    BlasiusSmagorinsky{typeof(mixing_length), T}(Pr, ν, κ, mixing_length)

function with_tracers(tracers, closure::BlasiusSmagorinsky{ML, T}) where {ML, T}
    Pr = tracer_diffusivities(tracers, closure.Pr)
     κ = tracer_diffusivities(tracers, closure.κ)
    return BlasiusSmagorinsky{T}(Pr, closure.ν, κ, closure.mixing_length)
end

const BS = BlasiusSmagorinsky

@inline Lm²(i, j, k, grid, closure::BS{<:AbstractFloat}, args...) =
    closure.mixing_length^2

@inline Lm²(i, j, k, grid, closure::BS{<:Nothing}, args...) =
    geo_mean_Δᶠ(i, j, k, grid)^2

#####
##### Mixing length via Monin-Obukhov theory
#####

"""
    MoninObukhovMixingLength(T=Float64; kwargs...)

Returns a `MoninObukhovMixingLength closure object of type `T` with

    * Cκ : Von Karman constant
    * z₀ : roughness length
    * L₀ : 'base' mixing length
    * Qu : surface velocity flux (Qu = τ/ρ₀)
    * Qb : surface buoyancy flux

The surface velocity flux and buoyancy flux are restricted to constants for now.
"""
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

"""
    ϕm(z, Qu, Qb, Cκ)

Return the stability function for monentum at height
`z` in terms of the velocity flux `Qu`, buoyancy flux
`Qb`, and von Karman constant `Cκ`.
"""
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

@inline function ν_ccc(i, j, k, grid, clo::BlasiusSmagorinsky, c, buoyancy, U, Φ)
    Σ² = ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, U.u, U.v, U.w)
    N² = buoyancy_frequency_squared(i, j, k, grid, buoyancy, Φ)
    Lm_sq = Lm²(i, j, k, grid, clo, Σ², N²)

    return νₑ_blasius(Lm_sq, Σ², N²) + clo.ν
end

#####
##### Abstract Smagorinsky functionality
#####

TurbulentDiffusivities(arch::AbstractArchitecture, grid::AbstractGrid, ::AbstractSmagorinsky) =
    (νₑ=CellField(arch, grid),)

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

@inline function κ_ccc(i, j, k, grid, clo::AbstractSmagorinsky, c, buoyancy, U, Φ)
    νₑ = ν_ccc(i, j, k, grid, clo, c, buoyancy, U, Φ)
    return (νₑ - clo.ν) / clo.Pr + clo.κ
end

"""
    κ_∂x_c(i, j, k, grid, c, κ, closure, buoyancy, u, v, w, T, S)

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
    κ_∂y_c(i, j, k, grid, c, κ, closure::AbstractSmagorinsky, buoyancy, u, v, w, T, S)

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
    κ_∂z_c(i, j, k, grid, c, κ, closure::AbstractSmagorinsky, buoyancy, u, v, w, T, S)

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
function calc_diffusivities!(diffusivities, grid, closure::AbstractSmagorinsky, buoyancy, U, Φ)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds diffusivities.νₑ[i, j, k] = ν_ccc(i, j, k, grid, closure, nothing, buoyancy, U, Φ)
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

@inline function ΣᵢⱼΣᵢⱼ_ccf(i, j, k, grid, u, v, w)
    return (
                    ▶z_aaf(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ▶xyz_ccf(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *   ▶x_caa(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *   ▶y_aca(i, j, k, grid, Σ₂₃², u, v, w)
            )
end
