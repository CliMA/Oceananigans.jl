#####
##### The turbulence closure proposed by Smagorinsky and Lilly.
##### We also call this 'Constant Smagorinsky'.
#####

"""
    SmagorinskyLilly{FT} <: AbstractSmagorinsky{FT}

Parameters for the Smagorinsky-Lilly turbulence closure.
"""
struct SmagorinskyLilly{FT, P, K} <: AbstractSmagorinsky{FT}
     C :: FT
    Cb :: FT
    Pr :: P
     ν :: FT
     κ :: K
    function SmagorinskyLilly{FT}(C, Cb, Pr, ν, κ) where FT
        Pr = convert_diffusivity(FT, Pr)
         κ = convert_diffusivity(FT, κ)
        return new{FT, typeof(Pr), typeof(κ)}(C, Cb, Pr, ν, κ)
    end
end

"""
    SmagorinskyLilly([FT=Float64;] C=0.23, Pr=1, ν=1.05e-6, κ=1.46e-7)

Return a `SmagorinskyLilly` type associated with the turbulence closure proposed by
Lilly (1962) and Smagorinsky (1958, 1963), which has an eddy viscosity of the form

    `νₑ = (C * Δᶠ)² * √(2Σ²) * √(1 - Cb * N² / Σ²) + ν`,

and an eddy diffusivity of the form

    `κₑ = (νₑ - ν) / Pr + κ`

where `Δᶠ` is the filter width, `Σ² = ΣᵢⱼΣᵢⱼ` is the double dot product of
the strain tensor `Σᵢⱼ`, `Pr` is the turbulent Prandtl number, and `N²` is 
the total buoyancy gradient, and `Cb` is a constant the multiplies the Richardson number
modification to the eddy viscosity.

Keyword arguments
=================
    - `C`  : Model constant
    - `Cb` : Buoyancy term multipler (`Cb = 0` turns it off, `Cb ≠ 0` turns it on. 
             Typically `Cb=1/Pr`.)
    - `Pr` : Turbulent Prandtl numbers for each tracer. Either a constant applied to every 
             tracer, or a `NamedTuple` with fields for each tracer individually.
    - `ν`  : Constant background viscosity for momentum
    - `κ`  : Constant background diffusivity for tracer. Can either be a single number 
             applied to all tracers, or `NamedTuple` of diffusivities corresponding to each 
             tracer.

References
==========
Smagorinsky, J. "On the numerical integration of the primitive equations of motion for 
    baroclinic flow in a closed region." Monthly Weather Review (1958)

Lilly, D. K. "On the numerical simulation of buoyant convection." Tellus (1962)

Smagorinsky, J. "General circulation experiments with the primitive equations: I. 
    The basic experiment." Monthly weather review (1963)
"""
SmagorinskyLilly(FT=Float64; C=0.23, Cb=1.0, Pr=1.0, ν=ν₀, κ=κ₀) =
    SmagorinskyLilly{FT}(C, Cb, Pr, ν, κ)

function with_tracers(tracers, closure::SmagorinskyLilly{FT}) where FT
    Pr = tracer_diffusivities(tracers, closure.Pr)
     κ = tracer_diffusivities(tracers, closure.κ)
    return SmagorinskyLilly{FT}(closure.C, closure.Cb, Pr, closure.ν, κ)
end

"""
    stability(N², Σ², Cb)

Return the stability function

    ``\$ \\sqrt(1 - Cb N^2 / Σ^2 ) \$``

when ``N^2 > 0``, and 1 otherwise.
"""
@inline stability(N²::FT, Σ²::FT, Cb::FT) where FT =
    ifelse(Σ²==0, zero(FT), sqrt(one(FT) - stability_factor(N², Σ², Cb)))

@inline stability_factor(N²::FT, Σ²::FT, Cb::FT) where FT = min(one(FT), Cb * N² / Σ²)

"""
    νₑ(ς, C, Δᶠ, Σ²)

Return the eddy viscosity for constant Smagorinsky
given the stability `ς`, model constant `C`,
filter width `Δᶠ`, and strain tensor dot product `Σ²`.
"""
@inline νₑ_deardorff(ς, C, Δᶠ, Σ²) = ς * (C*Δᶠ)^2 * sqrt(2Σ²)

@inline function ν_ccc(i, j, k, grid, clo::SmagorinskyLilly{FT}, buoyancy, U, C) where FT
    Σ² = ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, U.u, U.v, U.w)
    N² = max(zero(FT), ▶z_aac(i, j, k, grid, buoyancy_frequency_squared, buoyancy, C))
    Δᶠ = Δᶠ_ccc(i, j, k, grid, clo)
     ς = stability(N², Σ², clo.Cb) # Use unity Prandtl number.

    return νₑ_deardorff(ς, clo.C, Δᶠ, Σ²) + clo.ν
end

#####
##### Blasius Smagorinsky: the version of the Smagorinsky turbulence closure
##### used in the UK Met Office code "Blasius" (see Polton and Belcher 2007).
#####

"""
    BlasiusSmagorinsky{ML, FT}

Parameters for the version of the Smagorinsky closure used in the UK Met Office code
Blasius, according to Polton and Belcher (2007).
"""
struct BlasiusSmagorinsky{ML, FT, P, K} <: AbstractSmagorinsky{FT}
               Pr :: P
                ν :: FT
                κ :: K
    mixing_length :: ML

    function BlasiusSmagorinsky{FT}(Pr, ν, κ, mixing_length) where FT
        Pr = convert_diffusivity(FT, Pr)
         κ = convert_diffusivity(FT, κ)
        return new{typeof(mixing_length), FT, typeof(Pr), typeof(κ)}(Pr, ν, κ, mixing_length)
    end
end

"""
    BlasiusSmagorinsky(FT=Float64; Pr=1.0, ν=1.05e-6, κ=1.46e-7)

Returns a `BlasiusSmagorinsky` closure object of type `FT`.

Keyword arguments
=================
    - `Pr` : Turbulent Prandtl numbers for each tracer. Either a constant applied to every 
             tracer, or a `NamedTuple` with fields for each tracer individually.
    - `ν`  : Constant background viscosity for momentum
    - `κ`  : Constant background diffusivity for tracer. Can either be a single number 
             applied to all tracers, or `NamedTuple` of diffusivities corresponding to each 
             tracer.

References
==========
Polton, J. A., and Belcher, S. E. (2007), "Langmuir turbulence and deeply penetrating jets 
    in an unstratified mixed layer." Journal of Geophysical Research: Oceans.
"""
BlasiusSmagorinsky(FT=Float64; Pr=1.0, ν=ν₀, κ=κ₀, mixing_length=nothing) =
    BlasiusSmagorinsky{FT}(Pr, ν, κ, mixing_length)

function with_tracers(tracers, closure::BlasiusSmagorinsky{ML, FT}) where {ML, FT}
    Pr = tracer_diffusivities(tracers, closure.Pr)
     κ = tracer_diffusivities(tracers, closure.κ)
    return BlasiusSmagorinsky{FT}(Pr, closure.ν, κ, closure.mixing_length)
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
    MoninObukhovMixingLength(FT=Float64; kwargs...)

Returns a `MoninObukhovMixingLength closure object of type `FT` with

    * Cκ : Von Karman constant
    * z₀ : roughness length
    * L₀ : 'base' mixing length
    * Qu : surface velocity flux (Qu = τ/ρ₀)
    * Qb : surface buoyancy flux

The surface velocity flux and buoyancy flux are restricted to constants for now.
"""
struct MoninObukhovMixingLength{L, FT, U, B}
    Cκ :: FT
    z₀ :: FT
    L₀ :: L
    Qu :: U
    Qb :: U
end

MoninObukhovMixingLength(FT=Float64; Qu, Qb, Cκ=0.4, z₀=0.0, L₀=nothing) =
    MoninObukhovMixingLength(FT(Cκ), FT(z₀), L₀, Qu, Qb)

const MO = MoninObukhovMixingLength

@inline ϕm(z, Lm) = ϕm(z, Lm.Qu, Lm.Qb, Lm.Cκ)

"""
    ϕm(z, Qu, Qb, Cκ)

Return the stability function for monentum at height
`z` in terms of the velocity flux `Qu`, buoyancy flux
`Qb`, and von Karman constant `Cκ`.
"""
@inline function ϕm(z::FT, Qu, Qb, Cκ) where FT
    if Qu == 0
        return zero(FT)
    elseif Qb > 0 # unstable
        return ( 1 - 15Cκ * Qb * z / Qu^FT(1.5) )^FT(-0.25)
    else # neutral or stable
        return 1 + 5Cκ * Qb * z / Qu^FT(1.5)
    end
end

@inline function Lm²(i, j, k, grid, clo::BlasiusSmagorinsky{<:MO, FT}, Σ², N²) where FT
    Lm = clo.mixing_length
    z = @inbounds grid.zC[i, j, k]
    Lm⁻² = (
             ( ϕm(z + Lm.z₀, Lm) * buoyancy_factor(Σ², N²)^FT(0.25) / (Lm.Cκ * (z + Lm.z₀)) )^2
            + 1 / L₀(i, j, k, grid, Lm)^2
           )
    return 1 / Lm⁻²
end

@inline L₀(i, j, k, grid, mixing_length::MO{<:Number})  = mixing_length.L₀
@inline L₀(i, j, k, grid, mixing_length::MO{<:Nothing}) = geo_mean_Δᶠ(i, j, k, grid)
@inline buoyancy_factor(Σ², N²::FT) where FT = ifelse(Σ²==0, zero(FT), max(zero(FT), one(FT) - N² / Σ²))

"""
    νₑ_blasius(Lm², Σ², N²)

Return the eddy viscosity for the BLASIUS version of constant Smagorinsky
(Polton and Belcher, 2007) given the squared mixing length scale `Lm²`,
strain tensor dot product `Σ²`, and buoyancy gradient / squared buoyancy
frequency `N²`.
"""
@inline νₑ_blasius(Lm², Σ², N²::FT) where FT = Lm² * sqrt(Σ²/2 * buoyancy_factor(Σ², N²))

@inline function ν_ccc(i, j, k, grid, clo::BlasiusSmagorinsky{ML, FT}, buoyancy, U, C) where {ML, FT}
    Σ² = ΣᵢⱼΣᵢⱼ_ccc(i, j, k, grid, U.u, U.v, U.w)
    N² = max(zero(FT), ▶z_aac(i, j, k, grid, buoyancy_frequency_squared, buoyancy, C))
    Lm_sq = Lm²(i, j, k, grid, clo, Σ², N²)

    return νₑ_blasius(Lm_sq, Σ², N²) + clo.ν
end

#####
##### Abstract Smagorinsky functionality
#####

TurbulentDiffusivities(arch::AbstractArchitecture, grid::AbstractGrid, tracers, ::AbstractSmagorinsky) =
    (νₑ=CellField(arch, grid),)
"""
    κ_∂x_c(i, j, k, grid, c, tracer, closure, νₑ)

Return `κ ∂x c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂x_c(i, j, k, grid, c, tracer_index, closure::AbstractSmagorinsky, νₑ) 
    @inbounds Pr = closure.Pr[tracer_index]
    @inbounds κ = closure.κ[tracer_index]

    νₑ = ▶x_faa(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / Pr + κ
    ∂x_c = ∂x_faa(i, j, k, grid, c)
    return κₑ * ∂x_c
end

"""
    κ_∂y_c(i, j, k, grid, c, tracer, closure, νₑ)

Return `κ ∂y c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂y_c(i, j, k, grid, c, tracer_index, closure::AbstractSmagorinsky, νₑ) 
    @inbounds Pr = closure.Pr[tracer_index]
    @inbounds κ = closure.κ[tracer_index]

    νₑ = ▶y_afa(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / Pr + κ
    ∂y_c = ∂y_afa(i, j, k, grid, c)
    return κₑ * ∂y_c
end

"""
    κ_∂z_c(i, j, k, grid, c, tracer, closure, νₑ)

Return `κ ∂z c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂z_c(i, j, k, grid, c, tracer_index, closure::AbstractSmagorinsky, νₑ) 
    @inbounds Pr = closure.Pr[tracer_index]
    @inbounds κ = closure.κ[tracer_index]

    νₑ = ▶z_aaf(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / Pr + κ
    ∂z_c = ∂z_aaf(i, j, k, grid, c)
    return κₑ * ∂z_c
end

"""
    ∇_κ_∇c(i, j, k, grid, c, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ c)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇c(i, j, k, grid, c, tracer_index, closure::AbstractSmagorinsky, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, c, tracer_index, closure, diffusivities.νₑ)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, c, tracer_index, closure, diffusivities.νₑ)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, c, tracer_index, closure, diffusivities.νₑ)
)

function calculate_diffusivities!(K, grid, closure::AbstractSmagorinsky, buoyancy, U, C)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds K.νₑ[i, j, k] = ν_ccc(i, j, k, grid, closure, buoyancy, U, C)
            end
        end
    end
    return nothing
end

#####
##### Double dot product of strain on cell edges (currently unused)
#####

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
