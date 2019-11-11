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

@inline function νᶜᶜᶜ(i, j, k, grid, clo::SmagorinskyLilly{FT}, buoyancy, U, C) where FT
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, U.u, U.v, U.w)
    N² = max(zero(FT), ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_frequency_squared, buoyancy, C))
    Δᶠ = Δᶠ_ccc(i, j, k, grid, clo)
     ς = stability(N², Σ², clo.Cb) # Use unity Prandtl number.

    return νₑ_deardorff(ς, clo.C, Δᶠ, Σ²) + clo.ν
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
@inline function κ_∂x_c(i, j, k, grid, closure::AbstractSmagorinsky,
                        c, ::Val{tracer_index}, νₑ) where tracer_index

    @inbounds Pr = closure.Pr[tracer_index]
    @inbounds κ = closure.κ[tracer_index]

    νₑ = ℑxᶠᵃᵃ(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / Pr + κ
    ∂x_c = ∂xᶠᵃᵃ(i, j, k, grid, c)
    return κₑ * ∂x_c
end

"""
    κ_∂y_c(i, j, k, grid, c, tracer, closure, νₑ)

Return `κ ∂y c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂y_c(i, j, k, grid, closure::AbstractSmagorinsky,
                        c, ::Val{tracer_index}, νₑ) where tracer_index

    @inbounds Pr = closure.Pr[tracer_index]
    @inbounds κ = closure.κ[tracer_index]

    νₑ = ℑyᵃᶠᵃ(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / Pr + κ
    ∂y_c = ∂yᵃᶠᵃ(i, j, k, grid, c)
    return κₑ * ∂y_c
end

"""
    κ_∂z_c(i, j, k, grid, c, tracer, closure, νₑ)

Return `κ ∂z c`, where `κ` is a function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂z_c(i, j, k, grid, closure::AbstractSmagorinsky,
                        c, ::Val{tracer_index}, νₑ) where tracer_index

    @inbounds Pr = closure.Pr[tracer_index]
    @inbounds κ = closure.κ[tracer_index]

    νₑ = ℑzᵃᵃᶠ(i, j, k, grid, νₑ, closure)
    κₑ = (νₑ - closure.ν) / Pr + κ
    ∂z_c = ∂zᵃᵃᶠ(i, j, k, grid, c)
    return κₑ * ∂z_c
end

"""
    ∇_κ_∇c(i, j, k, grid, c, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ c)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇c(i, j, k, grid, closure::AbstractSmagorinsky, c, tracer_index,
               diffusivities, args...) = (
      ∂xᶜᵃᵃ(i, j, k, grid, κ_∂x_c, closure, c, tracer_index, diffusivities.νₑ)
    + ∂yᵃᶜᵃ(i, j, k, grid, κ_∂y_c, closure, c, tracer_index, diffusivities.νₑ)
    + ∂zᵃᵃᶜ(i, j, k, grid, κ_∂z_c, closure, c, tracer_index, diffusivities.νₑ)
)

function calculate_diffusivities!(K, arch, grid, closure::AbstractSmagorinsky, buoyancy, U, C)
    @launch(device(arch), config=launch_config(grid, 3),
            calculate_nonlinear_viscosity!(K.νₑ, grid, closure, buoyancy, U, C))
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
@inline function ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    return (
                    tr_Σ²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `ffc`."
@inline function ΣᵢⱼΣᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)
    return (
                  ℑxyᶠᶠᵃ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 *    Σ₁₂²(i, j, k, grid, u, v, w)
            + 2 * ℑyzᵃᶠᶜ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ℑxzᶠᵃᶜ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `fcf`."
@inline function ΣᵢⱼΣᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑxzᶠᵃᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑyzᵃᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *    Σ₁₃²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶠᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `cff`."
@inline function ΣᵢⱼΣᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑyzᵃᶠᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑxzᶜᵃᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ℑxyᶜᶠᵃ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *    Σ₂₃²(i, j, k, grid, u, v, w)
            )
end

@inline function ΣᵢⱼΣᵢⱼᶜᶜᶠ(i, j, k, grid, u, v, w)
    return (
                    ℑzᵃᵃᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑxyzᶜᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *   ℑxᶜᵃᵃ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *   ℑyᵃᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end
