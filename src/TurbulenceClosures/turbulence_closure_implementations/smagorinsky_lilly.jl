#####
##### The turbulence closure proposed by Smagorinsky and Lilly.
##### We also call this 'Constant Smagorinsky'.
#####

struct DirectionallyAveragedCoefficient{D}
    dims :: D
end

DirectionallyAveragedCoefficient(; dims) = DirectionallyAveragedCoefficient(dims)

struct LagrangianAveragedCoefficient end

struct LillyCoefficient{FT}
    smagorinsky :: FT
    reduction_factor :: FT
end

LillyCoefficient(FT=Float64, smagorinsky=0.16, reduction_factor=1.0) =
    LillyCoefficient(convert(FT, smagorinsky), convert(FT, reduction_factor))

struct SmagorinskyLilly{TD, C, P} <: AbstractScalarDiffusivity{TD, ThreeDimensionalFormulation, 2}
    coefficient :: C
    Pr :: P

    function SmagorinskyLilly{TD}(coefficient, Pr) where TD
        P = typeof(Pr)
        C = typeof(coefficient)
        return new{TD, C, P}(coefficient, Pr)
    end
end

const ConstantSmagorinsky = SmagorinskyLilly{<:Any, <:Number}
const DirectionallyAveragedSmagorinsky = SmagorinskyLilly{<:Any, <:DirectionallyAveragedCoefficient}
const SmagLilly = SmagorinskyLilly{<:Any, <:LillyCoefficient}

@inline viscosity(::SmagorinskyLilly, K) = K.νₑ
@inline diffusivity(closure::SmagorinskyLilly, K, ::Val{id}) where id = K.νₑ / closure.Pr[id]

"""
    SmagorinskyLilly([time_discretization::TD = ExplicitTimeDiscretization(), FT=Float64;]
                     coefficient = 0.16,
                     Pr = 1.0)

Return a `SmagorinskyLilly` type associated with the turbulence closure proposed by
[Lilly62](@citet), [Smagorinsky1958](@citet), [Smagorinsky1963](@citet), and [Lilly66](@citet),
which has an eddy viscosity of the form

```
νₑ = (C * Δᶠ)² * √(2Σ²) * √(1 - Cb * N² / Σ²)
```

and an eddy diffusivity of the form

```
κₑ = νₑ / Pr
```

where `Δᶠ` is the filter width, `Σ² = ΣᵢⱼΣᵢⱼ` is the double dot product of
the strain tensor `Σᵢⱼ`, `Pr` is the turbulent Prandtl number, `N²` is the
total buoyancy gradient, and `Cb` is a constant the multiplies the Richardson
number modification to the eddy viscosity.

Arguments
=========

* `time_discretization`: Either `ExplicitTimeDiscretization()` or `VerticallyImplicitTimeDiscretization()`, 
                         which integrates the terms involving only ``z``-derivatives in the
                         viscous and diffusive fluxes with an implicit time discretization.
                         Default `ExplicitTimeDiscretization()`.

* `FT`: Float type; default `Float64`.

Keyword arguments
=================

* `C`: Smagorinsky constant. Default value is 0.16 as obtained by Lilly (1966).

* `Cb`: Buoyancy term multipler based on Lilly (1962) (`Cb = 0` turns it off, `Cb ≠ 0` turns it on.
        Typically, and according to the original work by Lilly (1962), `Cb = 1 / Pr`.)

* `Pr`: Turbulent Prandtl numbers for each tracer. Either a constant applied to every
        tracer, or a `NamedTuple` with fields for each tracer individually.

References
==========

Smagorinsky, J. "On the numerical integration of the primitive equations of motion for
    baroclinic flow in a closed region." Monthly Weather Review (1958)

Lilly, D. K. "On the numerical simulation of buoyant convection." Tellus (1962)

Smagorinsky, J. "General circulation experiments with the primitive equations: I.
    The basic experiment." Monthly Weather Review (1963)

Lilly, D. K. "The representation of small-scale turbulence in numerical simulation experiments." 
    NCAR Manuscript No. 281, 0, (1966)
"""
function SmagorinskyLilly(time_discretization = ExplicitTimeDiscretization(), FT=Float64;
                          coefficient = 0.16,
                          Pr = 1.0)

    TD = typeof(time_discretization)
    Pr = convert_diffusivity(FT, Pr; discrete_form=false)
    return SmagorinskyLilly{TD}(coefficient, Pr)
end

SmagorinskyLilly(FT::DataType; kwargs...) = SmagorinskyLilly(ExplicitTimeDiscretization(), FT; kwargs...)

function with_tracers(tracers, closure::SmagorinskyLilly{TD}) where TD
    Pr = tracer_diffusivities(tracers, closure.Pr)
    return SmagorinskyLilly{TD}(closure.coefficient, Pr)
end

@kernel function _compute_smagorinsky_viscosity!(diffusivity_fields, grid, closure, buoyancy, velocities, tracers)
    i, j, k = @index(Global, NTuple)

    # Strain tensor dot product
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)

    # Filter width
    Δ³ = Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    Δᶠ = cbrt(Δ³)
    cˢ² = square_smagorinsky_coefficient(i, j, k, grid, closure, diffusivity_fields, Σ², buoyancy, tracers)

    νₑ = diffusivity_fields.νₑ

    @inbounds νₑ[i, j, k] = cˢ² * Δᶠ^2 * sqrt(2Σ²)
end

@inline square_smagorinsky_coefficient(i, j, k, grid, c::ConstantSmagorinsky, args...) = c.coefficient^2

@inline function square_smagorinsky_coefficient(i, j, k, grid, ::DirectionallyAveragedSmagorinsky, K, args...)
    LM_avg = K.LM_avg
    MM_avg = K.MM_avg

    @inbounds begin
        LM⁺ = max(LM_avg[i, j, k], zero(grid))
        MM = MM_avg[i, j, k]
    end

    return ifelse(MM == 0, zero(grid), LM⁺ / MM)
end

"""
    stability(N², Σ², Cb)

Return the stability function

```math
    \\sqrt(1 - Cb N^2 / Σ^2 )
```

when ``N^2 > 0``, and 1 otherwise.
"""
@inline function stability(N²::FT, Σ²::FT, cᵇ::FT) where FT
    N²⁺ = max(zero(FT), N²) # clip
    ς² = one(FT) - min(one(FT), cᵇ * N²⁺ / Σ²)
    return ifelse(Σ²==0, zero(FT), sqrt(ς²))
end

@inline function square_smagorinsky_coefficient(i, j, k, grid, closure::SmagLilly, K, Σ², buoyancy, tracers)
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    c₀ = closure.coefficient.smagorinsky
    cᵇ = closure.coefficient.reduction_factor
    ς  = stability(N², Σ², cᵇ) # Use unity Prandtl number.
    return ς * c₀^2
end

@kernel function _compute_LM_MM!(LM, MM, grid, u, v, w)
    i, j, k = @index(Global, NTuple)

    Sᶜᶜᶜ = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w))
    S̄ᶜᶜᶜ = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w))

    L₁₁ = L₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₂₂ = L₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₃₃ = L₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₁₂ = L₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₁₃ = L₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₂₃ = L₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w)

    M₁₁ = M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Sᶜᶜᶜ, S̄ᶜᶜᶜ)
    M₂₂ = M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Sᶜᶜᶜ, S̄ᶜᶜᶜ)
    M₃₃ = M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Sᶜᶜᶜ, S̄ᶜᶜᶜ)
    M₁₂ = M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1) 
    M₁₃ = M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1) 
    M₂₃ = M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)

    @inbounds begin
        LM[i, j, k] = L₁₁ * M₁₁ + L₂₂ * M₂₂ + L₃₃ * M₃₃ + 2L₁₂ * M₁₂ + 2L₁₃ * M₁₃ + 2L₂₃ * M₂₃
        MM[i, j, k] = M₁₁ * M₁₁ + M₂₂ * M₂₂ + M₃₃ * M₃₃ + 2M₁₂ * M₁₂ + 2M₁₃ * M₁₃ + 2M₂₃ * M₂₃
    end
end

function compute_diffusivities!(diffusivity_fields, closure::SmagorinskyLilly, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    buoyancy = model.buoyancy
    velocities = model.velocities
    tracers = model.tracers

    if closure.coefficient isa DirectionallyAveragedCoefficient
        LM = diffusivity_fields.LM
        MM = diffusivity_fields.MM
        u, v, w = velocities
        launch!(arch, grid, :xyz, _compute_LM_MM!, LM, MM, grid, u, v, w)

        LM_avg = diffusivity_fields.LM_avg
        MM_avg = diffusivity_fields.MM_avg
        compute!(LM_avg)
        compute!(MM_avg)
    end

    launch!(arch, grid, parameters, _compute_smagorinsky_viscosity!,
            diffusivity_fields, grid, closure, buoyancy, velocities, tracers)

    return nothing
end

@inline κᶠᶜᶜ(i, j, k, grid, c::SmagorinskyLilly, K, ::Val{id}, args...) where id = ℑxᶠᵃᵃ(i, j, k, grid, K.νₑ) / c.Pr[id]
@inline κᶜᶠᶜ(i, j, k, grid, c::SmagorinskyLilly, K, ::Val{id}, args...) where id = ℑyᵃᶠᵃ(i, j, k, grid, K.νₑ) / c.Pr[id]
@inline κᶜᶜᶠ(i, j, k, grid, c::SmagorinskyLilly, K, ::Val{id}, args...) where id = ℑzᵃᵃᶠ(i, j, k, grid, K.νₑ) / c.Pr[id]

#####
##### Double dot product of strain on cell edges (currently unused)
#####

# tr_Σ² : ccc
#   Σ₁₂ : ffc
#   Σ₁₃ : fcf
#   Σ₂₃ : cff

"Return the double dot product of strain at `ccc`."
@inline ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) =      tr_Σ²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)

"Return the double dot product of strain at `ffc`."
@inline ΣᵢⱼΣᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w) =     ℑxyᶠᶠᵃ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 *   Σ₁₂²(i, j, k, grid, u, v, w) +
                                            2 * ℑyzᵃᶠᶜ(i, j, k, grid, Σ₁₃², u, v, w) + 
                                            2 * ℑxzᶠᵃᶜ(i, j, k, grid, Σ₂₃², u, v, w)

"Return the double dot product of strain at `fcf`."
@inline ΣᵢⱼΣᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w) =     ℑxzᶠᵃᶠ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 * ℑyzᵃᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 *   Σ₁₃²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶠᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)

"Return the double dot product of strain at `cff`."
@inline ΣᵢⱼΣᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w) =     ℑyzᵃᶠᶠ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 * ℑxzᶜᵃᶠ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxyᶜᶠᵃ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 *   Σ₂₃²(i, j, k, grid, u, v, w)

"Return the double dot product of strain at `ccf`."
@inline ΣᵢⱼΣᵢⱼᶜᶜᶠ(i, j, k, grid, u, v, w) =       ℑzᵃᵃᶠ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 * ℑxyzᶜᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 *   ℑxᶜᵃᵃ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 *   ℑyᵃᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)

Base.summary(closure::SmagorinskyLilly) = string("Smagorinsky: coefficient=",
                                                 summary(closure.coefficient), ", Pr=$(closure.Pr)")
    
Base.show(io::IO, closure::SmagorinskyLilly) = print(io, summary(closure))

#####
##### For closures that only require an eddy viscosity νₑ field.
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::SmagorinskyLilly)

    default_eddy_viscosity_bcs = (; νₑ = FieldBoundaryConditions(grid, (Center, Center, Center)))
    bcs = merge(default_eddy_viscosity_bcs, bcs)
    νₑ = CenterField(grid, boundary_conditions=bcs.νₑ)

    if closure.coefficient isa DirectionallyAveragedCoefficient
        LM = CenterField(grid)
        MM = CenterField(grid)

        LM_avg = Field(Average(LM, dims=closure.coefficient.dims))
        MM_avg = Field(Average(MM, dims=closure.coefficient.dims))

        return (; νₑ, LM, MM, LM_avg, MM_avg)

    else closure.coefficient isa Number
        return (; νₑ)
    end
end

