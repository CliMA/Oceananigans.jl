#####
##### The turbulence closure proposed by Smagorinsky and Lilly.
##### We also call this 'Constant Smagorinsky'.
#####

struct DirectionallyAveragedCoefficient end
struct LagrangianAveragedCoefficient end
struct LillyCoefficient end

struct SmagorinskyLilly{TD, CO, FT, P} <: AbstractScalarDiffusivity{TD, ThreeDimensionalFormulation, 2}
    coefficient :: CO
    Cb :: FT
    Pr :: P

    function SmagorinskyLilly{TD}(coefficient, Cb, Pr) where TD
        FT = typeof(Cb)
        Pr = convert_diffusivity(FT, Pr; discrete_form=false)
        P = typeof(Pr)
        CO = typeof(coefficient)
        return new{TD, CO, FT, P}(coefficient, Cb, Pr)
    end
end

const DirectionallyAveragedSmagorinsky = SmagorinskyLilly{<:Any, <:DirectionallyAveragedCoefficient}
const ConstantSmagorinsky = SmagorinskyLilly{<:Any, <:Number}

@inline viscosity(::SmagorinskyLilly, K) = K.νₑ
@inline diffusivity(closure::SmagorinskyLilly, K, ::Val{id}) where id = K.νₑ / closure.Pr[id]

"""
    SmagorinskyLilly([time_discretization::TD = ExplicitTimeDiscretization(), FT=Float64;]
                     C = 0.16,
                     Cb = 1.0,
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
                          Cb = 1.0,
                          Pr = 1.0)

    TD = typeof(time_discretization)
    Cb = convert(FT, Cb)
    Pr = convert_diffusivity(FT, Pr; discrete_form=false)
    return SmagorinskyLilly{TD}(coefficient, Cb, Pr)
end

SmagorinskyLilly(FT::DataType; kwargs...) = SmagorinskyLilly(ExplicitTimeDiscretization(), FT; kwargs...)

function with_tracers(tracers, closure::SmagorinskyLilly{TD}) where TD
    Pr = tracer_diffusivities(tracers, closure.Pr)
    return SmagorinskyLilly{TD}(closure.coefficient, closure.Cb, Pr)
end

"""
    stability(N², Σ², Cb)

Return the stability function

```math
    \\sqrt(1 - Cb N^2 / Σ^2 )
```

when ``N^2 > 0``, and 1 otherwise.
"""
@inline function stability(N²::FT, Σ²::FT, Cb::FT) where FT
    N²⁺ = max(zero(FT), N²) # clip
    ς² = one(FT) - min(one(FT), Cb * N²⁺ / Σ²)
    return ifelse(Σ²==0, zero(FT), sqrt(ς²))
end

@kernel function _compute_smagorinsky_viscosity!(diffusivity_fields, grid, closure, buoyancy, velocities, tracers)
    i, j, k = @index(Global, NTuple)

    νₑ = diffusivity_fields.νₑ

    # Strain tensor dot product
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)

    # Stability function
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    ς  = stability(N², Σ², closure.Cb) # Use unity Prandtl number.

    # Filter width
    Δ³ = Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid)
    Δᶠ = cbrt(Δ³)
    cˢ² = square_smagorinsky_coefficient(i, j, k, grid, closure, diffusivity_fields)

    @inbounds νₑ[i, j, k] = ς * cˢ² * Δᶠ^2 * sqrt(2Σ²)
end

@inline square_smagorinsky_coefficient(i, j, k, grid, c::ConstantSmagorinsky, K) = c.coefficient^2

@inline function square_smagorinsky_coefficient(i, j, k, grid, ::DirectionallyAveragedSmagorinsky, K)
    LM_avg = K.LM_avg
    MM_avg = K.MM_avg

    @inbounds begin
        LM⁺ = max(LM_avg[i, j, k], zero(grid))
        MM = MM_avg[i, j, k]
    end

    return ifelse(MM == 0, zero(grid), LM⁺ / MM)
end

@kernel function _compute_LM_MM!(LM, MM, grid, u, v, w)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        LM[i, j, k] = LᵢⱼMᵢⱼ_ccc(i, j, k, grid, u, v, w)
        MM[i, j, k] = MᵢⱼMᵢⱼ_ccc(i, j, k, grid, u, v, w)
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

@inline κᶠᶜᶜ(i, j, k, grid, closure::SmagorinskyLilly, K, ::Val{id}, args...) where id = ℑxᶠᵃᵃ(i, j, k, grid, K.νₑ) / closure.Pr[id]
@inline κᶜᶠᶜ(i, j, k, grid, closure::SmagorinskyLilly, K, ::Val{id}, args...) where id = ℑyᵃᶠᵃ(i, j, k, grid, K.νₑ) / closure.Pr[id]
@inline κᶜᶜᶠ(i, j, k, grid, closure::SmagorinskyLilly, K, ::Val{id}, args...) where id = ℑzᵃᵃᶠ(i, j, k, grid, K.νₑ) / closure.Pr[id]

#####
##### Double dot product of strain on cell edges (currently unused)
#####

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
            + 2 *   Σ₁₂²(i, j, k, grid, u, v, w)
            + 2 * ℑyzᵃᶠᶜ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ℑxzᶠᵃᶜ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `fcf`."
@inline function ΣᵢⱼΣᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑxzᶠᵃᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑyzᵃᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *   Σ₁₃²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶠᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `cff`."
@inline function ΣᵢⱼΣᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑyzᵃᶠᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑxzᶜᵃᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ℑxyᶜᶠᵃ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *   Σ₂₃²(i, j, k, grid, u, v, w)
            )
end

"Return the double dot product of strain at `ccf`."
@inline function ΣᵢⱼΣᵢⱼᶜᶜᶠ(i, j, k, grid, u, v, w)
    return (
                    ℑzᵃᵃᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑxyzᶜᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *   ℑxᶜᵃᵃ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *   ℑyᵃᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

Base.summary(closure::SmagorinskyLilly) =
    string("SmagorinskyLilly: coefficient=$(closure.coefficient), Cb=$(closure.Cb), Pr=$(closure.Pr)")
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

        LM_avg = Field(Average(LM, dims=(1, 2)))
        MM_avg = Field(Average(MM, dims=(1, 2)))

        return (; νₑ, LM, MM, LM_avg, MM_avg)
    else closure.coefficient isa Number
        return (; νₑ)
    end
end

