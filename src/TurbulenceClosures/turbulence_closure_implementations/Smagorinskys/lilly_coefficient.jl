using Oceananigans.BuoyancyFormulations: ∂z_b

struct LillyCoefficient{FT}
    smagorinsky :: FT
    reduction_factor :: FT
end

"""
    LillyCoefficient([FT=Float64;] smagorinsky=0.16, reduction_factor=1)

When used with `Smagorinsky`, it calculates the Smagorinsky coefficient according to closure
proposed by [Lilly62](@citet), and [Lilly66](@citet), which has an eddy viscosity of the form

```
νₑ = (Cˢ * Δᶠ)² * √(2Σ²) * √(1 - Cb * N² / Σ²)
```

where `C²` is the Smagorinsky coefficient, `Δᶠ` is the filter width, `Σ² = ΣᵢⱼΣᵢⱼ` is the double dot
product of the strain tensor `Σᵢⱼ`, `N²` is the total buoyancy gradient, and `Cb` is a constant the
multiplies the Richardson number modification to the eddy viscosity.

Arguments
=========

* `FT`: Float type; default `Float64`.

Keyword arguments
=================

* `smagorinsky`: Smagorinsky coefficient `Cˢ`. Default value is 0.16 as obtained by Lilly (1966).

* `reduction_factor`: Buoyancy term multipler `Cb` based on Lilly (1962) (`reduction_factor = 0`
        turns it off, `reduction_factor ≠ 0` turns it on.
        Typically, and according to the original work by Lilly (1962), `Cb = 1 / Pr`.)

References
==========

Lilly, D. K. "On the numerical simulation of buoyant convection." Tellus (1962)

Lilly, D. K. "The representation of small-scale turbulence in numerical simulation experiments."
    NCAR Manuscript No. 281, 0, (1966)
"""
LillyCoefficient(FT=Oceananigans.defaults.FloatType; smagorinsky=0.16, reduction_factor=1) =
    LillyCoefficient(convert(FT, smagorinsky), convert(FT, reduction_factor))

const SmagorinskyLilly = Smagorinsky{<:Any, <:LillyCoefficient}

"""
    SmagorinskyLilly([time_discretization::TD = ExplicitTimeDiscretization(), FT=Float64;] C=0.16, Cb=1.0, Pr=1.0)

Return a `SmagorinskyLilly` type associated with the turbulence closure proposed by
[Lilly62](@citet), [Smagorinsky1958](@citet), [Smagorinsky1963](@citet), and [Lilly66](@citet),
which has an eddy viscosity of the form

```
νₑ = (Cˢ * Δᶠ)² * √(2Σ²) * √(1 - Cb * N² / Σ²)
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

* `Cˢ`: Smagorinsky coefficient. Default value is 0.16 as obtained by Lilly (1966).

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
function SmagorinskyLilly(time_discretization=ExplicitTimeDiscretization(), FT=Oceananigans.defaults.FloatType; C=0.16, Cb=1, Pr=1)
    coefficient = LillyCoefficient(FT, smagorinsky=C, reduction_factor=Cb)
    TD = typeof(time_discretization)
    Pr = convert_diffusivity(FT, Pr; discrete_form=false)
    return Smagorinsky{TD}(coefficient, Pr)
end

SmagorinskyLilly(FT::DataType; kwargs...) = SmagorinskyLilly(ExplicitTimeDiscretization(), FT; kwargs...)

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

@inline function square_smagorinsky_coefficient(i, j, k, grid, closure::SmagorinskyLilly,
                                                diffusivity_fields, Σ², buoyancy, tracers)
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    c₀ = closure.coefficient.smagorinsky
    cᵇ = closure.coefficient.reduction_factor
    ς  = stability(N², Σ², cᵇ) # Use unity Prandtl number.
    return ς * c₀^2
end

Base.summary(dc::LillyCoefficient) = string("LillyCoefficient(smagorinsky = $(dc.smagorinsky), reduction_factor = $(dc.reduction_factor))")
Base.show(io::IO, dc::LillyCoefficient) = print(io, "LillyCoefficient with\n",
                                                    "├── Smagorinsky coefficient = ", dc.smagorinsky, "\n",
                                                    "└── reduction_factor = ", dc.reduction_factor)


