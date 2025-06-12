using Oceananigans.AbstractOperations: Average
using Oceananigans.Fields: FieldBoundaryConditions
using Oceananigans.Utils: launch!, IterationInterval

using Adapt

using ..TurbulenceClosures:
    AbstractScalarDiffusivity,
    ThreeDimensionalFormulation,
    ExplicitTimeDiscretization,
    convert_diffusivity

import Oceananigans.Utils: with_tracers

import ..TurbulenceClosures:
    viscosity,
    diffusivity,
    κᶠᶜᶜ,
    κᶜᶠᶜ,
    κᶜᶜᶠ,
    compute_diffusivities!,
    build_diffusivity_fields,
    tracer_diffusivities

#####
##### The turbulence closure proposed by Smagorinsky and Lilly.
#####

# struct LagrangianAveragedCoefficient end

struct Smagorinsky{TD, C, P} <: AbstractScalarDiffusivity{TD, ThreeDimensionalFormulation, 2}
    coefficient :: C
    Pr :: P

    function Smagorinsky{TD}(coefficient, Pr) where TD
        P = typeof(Pr)
        C = typeof(coefficient)
        return new{TD, C, P}(coefficient, Pr)
    end
end

@inline viscosity(::Smagorinsky, K) = K.νₑ
@inline diffusivity(closure::Smagorinsky, K, ::Val{id}) where id = K.νₑ / closure.Pr[id]

Adapt.adapt_structure(to, smag::Smagorinsky{TD}) where TD =
    Smagorinsky{TD}(adapt(to, smag.coefficient), adapt(to, smag.Pr))

const ConstantSmagorinsky = Smagorinsky{<:Any, <:Number}

"""
    Smagorinsky([time_discretization::TD = ExplicitTimeDiscretization(), FT=Float64;]
                coefficient = 0.16,
                Pr = 1.0)

Return a `Smagorinsky` type associated with the turbulence closure proposed by
[Smagorinsky1958](@citet) and [Smagorinsky1963](@citet)
which has an eddy viscosity of the form

```
νₑ = (Cˢ * Δᶠ)² * √(2Σ²)
```

and an eddy diffusivity of the form

```
κₑ = νₑ / Pr.
```

where `Δᶠ` is the filter width, `Σ² = ΣᵢⱼΣᵢⱼ` is the double dot product of
the strain tensor `Σᵢⱼ`, `Pr` is the turbulent Prandtl number, `N²` is the
total buoyancy gradient, and `Cb` is a constant the multiplies the Richardson
number modification to the eddy viscosity.

`Cˢ` is the Smagorinsky coefficient and the default value is 0.16, according
to the analysis by [Lilly66](@citet). For other options, see `LillyCoefficient`
and `DynamicCoefficient`.
"""
function Smagorinsky(time_discretization::TD = ExplicitTimeDiscretization(), FT=Oceananigans.defaults.FloatType;
                     coefficient = 0.16, Pr = 1.0) where TD
    Pr = convert_diffusivity(FT, Pr; discrete_form=false)
    return Smagorinsky{TD}(coefficient, Pr)
end

Smagorinsky(FT::DataType; kwargs...) = Smagorinsky(ExplicitTimeDiscretization(), FT; kwargs...)

function with_tracers(tracers, closure::Smagorinsky{TD}) where TD
    Pr = tracer_diffusivities(tracers, closure.Pr)
    return Smagorinsky{TD}(closure.coefficient, Pr)
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

compute_coefficient_fields!(diffusivity_fields, closure, model; parameters) = nothing

function compute_diffusivities!(diffusivity_fields, closure::Smagorinsky, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    buoyancy = model.buoyancy
    velocities = model.velocities
    tracers = model.tracers

    compute_coefficient_fields!(diffusivity_fields, closure, model; parameters)

    launch!(arch, grid, parameters, _compute_smagorinsky_viscosity!,
            diffusivity_fields, grid, closure, buoyancy, velocities, tracers)

    return nothing
end

allocate_coefficient_fields(closure, grid) = NamedTuple()

function build_diffusivity_fields(grid, clock, tracer_names, bcs, closure::Smagorinsky)
    coefficient_fields = allocate_coefficient_fields(closure, grid)

    default_eddy_viscosity_bcs = (; νₑ = FieldBoundaryConditions(grid, (Center, Center, Center)))
    bcs = merge(default_eddy_viscosity_bcs, bcs)
    νₑ = CenterField(grid, boundary_conditions=bcs.νₑ)
    viscosity_nt = (; νₑ)

    return merge(viscosity_nt, coefficient_fields)
end

@inline κᶠᶜᶜ(i, j, k, grid, c::Smagorinsky, K, ::Val{id}, args...) where id = ℑxᶠᵃᵃ(i, j, k, grid, K.νₑ) / c.Pr[id]
@inline κᶜᶠᶜ(i, j, k, grid, c::Smagorinsky, K, ::Val{id}, args...) where id = ℑyᵃᶠᵃ(i, j, k, grid, K.νₑ) / c.Pr[id]
@inline κᶜᶜᶠ(i, j, k, grid, c::Smagorinsky, K, ::Val{id}, args...) where id = ℑzᵃᵃᶠ(i, j, k, grid, K.νₑ) / c.Pr[id]

Base.summary(closure::Smagorinsky) = string("Smagorinsky with coefficient = ", summary(closure.coefficient), ", Pr=$(closure.Pr)")
function Base.show(io::IO, closure::Smagorinsky)
    coefficient_summary = closure.coefficient isa Number ? closure.coefficient : summary(closure.coefficient)
    print(io, "Smagorinsky closure with\n",
              "├── coefficient = ", coefficient_summary, "\n",
              "└── Pr = ", closure.Pr)
end

