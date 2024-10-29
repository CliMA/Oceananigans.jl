using Oceananigans.AbstractOperations: Average
using Oceananigans.Fields: FieldBoundaryConditions
using Oceananigans.Utils: launch!, IterationInterval

using Adapt

using ..TurbulenceClosures:
    AbstractScalarDiffusivity,
    ThreeDimensionalFormulation,
    ExplicitTimeDiscretization,
    convert_diffusivity

import ..TurbulenceClosures:
    viscosity,
    diffusivity,
    κᶠᶜᶜ,
    κᶜᶠᶜ,
    κᶜᶜᶠ,
    compute_diffusivities!,
    DiffusivityFields,
    tracer_diffusivities

import Oceananigans.Utils: with_tracers

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

const ConstantSmagorinsky = Smagorinsky{<:Any, <:Number}

"""
    Smagorinsky([time_discretization::TD = ExplicitTimeDiscretization(), FT=Float64;]
                coefficient = 0.16,
                Pr = 1.0)

Return a `Smagorinsky` type associated with the turbulence closure proposed by
[Lilly62](@citet), [Smagorinsky1958](@citet), [Smagorinsky1963](@citet), and [Lilly66](@citet),
which has an eddy viscosity of the form

```
νₑ = (C * Δᶠ)² * √(2Σ²)
```

and an eddy diffusivity of the form

```
κₑ = νₑ / Pr.
```

where `Δᶠ` is the filter width, `Σ² = ΣᵢⱼΣᵢⱼ` is the double dot product of
the strain tensor `Σᵢⱼ`, `Pr` is the turbulent Prandtl number, `N²` is the
total buoyancy gradient, and `Cb` is a constant the multiplies the Richardson
number modification to the eddy viscosity.
"""
function Smagorinsky(time_discretization = ExplicitTimeDiscretization(), FT=Float64;
                     coefficient = 0.16, Pr = 1.0)
    TD = typeof(time_discretization)
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

function DiffusivityFields(grid, tracer_names, bcs, closure::Smagorinsky)
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

Base.summary(closure::Smagorinsky) = string("Smagorinsky: coefficient=", summary(closure.coefficient), ", Pr=$(closure.Pr)")
Base.show(io::IO, closure::Smagorinsky) = print(io, summary(closure))

