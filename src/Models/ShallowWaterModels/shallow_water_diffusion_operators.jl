using Oceananigans.Operators
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, ThreeDimensionalFormulation

using Oceananigans.TurbulenceClosures:
                        AbstractScalarDiffusivity,
                        convert_diffusivity,
                        viscosity_location,
                        viscosity,
                        ν_σᶜᶜᶜ,
                        ∂ⱼ_τ₁ⱼ,
                        ∂ⱼ_τ₂ⱼ

import Oceananigans.TurbulenceClosures:
                        build_closure_fields,
                        compute_closure_fields!,
                        viscosity,
                        with_tracers,
                        νᶜᶜᶜ

struct ShallowWaterScalarDiffusivity{V, X, N} <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, ThreeDimensionalFormulation, N}
    ν :: V
    ξ :: X
    ShallowWaterScalarDiffusivity{N}(ν::V, ξ::X) where {N, V, X} = new{V, X, N}(ν, ξ)
end

"""
$(TYPEDSIGNATURES)

Return a scalar diffusivity for the shallow water model.

The diffusivity for the shallow water model is calculated as `h * ν` so that we get a
viscous term in the form ``h^{-1} 𝛁 ⋅ (h ν t)``, where ``t`` is the 2D stress tensor plus
a trace, i.e., ``t = 𝛁𝐮 + (𝛁𝐮)^T - ξ I ⋅ (𝛁 ⋅ 𝐮)``.

With the `VectorInvariantFormulation()` (that evolves ``u`` and ``v``) we compute
``h^{-1} 𝛁(ν h 𝛁 t)``, while with the `ConservativeFormulation()` (that evolves
``u h`` and ``v h``) we compute ``𝛁 (ν h 𝛁 t)``.
"""
function ShallowWaterScalarDiffusivity(FT::DataType=Float64; ν=0, ξ=0, discrete_form=false, required_halo_size = 1)
    ν = convert_diffusivity(FT, ν; discrete_form)
    ξ = convert_diffusivity(FT, ξ; discrete_form)
    return ShallowWaterScalarDiffusivity{required_halo_size}(ν, ξ)
end

# We have no tracers in the shallow water diffusivity
with_tracers(tracers, closure::ShallowWaterScalarDiffusivity) = closure
viscosity(closure::ShallowWaterScalarDiffusivity, K) = closure.ν

Adapt.adapt_structure(to, closure::ShallowWaterScalarDiffusivity{B}) where B =
    ShallowWaterScalarDiffusivity{B}(Adapt.adapt(to, closure.ν), Adapt.adapt(to, closure.ξ))

on_architecture(to, closure::ShallowWaterScalarDiffusivity{B}) where B =
    ShallowWaterScalarDiffusivity{B}(on_architecture(to, closure.ν), on_architecture(to, closure.ξ))


# The diffusivity for the shallow water model is calculated as h*ν in order to have a viscous term in the form
# h⁻¹ ∇ ⋅ (hν t) where t is the 2D stress tensor plus a trace => t = ∇u + (∇u)ᵀ - ξI⋅(∇⋅u)

@kernel function _calculate_shallow_water_viscosity!(νₑ, grid, closure, clock, fields)
    i, j, k = @index(Global, NTuple)
    νₑ[i, j, k] = fields.h[i, j, k] * νᶜᶜᶜ(i, j, k, grid, viscosity_location(closure), closure.ν, clock, fields)
end

function compute_closure_fields!(closure_fields, closure::ShallowWaterScalarDiffusivity, model)

    arch  = model.architecture
    grid  = model.grid
    clock = model.clock

    model_fields = shallow_water_fields(model.velocities, model.tracers, model.solution, formulation(model))

    launch!(arch, grid, :xyz,
            _calculate_shallow_water_viscosity!,
            closure_fields.νₑ, grid, closure, clock, model_fields)

    return nothing
end

build_closure_fields(grid, clock, tracer_names, bcs, ::ShallowWaterScalarDiffusivity) = (; νₑ=CenterField(grid, boundary_conditions=bcs.h))

#####
##### Diffusion flux divergence operators
#####

@inline sw_∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, fields, ::ConservativeFormulation) =
        ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, fields, nothing) + trace_term_x(i, j, k, grid, closure, K, clock, fields)

@inline sw_∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, fields, ::ConservativeFormulation) =
        ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, fields, nothing) + trace_term_y(i, j, k, grid, closure, K, clock, fields)

@inline sw_∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, fields, ::VectorInvariantFormulation) =
       (∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, fields, nothing) + trace_term_x(i, j, k, grid, closure, K, clock, fields) ) / ℑxᶠᵃᵃ(i, j, k, grid, fields.h)

@inline sw_∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, fields, ::VectorInvariantFormulation) =
       (∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, fields, nothing) + trace_term_y(i, j, k, grid, closure, K, clock, fields) ) / ℑyᵃᶠᵃ(i, j, k, grid, fields.h)

@inline trace_term_x(i, j, k, grid, clo, K, clk, fields) = - δxᶠᵃᵃ(i, j, k, grid, ν_σᶜᶜᶜ, clo, K, clk, fields, div_xyᶜᶜᶜ, fields.u, fields.v) * clo.ξ * Az⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline trace_term_y(i, j, k, grid, clo, K, clk, fields) = - δyᵃᶠᵃ(i, j, k, grid, ν_σᶜᶜᶜ, clo, K, clk, fields, div_xyᶜᶜᶜ, fields.u, fields.v) * clo.ξ * Az⁻¹ᶠᶜᶜ(i, j, k, grid)

@inline trace_term_x(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline trace_term_y(i, j, k, grid, ::Nothing, args...) = zero(grid)
