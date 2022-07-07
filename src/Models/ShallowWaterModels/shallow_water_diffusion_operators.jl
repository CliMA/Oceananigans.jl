using Oceananigans.Operators
using Oceananigans.Architectures: device, device_event
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
                        DiffusivityFields,
                        calculate_diffusivities!,
                        calculate_nonlinear_viscosity!, 
                        viscosity,
                        with_tracers,
                        calc_νᶜᶜᶜ,
                        νᶜᶜᶜ

struct ShallowWaterScalarDiffusivity{N, X} <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, ThreeDimensionalFormulation}
    ν :: N
    ξ :: X
end

function ShallowWaterScalarDiffusivity(FT::DataType=Float64; ν=0, ξ=0, discrete_form = false) 
    ν = convert_diffusivity(FT, ν; discrete_form)
    ξ = convert_diffusivity(FT, ξ; discrete_form)
    return ShallowWaterScalarDiffusivity(ν, ξ)
end

# We have no tracers in the shallow water diffusivity
with_tracers(tracers, closure::ShallowWaterScalarDiffusivity) = closure
viscosity(closure::ShallowWaterScalarDiffusivity, K) = closure.ν

Adapt.adapt_structure(to, closure::ShallowWaterScalarDiffusivity) = 
    ShallowWaterScalarDiffusivity(Adapt.adapt(to, closure.ν), Adapt.adapt(to, closure.ξ))
     
# The diffusivity for the shallow water model is calculated as h*ν in order to have a viscous term in the form
# h⁻¹ ∇ ⋅ (hν t) where t is the 2D stress tensor plus a trace => t = ∇u + (∇u)ᵀ - ξI⋅(∇⋅u)

@inline calc_νᶜᶜᶜ(i, j, k, grid, closure::ShallowWaterScalarDiffusivity, clock, fields) = 
        solution.h[i, j, k] * νᶜᶜᶜ(i, j, k, grid, clock, viscosity_location(closure), closure.ν, clock, fields)

function calculate_diffusivities!(diffusivity_fields, closure::ShallowWaterScalarDiffusivity, model)
    arch = model.architecture
    grid = model.grid
    solution = model.solution
    clock = model.clock

    model_fields = shallow_water_fields(model.velocities, model.tracers, model.solution, formulation(model))
    
    event = launch!(arch, grid, :xyz,
                    calculate_nonlinear_viscosity!,
                    diffusivity_fields.νₑ, grid, closure, clock, model_fields,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

DiffusivityFields(grid, tracer_names, bcs, ::ShallowWaterScalarDiffusivity)  = (; νₑ=CenterField(grid, boundary_conditions=bcs.h))

#####
##### Diffusion flux divergence operators
#####

@inline shallow_∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, fields, ::ConservativeFormulation) = 
        ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, fields, nothing) + trace_term_x(i, j, k, grid, closure, K, clock, fields)

@inline shallow_∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, fields, ::ConservativeFormulation) = 
        ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, fields, nothing) + trace_term_y(i, j, k, grid, closure, K, clock, fields)

@inline shallow_∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, fields, ::VectorInvariantFormulation) = 
       (∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, fields, nothing) + trace_term_x(i, j, k, grid, closure, K, clock, fields) ) / ℑxᶠᵃᵃ(i, j, k, grid, fields.h)

@inline shallow_∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, fields, ::VectorInvariantFormulation) = 
       (∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, fields, nothing) + trace_term_y(i, j, k, grid, closure, K, clock, fields) ) / ℑyᵃᶠᵃ(i, j, k, grid, fields.h)

@inline trace_term_x(i, j, k, grid, clo, K, clk, fields) = - δxᶠᵃᵃ(i, j, k, grid, ν_σᶜᶜᶜ, clo, K, clk, div_xyᶜᶜᶜ, fields.u, fields.v) * clo.ξ / Azᶠᶜᶜ(i, j, k, grid)
@inline trace_term_y(i, j, k, grid, clo, K, clk, fields) = - δyᵃᶠᵃ(i, j, k, grid, ν_σᶜᶜᶜ, clo, K, clk, div_xyᶜᶜᶜ, fields.u, fields.v) * clo.ξ / Azᶠᶜᶜ(i, j, k, grid)

@inline trace_term_x(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline trace_term_y(i, j, k, grid, ::Nothing, args...) = zero(grid)
