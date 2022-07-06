using Oceananigans.Operators
using Oceananigans.Operators: identity1
using Oceananigans.Architectures: device, device_event
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, ThreeDimensionalFormulation

using Oceananigans.TurbulenceClosures: 
                        AbstractScalarDiffusivity, 
                        convert_diffusivity,
                        viscosity_location, 
                        viscosity, 
                        ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ

import Oceananigans.TurbulenceClosures:
                        DiffusivityFields,
                        calculate_diffusivities!,
                        calculate_nonlinear_viscosity!, 
                        viscosity,
                        with_tracers,
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

# The diffusivity for the shallow water model is calculated as h*ν in order to have a viscous term in the form
# h⁻¹ ∇ ⋅ (hν t) where t is the 2D stress tensor plus a trace => t = ∇u + (∇u)ᵀ - ξI⋅(∇⋅u)

@inline νᶜᶜᶜ(i, j, k, grid, closure::ShallowWaterScalarDiffusivity, clock, solution, C) = 
        solution.h[i, j, k] * νᶜᶜᶜ(i, j, k, grid, clock, viscosity_location(closure), closure.ν)

function calculate_diffusivities!(diffusivity_fields, closure::ShallowWaterScalarDiffusivity, model)
    arch = model.architecture
    grid = model.grid
    solution = model.solution
    clock = model.clock
    
    event = launch!(arch, grid, :xyz,
                    calculate_nonlinear_viscosity!,
                    diffusivity_fields.νₑ, grid, closure, clock, solution, nothing,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

DiffusivityFields(grid, tracer_names, bcs, ::ShallowWaterScalarDiffusivity)  = (; νₑ=CenterField(grid, boundary_conditions=bcs.h))

#####
##### Diffusion flux divergence operators
#####

@inline shallow_∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, velocities, tracers, solution, ::ConservativeFormulation) = 
        ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, velocities, tracers, clock, nothing) + trace_term_x(i, j, k, grid, closure, velocities)

@inline shallow_∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, velocities, tracers, solution, ::ConservativeFormulation) = 
        ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, velocities, tracers, clock, nothing) + trace_term_y(i, j, k, grid, closure, velocities)

@inline shallow_∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, velocities, tracers, solution, ::VectorInvariantFormulation) = 
       (∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, velocities, tracers, clock, nothing) + trace_term_x(i, j, k, grid, closure, velocities) ) / ℑxᶠᵃᵃ(i, j, k, grid, solution.h)

@inline shallow_∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, velocities, tracers, solution, ::VectorInvariantFormulation) = 
       (∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, velocities, tracers, clock, nothing) + trace_term_y(i, j, k, grid, closure, velocities) ) / ℑyᵃᶠᵃ(i, j, k, grid, solution.h)

@inline trace_term_x(i, j, k, grid, clo, U) = - δxᶠᵃᵃ(i, j, k, grid, div_xyᶜᶜᶜ, U.u, U.v) * clo.ξ / Azᶠᶜᶜ(i, j, k, grid)
@inline trace_term_y(i, j, k, grid, clo, U) = - δyᵃᶠᵃ(i, j, k, grid, div_xyᶜᶜᶜ, U.u, U.v) * clo.ξ / Azᶠᶜᶜ(i, j, k, grid)