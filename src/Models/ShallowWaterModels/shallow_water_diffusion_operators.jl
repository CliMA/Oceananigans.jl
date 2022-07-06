using Oceananigans.Operators
using Oceananigans.Operators: identity1
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, ThreeDimensionalFormulation

using Oceananigans.TurbulenceClosures: 
                        AbstractScalarDiffusivity, 
                        calculate_nonlinear_viscosity!, 
                        viscosity_location, 
                        viscosity, 
                        ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ

import Oceananigans.TurbulenceClosures:
                        DiffusivityFields,
                        calculate_diffusivities,
                        νᶜᶜᶜ

struct ShallowWaterScalarDiffusivity{N, X} <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, ThreeDimensionalFormulation}
    ν :: N
    ξ :: X
end

function ShallowWaterScalarDiffusivity(FT::DataType=Float64; ν=0, ξ=0, discrete_form = false) 
    ν = convert_diffusivity(FT, ν; discrete_form)
    ξ = convert_diffusivity(FT, ξ; discrete_form)
    return ScalarDiffusivity(ν, ξ)
end

# The diffusivity for the shallow water model is calculated as h*ν in order to have a viscous term in the form
# h⁻¹ ∇ ⋅ (hνt) where t is the velocity tensor plus a the trace t = ∇u + (∇u)ᵀ - ξI⋅(∇⋅u)

@inline νᶜᶜᶜ(i, j, k, grid, clo::ShallowWaterScalarDiffusivity, clk, solution) = 
        solution.h[i, j, k] * νᶜᶜᶜ(i, j, k, grid, clk, viscosity_location(clo), viscosity(clo, nothing))

function calculate_diffusivities!(diffusivity_fields, closure::ShallowWaterScalarDiffusivity, model)
    arch = model.architecture
    grid = model.grid
    solution = model.solution
    clock = model.clock
    
    event = launch!(arch, grid, :xyz,
                    calculate_nonlinear_viscosity!,
                    diffusivity_fields.νₑ, grid, closure, clock, solution, 
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

function DiffusivityFields(grid, tracer_names, bcs, ::ShallowWaterScalarDiffusivity)
    default_eddy_viscosity_bcs = (; νₑ = FieldBoundaryConditions(grid, (Center, Center, Center)))
    bcs = merge(default_eddy_viscosity_bcs, bcs)
    return (; νₑ=CenterField(grid, boundary_conditions=bcs.νₑ))
end

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


@inline trace_term_x(i, j, k, grid, clo, U) = - δxᶠᶜᶜ(i, j, k, grid, div_xyᶜᶜᶜ, U.u, U.v) * clo.ξ / Azᶠᶜᶜ(i, j, k, grid)
@inline trace_term_y(i, j, k, grid, clo, U) = - δyᶜᶠᶜ(i, j, k, grid, div_xyᶜᶜᶜ, U.u, U.v) * clo.ξ / Azᶠᶜᶜ(i, j, k, grid)