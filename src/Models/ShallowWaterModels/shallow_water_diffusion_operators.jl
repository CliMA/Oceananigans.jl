using Oceananigans.Operators
using Oceananigans.Operators: identity1

using Oceananigans.TurbulenceClosures: 
                            AbstractScalarDiffusivity, 
                            calculate_nonlinear_viscosity!, 
                            viscosity_location, 
                            viscosity, 
                            ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ

struct ShallowWaterScalarDiffusivity{N} <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, ThreeDimensionalFormulation}
    ν :: N
end

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
        ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, velocities, tracers, clock, nothing)

@inline shallow_∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, velocities, tracers, solution, ::ConservativeFormulation) = 
        ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, velocities, tracers, clock, nothing) 

@inline shallow_∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, clock, velocities, tracers, solution, ::VectorInvariantFormulation) = 
        ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, K, velocities, tracers, clock, nothing) / ℑxᶠᵃᵃ(i, j, k, grid, solution.h)

@inline shallow_∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, clock, velocities, tracers, solution, ::VectorInvariantFormulation) = 
        ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, K, velocities, tracers, clock, nothing) / ℑyᵃᶠᵃ(i, j, k, grid, solution.h)
