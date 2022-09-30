using Oceananigans
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: device_event, device
using Oceananigans.TurbulenceClosures: AbstractScalarBiharmonicDiffusivity, DiffusivityFields, calculate_nonlinear_viscosity!
import Oceananigans.TurbulenceClosures: with_tracers, viscosity, diffusivity, DiffusivityFields, calc_νᶜᶜᶜ, calculate_diffusivities!

struct LeithBiharmonicViscosity{FT, F} <: AbstractScalarBiharmonicDiffusivity{F}
    C₁ :: FT
    C₂ :: FT

    LeithBiharmonicViscosity{F}(C₁::FT, C₂::FT) where {F, FT} = new{FT, F}(C₁, C₂)
end

function DiffusivityFields(grid, tracer_names, bcs, ::LeithBiharmonicViscosity)
    default_eddy_viscosity_bcs = (; νₑ = FieldBoundaryConditions(grid, (Center, Center, Center)))
    bcs = merge(default_eddy_viscosity_bcs, bcs)
    return (; νₑ=CenterField(grid, boundary_conditions=bcs.νₑ))
end

with_tracers(tracers, closure::LeithBiharmonicViscosity) = closure

LeithBiharmonicViscosity(formulation=HorizontalDivergenceFormulation(); C_vort = 2.0, C_div = 2.0) = 
            LeithBiharmonicViscosity{typeof(formulation)}((C_vort / π)^6 / 8, (C_div / π)^6 / 8)

@inline viscosity(::LeithBiharmonicViscosity, K) = K.νₑ
@inline diffusivity(::LeithBiharmonicViscosity, K, ::Val{id}) where id = zero(eltype(K.νₑ))

@inline Δ²ᶜᶜᶜ(i, j, k, grid) = (1 / (1 / Δxᶜᶜᶜ(i, j, k, grid)^2 + 1 / Δyᶜᶜᶜ(i, j, k, grid)^2))

using Oceananigans.Operators: ℑxyz

@inline function calc_νᶜᶜᶜ(i, j, k, grid, closure::LeithBiharmonicViscosity, fields)
	
    ∂xζ = ℑyᵃᶜᵃ(i, j, k, grid, ∂xᶜᶠᶜ, ζ₃ᶠᶠᶜ, fields.u, fields.v)
    ∂yζ = ℑxᶜᵃᵃ(i, j, k, grid, ∂yᶠᶜᶜ, ζ₃ᶠᶠᶜ, fields.u, fields.v)
    ∂xδ = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, div_xyᶜᶜᶜ, fields.u, fields.v)
    ∂yδ = ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, div_xyᶜᶜᶜ, fields.u, fields.v)
   
    dynamic_visc = sqrt(closure.C₁ * (∂xζ^2 + ∂yζ^2) + closure.C₂ * (∂xδ^2 + ∂yδ^2))

    return Δ²ᶜᶜᶜ(i, j, k, grid)^2.5 * dynamic_visc
end

function calculate_diffusivities!(diffusivity_fields, closure::LeithBiharmonicViscosity, model)

    arch = model.architecture
    grid = model.grid

    event = launch!(arch, grid, :xyz,
                    calculate_nonlinear_viscosity!,
                    diffusivity_fields.νₑ, grid, closure, fields(model),
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end
