function assemble_dissipation!(simulation, dissipation)
    model = simulation.model

    for tracer_name in keys(dissipation.advective_production)
        assemble_dissipation!(dissipation, model, tracer_name)
    end

    return nothing
end

@inline c★(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] + cⁿ[i, j, k]) / 2
@inline c²(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] * cⁿ[i, j, k])

@inline ζ★(i, j, k, grid, uⁿ⁺¹, vⁿ⁺¹, ζⁿ) = @inbounds (ζ₃ᶠᶠᶜ(i, j, k, grid, uⁿ⁺¹, vⁿ⁺¹) + ζⁿ[i, j, k]) / 2
@inline ζ²(i, j, k, grid, uⁿ⁺¹, vⁿ⁺¹, ζⁿ) = @inbounds (ζ₃ᶠᶠᶜ(i, j, k, grid, uⁿ⁺¹, vⁿ⁺¹) * ζⁿ[i, j, k])

function assemble_dissipation!(dissipation, model, tracer_name::Symbol)
    
    
    arch = architecture(grid)
    χ = simulation.model.timestepper.χ

    # General velocities
    Uⁿ⁺¹ = model.velocities
    Uⁿ   = dissipation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation.previous_state.Uⁿ⁻¹

    cⁿ⁺¹ = tracer_symbol == :ζ ? nothing : model.tracers[tracer_name]
    cⁿ   = dissipation.previous_state[tracer_name]


    _assemble_advective_dissipation! = assemble_advective_dissipation_kernel(Val(tracer_name))
    _assemble_diffusive_dissipation! = assemble_diffusive_dissipation_kernel(Val(tracer_name))

    ####
    #### Assemble the advective dissipation
    ####

    P    = dissipation.advective_production[tracer_name]
    Fⁿ   = dissipation.advective_fluxes.Fⁿ[tracer_name]
    Fⁿ⁻¹ = dissipation.advective_fluxes.Fⁿ⁻¹[tracer_name]

    launch!(arch, grid, :xyz, _assemble_advective_dissipation!, P, grid, χ, Fⁿ, Fⁿ⁻¹, Uⁿ⁺¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)

    ####
    #### Assemble the diffusive dissipation
    #### 

    K    = dissipation.diffusive_production[tracer_name]
    Vⁿ   = dissipation.advective_fluxes.Vⁿ[tracer_name]
    Vⁿ⁻¹ = dissipation.advective_fluxes.Vⁿ⁻¹[tracer_name]

    launch!(arch, grid, params, _assemble_diffusive_dissipation!, K, grid, χ, Vⁿ, Vⁿ⁻¹, Uⁿ⁺¹, cⁿ⁺¹, cⁿ)

    return nothing
end

assemble_advective_dissipation_kernel(val_tracer_name) = _assemble_advective_tracer_dissipation!
assemble_advective_dissipation_kernel(::Val{:ζ})       = _assemble_advective_vorticity_dissipation!
assemble_diffusive_dissipation_kernel(val_tracer_name) = _assemble_diffusive_tracer_dissipation!
assemble_diffusive_dissipation_kernel(::Val{:ζ})       = _assemble_diffusive_vorticity_dissipation!
