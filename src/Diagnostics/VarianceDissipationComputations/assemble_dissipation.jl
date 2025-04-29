function assemble_dissipation!(model, dissipation)

    for tracer_name in keys(dissipation.advective_production)
        assemble_dissipation!(dissipation, model, tracer_name)
    end

    return nothing
end

@inline c★(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] + cⁿ[i, j, k]) / 2
@inline c²(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] * cⁿ[i, j, k])

function assemble_dissipation!(dissipation, model, tracer_name::Symbol)
    
    grid = model.grid
    arch = architecture(grid)
    χ = model.timestepper.χ

    # General velocities
    Uⁿ⁺¹ = model.velocities
    Uⁿ   = dissipation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation.previous_state.Uⁿ⁻¹

    cⁿ⁺¹ = model.tracers[tracer_name]
    cⁿ   = dissipation.previous_state[tracer_name]
    sz   = size(cⁿ⁺¹.data)
    of   = cⁿ⁺¹.data.offsets

    params = KernelParameters(sz, of)

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
    Vⁿ   = dissipation.diffusive_fluxes.Vⁿ[tracer_name]
    Vⁿ⁻¹ = dissipation.diffusive_fluxes.Vⁿ⁻¹[tracer_name]

    launch!(arch, grid, params, _assemble_diffusive_dissipation!, K, grid, χ, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ)

    return nothing
end