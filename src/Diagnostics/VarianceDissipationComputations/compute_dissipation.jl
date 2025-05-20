function compute_dissipation!(model, dissipation)

    for tracer_name in keys(dissipation.advective_production)
        compute_dissipation!(dissipation, model, tracer_name)
    end

    return nothing
end

@inline c★(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] + cⁿ[i, j, k]) / 2
@inline c²(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] * cⁿ[i, j, k])

"""
    compute_dissipation!(dissipation, model, tracer_name::Symbol)

Compute the numerical dissipation for tracer `tracer_name`, from the previously calculated advective and diffusive fluxes, 
the formulation is:

A = 2 * δc★ * F - U δc²  # For advective dissipation 
D = 2 * δc★ * F          # For diffusive dissipation

Where ``F'' is the flux associated with the particular process,``U'' is the adecting velocity,
while ``c★'' and ``c²'' are functions defined above.
Note that ``F'' and ``U'' need to be numerically accurate for the budgets to close,
i.e. for and AB2 scheme:

F = 1.5 Fⁿ - 0.5 Fⁿ⁻¹
U = 1.5 Uⁿ - 0.5 Uⁿ⁻¹

For an RK3 method (not implemented at the moment), the whole substepping procedure needs to be accounted for.
"""
function compute_dissipation!(dissipation, model, tracer_name::Symbol)
    
    grid = model.grid
    arch = architecture(grid)
    χ = model.timestepper.χ

    # General velocities
    Uⁿ⁺¹ = model.velocities
    Uⁿ   = dissipation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation.previous_state.Uⁿ⁻¹

    cⁿ⁺¹ = model.tracers[tracer_name]
    cⁿ   = dissipation.previous_state[tracer_name]
    
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

    launch!(arch, grid, :xyz, _assemble_diffusive_dissipation!, K, grid, χ, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ)

    return nothing
end