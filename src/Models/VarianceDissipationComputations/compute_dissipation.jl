@inline c★(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] + cⁿ[i, j, k]) / 2
@inline c²(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] * cⁿ[i, j, k])

"""
$(TYPEDSIGNATURES)

Compute the numerical dissipation for tracer `tracer_name` from the cached advective and diffusive fluxes:

    A = 2 * δc★ * F - U δc²  # For advective dissipation
    D = 2 * δc★ * F          # For diffusive dissipation

where `F` is the flux associated with the particular process, `U` the transport that advects the tracer, and `c★`
and `c²` the functions defined above. `F` and `U` are those that advanced the tracer over the step: for the AB2
scheme

    F = (1.5 + χ) Fⁿ - (0.5 + χ) Fⁿ⁻¹
    U = (1.5 + χ) Uⁿ - (0.5 + χ) Uⁿ⁻¹

and for the Runge-Kutta schemes the flux and transport of the final stage, or their quadrature over the stages.
"""
function compute_dissipation!(dissipation, model, tracer_name)

    grid = model.grid

    # General velocities
    Uⁿ   = dissipation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation.previous_state.Uⁿ⁻¹

    cⁿ⁺¹ = model.tracers[tracer_name]
    cⁿ   = dissipation.previous_state.cⁿ⁻¹

    substep = model.clock.stage
    scheme  = getadvection(model.advection, tracer_name)

    ####
    #### Assemble the advective dissipation
    ####

    P    = dissipation.advective_production
    Fⁿ   = dissipation.advective_fluxes.Fⁿ
    Fⁿ⁻¹ = dissipation.advective_fluxes.Fⁿ⁻¹

    !(scheme isa Nothing) &&
        assemble_advective_dissipation!(P, grid, model.timestepper, substep, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)

    ####
    #### Assemble the diffusive dissipation
    ####

    K    = dissipation.diffusive_production
    Vⁿ   = dissipation.diffusive_fluxes.Vⁿ
    Vⁿ⁻¹ = dissipation.diffusive_fluxes.Vⁿ⁻¹

    assemble_diffusive_dissipation!(K, grid, model.timestepper, substep, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ)

    return nothing
end

function assemble_advective_dissipation!(P, grid, ts::QuasiAdamsBashforth2TimeStepper, substep, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)
    FT = eltype(grid)
    C₁ = convert(FT, 1.5 + ts.χ)
    C₂ = convert(FT, 0.5 + ts.χ)
    launch!(architecture(grid), grid, :xyz, _assemble_advective_dissipation!, P, grid, C₁, C₂, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)
    return nothing
end

function assemble_advective_dissipation!(P, grid, ts::MultiStageTimeStepper, substep, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)
    if substep == ts.Nstages
        FT = eltype(grid)
        launch!(architecture(grid), grid, :xyz, _assemble_advective_dissipation!, P, grid, one(FT), zero(FT),
                Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)
    end
    return nothing
end

assemble_diffusive_dissipation!(K, grid, ts::QuasiAdamsBashforth2TimeStepper, substep, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ) =
    launch!(architecture(grid), grid, :xyz, _assemble_ab2_diffusive_dissipation!, K, grid, ts.χ, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ)

function assemble_diffusive_dissipation!(K, grid, ts::SplitRungeKuttaTimeStepper, substep, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ)
    if substep == ts.Nstages
        launch!(architecture(grid), grid, :xyz, _assemble_rk3_diffusive_dissipation!, K, grid, Vⁿ, cⁿ⁺¹, cⁿ)
    end
    return nothing
end

# As for the advective production: the β-weighted sum of the stage fluxes, at the endpoints of the step.
function assemble_diffusive_dissipation!(K, grid, ts::SSPRungeKuttaTimeStepper, substep, Vⁿ, V̄, cⁿ⁺¹, cⁿ)
    if substep == ts.Nstages
        launch!(architecture(grid), grid, :xyz, _assemble_rk3_diffusive_dissipation!, K, grid, V̄, cⁿ⁺¹, cⁿ)
    end
    return nothing
end
