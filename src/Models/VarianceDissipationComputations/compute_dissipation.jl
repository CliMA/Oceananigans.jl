@inline c★(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] + cⁿ[i, j, k]) / 2
@inline c²(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] * cⁿ[i, j, k])

"""
$(TYPEDSIGNATURES)

Compute the numerical dissipation for tracer `tracer_name`, from the previously calculated advective and diffusive fluxes,
the formulation is:

    A = 2 * δc★ * F - U δc²  # For advective dissipation
    D = 2 * δc★ * F          # For diffusive dissipation

Where ``F'' is the flux associated with the particular process,``U'' is the advecting volume transport,
while ``c★'' and ``c²'' are functions defined above.
Note that ``F'' and ``U'' need to be numerically accurate for the budgets to close,
i.e. for the AB2 scheme:

    F = 1.5 Fⁿ - 0.5 Fⁿ⁻¹
    U = 1.5 Uⁿ - 0.5 Uⁿ⁻¹

For a split Runge-Kutta scheme, ``F'' and ``U'' are those of the last stage.
The vertical flux of vertically implicit closures is evaluated with cⁿ⁺¹.
"""
function compute_dissipation!(dissipation, model, tracer_name)

    grid = model.grid

    # General velocities
    Uⁿ   = dissipation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation.previous_state.Uⁿ⁻¹

    cⁿ⁺¹ = model.tracers[tracer_name]
    cⁿ   = dissipation.previous_state.cⁿ⁻¹

    scheme = getadvection(model.advection, tracer_name)

    ####
    #### Assemble the advective dissipation
    ####

    P    = dissipation.advective_production
    Fⁿ   = dissipation.advective_fluxes.Fⁿ
    Fⁿ⁻¹ = dissipation.advective_fluxes.Fⁿ⁻¹

    !(scheme isa Nothing) &&
        assemble_advective_dissipation!(P, grid, model.timestepper, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)

    ####
    #### Assemble the diffusive dissipation
    ####

    K    = dissipation.diffusive_production
    Vⁿ   = dissipation.diffusive_fluxes.Vⁿ
    Vⁿ⁻¹ = dissipation.diffusive_fluxes.Vⁿ⁻¹
    κᶻ   = dissipation.diffusive_fluxes.κᶻ

    assemble_diffusive_dissipation!(K, grid, model.timestepper, Vⁿ, Vⁿ⁻¹, κᶻ, cⁿ⁺¹, cⁿ)

    return nothing
end

assemble_advective_dissipation!(P, grid, ts::QuasiAdamsBashforth2TimeStepper, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ) =
    launch!(architecture(grid), grid, :xyz, _assemble_ab2_advective_dissipation!, P, grid, ts.χ, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)

assemble_advective_dissipation!(P, grid, ::SplitRungeKuttaTimeStepper, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ) =
    launch!(architecture(grid), grid, :xyz, _assemble_rk3_advective_dissipation!, P, grid, Fⁿ, Uⁿ, cⁿ⁺¹, cⁿ)

assemble_diffusive_dissipation!(K, grid, ts::QuasiAdamsBashforth2TimeStepper, Vⁿ, Vⁿ⁻¹, κᶻ, cⁿ⁺¹, cⁿ) =
    launch!(architecture(grid), grid, :xyz, _assemble_ab2_diffusive_dissipation!, K, grid, ts.χ, Vⁿ, Vⁿ⁻¹, κᶻ, cⁿ⁺¹, cⁿ)

assemble_diffusive_dissipation!(K, grid, ::SplitRungeKuttaTimeStepper, Vⁿ, Vⁿ⁻¹, κᶻ, cⁿ⁺¹, cⁿ) =
    launch!(architecture(grid), grid, :xyz, _assemble_rk3_diffusive_dissipation!, K, grid, Vⁿ, κᶻ, cⁿ⁺¹, cⁿ)
