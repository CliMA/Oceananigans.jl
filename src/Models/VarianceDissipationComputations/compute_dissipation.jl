@inline c★(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] + cⁿ[i, j, k]) / 2
@inline c²(i, j, k, grid, cⁿ⁺¹, cⁿ) = @inbounds (cⁿ⁺¹[i, j, k] * cⁿ[i, j, k])

"""
$(TYPEDSIGNATURES)

Compute the numerical dissipation for tracer `tracer_name` from the cached advective and diffusive fluxes:

    A = 2 * δc★ * F - U δc²  # For advective dissipation
    D = 2 * δc★ * F          # For diffusive dissipation

where `F` is the flux associated with the particular process, `U` the volume transport that advects the tracer, and
`c★` and `c²` the functions defined above. `F` and `U` are those that advanced the tracer over the step: for the AB2
scheme

    F = (1.5 + χ) Fⁿ - (0.5 + χ) Fⁿ⁻¹
    U = (1.5 + χ) Uⁿ - (0.5 + χ) Uⁿ⁻¹

for a split Runge-Kutta scheme those of the last stage, and for an SSP Runge-Kutta scheme their quadrature over the
stages. The vertical flux of vertically implicit closures is evaluated with cⁿ⁺¹.
"""
function compute_dissipation!(dissipation, model, tracer_name)

    grid = model.grid
    arch = architecture(grid)

    Uⁿ   = dissipation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation.previous_state.Uⁿ⁻¹

    cⁿ⁺¹ = model.tracers[tracer_name]
    cⁿ   = dissipation.previous_state.cⁿ⁻¹

    scheme = getadvection(model.advection, tracer_name)
    C₁, C₂ = step_weights(model.timestepper, eltype(grid))

    ####
    #### Assemble the advective dissipation
    ####

    P    = dissipation.advective_production
    Fⁿ   = dissipation.advective_fluxes.Fⁿ
    Fⁿ⁻¹ = dissipation.advective_fluxes.Fⁿ⁻¹

    !(scheme isa Nothing) &&
        launch!(arch, grid, :xyz, _assemble_advective_dissipation!, P, grid, C₁, C₂, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)

    ####
    #### Assemble the diffusive dissipation
    ####

    K    = dissipation.diffusive_production
    Vⁿ   = dissipation.diffusive_fluxes.Vⁿ
    Vⁿ⁻¹ = dissipation.diffusive_fluxes.Vⁿ⁻¹
    κᶻ   = dissipation.diffusive_fluxes.κᶻ

    launch!(arch, grid, :xyz, _assemble_diffusive_dissipation!, K, grid, C₁, C₂, Vⁿ, Vⁿ⁻¹, κᶻ, cⁿ⁺¹, cⁿ)

    return nothing
end

# The weights of the fluxes of the current and of the previous step in the fluxes that advanced the tracer
step_weights(ts::QuasiAdamsBashforth2TimeStepper, FT) = (convert(FT, 1.5 + ts.χ), convert(FT, 0.5 + ts.χ))
step_weights(::MultiStageTimeStepper, FT) = (one(FT), zero(FT))
