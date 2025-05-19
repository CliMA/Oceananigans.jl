using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper, SplitRungeKutta3TimeStepper

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
function compute_dissipation!(model, dissipation, tracer_name::Symbol)
    
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

const RungeKuttaScheme = Union{RungeKutta3TimeStepper, SplitRungeKutta3TimeStepper}

assemble_advective_dissipation!(P, grid, ts::QuasiAdamsBashforth2TimeStepper, substep, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ) = 
    launch!(architecture(grid), grid, :xyz, _assemble_ab2_advective_dissipation!, P, grid, ts.χ, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ)

function assemble_advective_dissipation!(P, grid, ::RungeKuttaScheme, substep, Fⁿ, Fⁿ⁻¹, Uⁿ, Uⁿ⁻¹, cⁿ⁺¹, cⁿ) 
    if substep == 3
        launch!(architecture(grid), grid, :xyz, _assemble_rk3_advective_dissipation!, P, grid, Fⁿ, Uⁿ, cⁿ⁺¹, cⁿ)
    end
    return nothing
end

assemble_diffusive_dissipation!(K, grid, ts::QuasiAdamsBashforth2TimeStepper, substep, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ) = 
    launch!(architecture(grid), grid, :xyz, _assemble_ab2_diffusive_dissipation!, K, grid, ts.χ, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ)

function assemble_diffusive_dissipation!(K, grid, ::RungeKuttaScheme, substep, Vⁿ, Vⁿ⁻¹, cⁿ⁺¹, cⁿ) 
    if substep == 3
        launch!(architecture(grid), grid, :xyz, _assemble_rk3_diffusive_dissipation!, K, grid, Vⁿ, cⁿ⁺¹, cⁿ)
    end
    return nothing
end
