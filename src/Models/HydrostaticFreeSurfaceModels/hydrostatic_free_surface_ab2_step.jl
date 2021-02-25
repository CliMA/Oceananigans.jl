using Oceananigans.TimeSteppers: ab2_step_field!

import Oceananigans.TimeSteppers: ab2_step!

function ab2_step!(model::HydrostaticFreeSurfaceModel, Δt, χ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    step_field_kernel! = ab2_step_field!(device(model.architecture), workgroup, worksize)

    # Launch velocity update kernels

    velocities_events = []

    for name in (:u, :v)
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        velocity_field = model.velocities[name]

        event = step_field_kernel!(velocity_field, Δt, χ, Gⁿ, G⁻,
                                   dependencies=Event(device(model.architecture)))

        push!(velocities_events, event)
    end

    # Launch tracer update kernels

    tracer_events = []

    for name in tracernames(model.tracers)
        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        tracer_field = model.tracers[name]

        event = step_field_kernel!(tracer_field, Δt, χ, Gⁿ, G⁻,
                                   dependencies=Event(device(model.architecture)))

        push!(tracer_events, event)
    end

    velocities_update = MultiEvent(Tuple(velocities_events))

    # Update the free surface if not using a rigid lid once the velocities have finished updating.
    free_surface_event = ab2_step_free_surface!(model.free_surface, velocities_update, model, χ, Δt)

    wait(device(model.architecture), MultiEvent(tuple(free_surface_event, tracer_events...)))

    return nothing
end

#####
##### Free surface time-stepping: explicit, implicit, rigid lid ?
#####

ab2_step_free_surface!(free_surface::ExplicitFreeSurface, velocities_update, model, χ, Δt) =
    explicit_ab2_step_free_surface!(free_surface, velocities_update, model, χ, Δt)

function explicit_ab2_step_free_surface!(free_surface, velocities_update, model, χ, Δt)

    event = launch!(model.architecture, model.grid, :xy,
                    _ab2_step_free_surface!,
                    model.free_surface.η,
                    χ,
                    Δt,
                    model.timestepper.Gⁿ.η,
                    model.timestepper.G⁻.η,
                    dependencies=Event(device(model.architecture)))

    return event
end

@kernel function _ab2_step_free_surface!(η, χ::FT, Δt, Gηⁿ, Gη⁻) where FT
    i, j = @index(Global, NTuple)

    @inbounds begin
        η[i, j, 1] += Δt * ((FT(1.5) + χ) * Gηⁿ[i, j, 1] - (FT(0.5) + χ) * Gη⁻[i, j, 1])
    end
end


function ab2_step_free_surface!(free_surface::ImplicitFreeSurface, velocities_update, model, χ, Δt)

    ##### Implicit solver for η
    
    ## Need to wait for u* and v* to finish
    ### wait(device(model.architecture), velocities_update)

    ## Note Jean-Michel is a fan of doing ExplicitFreeSurface step before solve, so maybe this code is part of free_surface::ExplicitFreeSurface
    ## that comes after explicit step
    event = explicit_ab2_step_free_surface!(free_surface, velocities_update, model, χ, Δt)
    wait(device(model.architecture), event)

    ## We need vertically integrated U,V (see continuity bits in src/Models/HydrostaticFreeSurfaceModels/compute_w_from_continuity.jl), 
    ## model.free_surface.η, g and Δt and grid.... 
    event = compute_vertcally_integrated_transport!(free_surface, model)

    ## Then we can invoke solve_for_pressure! on the right type via calculate_pressure_correction!

    ## Once we have η we can update u* and v* with pressure gradient just as in pressure_correct_velocities!

    ## The explicit form of this function defaults to returning an event, we do the same for now.
    return event
end


using Oceananigans.Architectures: device
using Oceananigans.Operators: ΔzC

"""
Compute the vertical integrated transport from the bottom to z=0 (i.e. linear free-surface)

    `U^{*} = ∫ [(u^{*})] dz`
    `V^{*} = ∫ [(v^{*})] dz`
"""
### Note - what we really want is RHS = divergence of the vertically integrated transport
###        we can optimize this a bit later to do this all in one go to save using intermediate variables.
function compute_vertcally_integrated_transport!(free_surface, model)

    event = launch!(model.architecture,
                    model.grid,
                    :xy,
                    _compute_vertically_integrated_transport!,
                    model.velocities,
                    model.grid,
                    free_surface.barotropic_transport,
                    dependencies=Event(device(model.architecture)))

    ## wait(device(model.architecture), event)

    return event
end

@kernel function _compute_vertically_integrated_transport!(U, grid, barotropic_transport )
    i, j = @index(Global, NTuple)
    # U.w[i, j, 1] = 0 is enforced via halo regions.
    barotropic_transport.u[i, j, 1] = 0.
    barotropic_transport.v[i, j, 1] = 0.
    @unroll for k in 1:grid.Nz
        #### @inbounds barotropic_transport.u[i, j, 1] += U.u[i, j, k-1]*Δyᶠᶜᵃ(i, j, k, grid)*Δzᵃᵃᶜ(i, j, k, grid)
        #### @inbounds barotropic_transport.v[i, j, 1] += U.v[i, j, k-1]*Δyᶠᶜᵃ(i, j, k, grid)*Δzᵃᵃᶜ(i, j, k, grid)
        @inbounds barotropic_transport.u[i, j, 1] += U.u[i, j, k]*Δyᶠᶜᵃ(i, j, k, grid)*ΔzC(i, j, k, grid)
        @inbounds barotropic_transport.v[i, j, 1] += U.v[i, j, k]*Δyᶠᶜᵃ(i, j, k, grid)*ΔzC(i, j, k, grid)
    end
end

