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
    wait(device(model.architecture), velocities_update)
    fill_halo_regions!(model.velocities, model.architecture, model.clock, fields(model) )

    η_save = deepcopy(free_surface.η)
    ## event = explicit_ab2_step_free_surface!(free_surface, velocities_update, model, χ, Δt)
    ## wait(device(model.architecture), event)

    ## We need vertically integrated U,V
    event = compute_vertically_integrated_volume_flux!(free_surface, model)
    wait(device(model.architecture), event)
    u=free_surface.barotropic_volume_flux.u
    v=free_surface.barotropic_volume_flux.v
    fill_halo_regions!(u.data ,u.boundary_conditions, model.architecture, model.grid, model.clock, fields(model) )
    fill_halo_regions!(v.data ,v.boundary_conditions, model.architecture, model.grid, model.clock, fields(model) )
    ### We don't need the halo below, its just here for some debugging
    Ax=free_surface.vertically_integrated_lateral_face_areas.Ax
    Ay=free_surface.vertically_integrated_lateral_face_areas.Ay
    fill_halo_regions!(Ax.data ,Ax.boundary_conditions, model.architecture, model.grid, model.clock, fields(model) )
    fill_halo_regions!(Ay.data ,Ay.boundary_conditions, model.architecture, model.grid, model.clock, fields(model) )


    ## Compute volume scaled divergence of the barotropic transport and put into solver RHS
    event = compute_volume_scaled_divergence!(free_surface, model)
    wait(device(model.architecture), event)
    
    ## Include surface pressure term into RHS
    RHS = free_surface.implicit_step_solver.solver.settings.RHS
    RHS .= RHS/(model.free_surface.gravitational_acceleration*Δt)
    ## fill_halo_regions!(RHS.data ,RHS.boundary_conditions, model.architecture, model.grid, model.clock, fields(model) )
    η = free_surface.η
    η = η_save
    fill_halo_regions!(RHS   , η.boundary_conditions, model.architecture, model.grid)
    fill_halo_regions!(η.data, η.boundary_conditions, model.architecture, model.grid)
    ##  need to subtract Azᵃᵃᵃ(i, j, 1, grid)*η[i,j, 1]/(g*Δt^2)
    event = add_previous_free_surface_contribution(free_surface, model, Δt )
    wait(device(model.architecture), event)
    fill_halo_regions!(RHS   , η.boundary_conditions, model.architecture, model.grid)
    ## RHS .= RHS .+ free_surface.η.data/Δt

    ## Then we can invoke solve_for_pressure! on the right type via calculate_pressure_correction!
    x  = free_surface.implicit_step_solver.solver.settings.x
    x .= η.data
    fill_halo_regions!(x ,η.boundary_conditions, model.architecture, model.grid)
    solve_poisson_equation!(free_surface.implicit_step_solver.solver, RHS, x; Δt=Δt, g=free_surface.gravitational_acceleration)
    ## exit()
    fill_halo_regions!(x ,η.boundary_conditions, model.architecture, model.grid)
    free_surface.η.data .= x

    ## Once we have η we can update u* and v* with pressure gradient just as in pressure_correct_velocities!

    ## The explicit form of this function defaults to returning an event, we do the same for now.
    return event
end
