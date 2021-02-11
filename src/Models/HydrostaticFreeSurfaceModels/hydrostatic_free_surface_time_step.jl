using Oceananigans.TimeSteppers: store_tendencies!, update_particle_properties!, ab2_step_field!

import Oceananigans.TimeSteppers: time_step!, tick!, ab2_step!

"""
    time_step!(model::HydrostaticFreeSurfaceModel, Δt; euler=false)

Step forward `model` one time step `Δt` with a 2nd-order Adams-Bashforth method and
pressure-correction substep. Setting `euler=true` will take a forward Euler time step.
"""
function time_step!(model::HydrostaticFreeSurfaceModel, Δt; euler=false)

    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    χ = ifelse(euler, convert(eltype(model.grid), -0.5), model.timestepper.χ)

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model)

    calculate_tendencies!(model)

    ab2_step!(model, Δt, χ) # full step for tracers, fractional step for velocities.

    calculate_pressure_correction!(model, Δt)
    pressure_correct_velocities!(model, Δt)

    update_state!(model)
    store_tendencies!(model)
    update_particle_properties!(model, Δt)

    tick!(model.clock, Δt)

    return nothing
end

#####
##### Time stepping in each step
#####

function ab2_step!(model::HydrostaticFreeSurfaceModel, Δt, χ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    step_field_kernel! = ab2_step_field!(device(model.architecture), workgroup, worksize)

    events = []

    prognostic_fields = fields(model)
    three_dimensional_field_names = tuple(:u, :v, propertynames(model.tracers)...)

    for name in three_dimensional_field_names

        Gⁿ = model.timestepper.Gⁿ[name]
        G⁻ = model.timestepper.G⁻[name]
        field = prognostic_fields[name]

        field_event = step_field_kernel!(field, Δt, χ, Gⁿ, G⁻,
                                         dependencies=Event(device(model.architecture)))

        push!(events, field_event)
    end

    free_surface_event = ab2_step_free_surface!(model.free_surface, model.architecture, model.grid, model.timestepper, Δt, χ)

    push!(events, free_surface_event)

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    return nothing
end

function ab2_step_free_surface!(free_surface::ExplicitFreeSurface, arch, grid, timestepper, Δt, χ)

    event = launch!(arch, grid, :xy,
                    _ab2_step_field!,
                    free_surface.η,
                    Δt,
                    χ,
                    timestepper.Gⁿ.η,
                    timestepper.G⁻.η,
                    dependencies = Event(device(arch)))

    return event
end

@kernel function _ab2_step_free_surface!(η, Δt, χ::FT, Gηⁿ, Gη⁻) where FT
    i, j = @index(Global, NTuple)

    @inbounds begin
        η[i, j, 1] += Δt * ( (FT(1.5) + χ) * Gηⁿ[i, j, 1] - (FT(0.5) + χ) * Gη⁻[i, j, 1] )
    end
end
