function cell_advection_timescale(u, v, w, grid)
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    Δx = model.grid.Δx
    Δy = model.grid.Δy
    Δz = model.grid.Δz

    return min(Δx/umax, Δy/vmax, Δz/wmax)
end

cell_advection_timescale(model) =
    cell_advection_timescale(model.velocities.u.data.parent,
                             model.velocities.v.data.parent,
                             model.velocities.w.data.parent,
                             model.grid)

"""
    TimeStepWizard(cfl=0.1, max_change=2.0, min_change=0.5, max_Δt=Inf, kwargs...)

Instantiate a `TimeStepWizard`. On calling `update_Δt!(wizard, model)`,
the `TimeStepWizard` computes a time-step such that 
`cfl = max(u/Δx, v/Δy, w/Δz) Δt`, where `max(u/Δx, v/Δy, w/Δz)` is the 
maximum ratio between model velocity and along-velocity grid spacing 
anywhere on the model grid. The new `Δt` is constrained to change by a 
multiplicative factor no more than `max_change` or no less than 
`min_change` from the previous `Δt`, and to be no greater in absolute 
magnitude than `max_Δt`. 
"""
Base.@kwdef mutable struct TimeStepWizard{T}
              cfl :: T = 0.1
    cfl_diffusion :: T = 2e-2
       max_change :: T = 2.0
       min_change :: T = 0.5
           max_Δt :: T = Inf
               Δt :: T = 0.01
end


"""
    update_Δt!(wizard, model)

Compute `wizard.Δt` given the velocities and diffusivities
of `model`, and the parameters of `wizard`.
"""
function update_Δt!(wizard, model)
    Δt = wizard.cfl * cell_advection_timescale(model)

    # Put the kibosh on if needed
    Δt = min(wizard.max_change * wizard.Δt, Δt)
    Δt = max(wizard.min_change * wizard.Δt, Δt)
    Δt = min(wizard.max_Δt, Δt)

    wizard.Δt = Δt

    return nothing
end
