Δmin(grid::RegularCartesianGrid) = min(grid.Δx, grid.Δy, grid.Δz)

function Umax(model)
    u = model.velocities.u.data.parent
    v = model.velocities.v.data.parent
    w = model.velocities.w.data.parent

    u_max = maximum(u)
    v_max = maximum(v)
    w_max = maximum(w)

    max(u_max, v_max, w_max)
end

"""
    get_cfl(Δt, model)

Compute the maximum Courant number given a `model` state and time step `Δt`
assuming the Courant–Friedrichs–Lewy (CFL) condition uΔt/Δx <= 1.
"""
get_cfl(Δt, model) = Δt * Umax(model) / Δmin(model.grid)


"""
    cfl_Δt(model, cfl, max_Δt)

Compute the maximum allowable time step for a `model` by the Courant–Friedrichs–Lewy (CFL)
condition given a Courant number `cfl`. Will never return a time step greater than `max_Δt`.
"""
function cfl_Δt(model, cfl, max_Δt)
    τ = Δmin(model.grid) / Umax(model)
    return min(max_Δt, cfl*τ)
end

"""
    safe_Δt(model, αu, αν=0.01)

Compute a safe time step for a `model`.
"""
function safe_Δt(model, αu, αν=0.01)
    τu = Δmin(model.grid) / Umax(model)
    τν = Δmin(model.grid)^2 / model.closure.ν

    return min(αν*τν, αu*τu)
end

function cell_advection_timescale(u, v, w, grid)
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    Δx = model.grid.Δx
    Δy = model.grid.Δy
    Δz = model.grid.Δz

    return min(Δx/umax, Δy/vmax, Δz/wmax)
end

function cell_diffusion_timescale(ν, κ, grid)
    νmax = maximum(abs, ν)
    κmax = maximum(abs, κ)

    Δ = Δmin(model.grid) # assuming diffusion is isotropic for now

    return min(Δ^2/νmax, Δ^2/κmax)

end

cell_advection_timescale(model) =
    cell_advection_timescale(model.velocities.u.data.parent,
                             model.velocities.v.data.parent,
                             model.velocities.w.data.parent,
                             model.grid)

cell_diffusion_timescale(model) =
    cell_diffusion_timescale(model.closure.ν,
                             model.closure.κ,
                             model.grid)

"""
    TimeStepWizard(cfl=0.1, max_change=2.0, min_change=0.5, max_Δt=Inf, kwargs...)

Instantiate a `TimeStepWizard`. On calling `update_Δt!(wizard, model)`,
the `TimeStepWizard` computes a time-step such that the Courant-Freidrichs-Levy
number is equal to `cfl`. The new `Δt` is constrained to change by a multiplicative
factor no more than `max_change` or no less than `min_change` from the previous 
`Δt`, and to be no greater in absolute magnitude than `max_Δt`. 
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
    Δt_advection = wizard.cfl           * cell_advection_timescale(model)
    Δt_diffusion = wizard.cfl_diffusion * cell_diffusion_timescale(model)

    # Desired Δt
    Δt = min(Δt_advection, Δt_diffusion)

    # Put the kibosh on if needed
    Δt = min(wizard.max_change * wizard.Δt, Δt)
    Δt = max(wizard.min_change * wizard.Δt, Δt)
    Δt = min(wizard.max_Δt, Δt)

    wizard.Δt = Δt

    return nothing
end
